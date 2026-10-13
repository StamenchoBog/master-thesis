"""SISA client: S shards x R slices, one constituent per shard, a
checkpoint after every slice.

Constituents never take the global weights (that would carry the poison
into every checkpoint), and the FL update is their parameter average.
Rounds are counted locally (state.json), because Phase 4 is a new `flwr
run` whose rounds restart at 1.
"""

import json
import os
import time

import numpy as np
import torch
from flwr.common import Context
from torch.utils.data import DataLoader, TensorDataset

from .client_app import DEVICE, SEED, FlowerClient, seed_everything
from .data_loader import load_arrays, load_data
from .model import IDSModel
from .sisa_partition import shard_slice_assignment

NUM_SHARDS = int(os.getenv("NUM_SHARDS", "5"))
NUM_SLICES = int(os.getenv("NUM_SLICES", "5"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/checkpoints")
TELEMETRY_DIR = os.getenv("TELEMETRY_DIR", "/dev/shm")


def ckpt_path(shard: int, rnd: int, slc: int) -> str:
    return os.path.join(
        CHECKPOINT_DIR, f"shard{shard}_round{rnd}_slice{slc}.pt"
    )


def save_checkpoint(
    path: str, model, optimizer, rnd: int, slc: int
) -> tuple[float, int]:
    """Returns (seconds, bytes); fsync, because checkpoint I/O is a
    measured quantity."""
    start = time.perf_counter()
    with open(path, "wb") as f:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "round": rnd,
                "slice": slc,
            },
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - start, os.path.getsize(path)


def slice_loader(X, y, idx, seed_key: int, batch_size: int = 512):
    """seed_key fixes the batch order, so a replay is exact."""
    gen = torch.Generator().manual_seed(seed_key)
    return DataLoader(
        TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx])),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )


def train_slice(model, optimizer, loader, criterion):
    model.train()
    total, batches = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE).float()
        optimizer.zero_grad()
        loss = criterion(model(Xb).squeeze(), yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total, batches


def batch_seed(seed: int, rnd: int, shard: int, slc: int) -> int:
    return seed * 1_000_000 + rnd * 10_000 + shard * 100 + slc


class SISATrainer:

    def __init__(self, partition_id: int, num_partitions: int):
        X, y, split, poison_idx = load_arrays(partition_id, num_partitions)
        self.X, self.y = X[:split], y[:split]
        self.assignment = shard_slice_assignment(
            len(self.X), SEED, NUM_SHARDS, NUM_SLICES
        )
        if os.getenv("POISON_MODE", "off") == "drop":
            # Drop poisoned rows per slice rather than from X, so
            # indices stay positional.
            self.assignment = [
                [np.setdiff1d(idx, poison_idx) for idx in shard]
                for shard in self.assignment
            ]
            print(
                "[SISA] POISON DROPPED from slice assignment "
                f"({len(poison_idx)} rows)"
            )
        self.input_dim = X.shape[1]
        self.criterion = torch.nn.BCELoss()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.state_file = os.path.join(CHECKPOINT_DIR, "state.json")
        self.round = self._load_round()
        self.models = [self._init_or_resume(s) for s in range(NUM_SHARDS)]

    def _load_round(self) -> int:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)["completed_rounds"]
        return 0

    def _init_or_resume(self, shard: int):
        """Fresh seeded constituent, or its latest checkpoint when
        rejoining in Phase 4."""
        torch.manual_seed(SEED * 1000 + shard)
        model = IDSModel(input_dim=self.input_dim).to(DEVICE)
        if self.round > 0:
            ckpt = ckpt_path(shard, self.round, NUM_SLICES - 1)
            model.load_state_dict(
                torch.load(ckpt, map_location=DEVICE)["model"]
            )
        return model

    def train_round(self, lr: float, telemetry) -> float:
        rnd = self.round + 1
        # Dropout uses the global torch RNG; re-seed per round so
        # results don't depend on how long the process has been running.
        torch.manual_seed(SEED * 7919 + rnd)
        losses, batches = 0.0, 0
        for shard, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for slc, idx in enumerate(self.assignment[shard]):
                loader = slice_loader(
                    self.X, self.y, idx, batch_seed(SEED, rnd, shard, slc)
                )
                t0 = time.perf_counter()
                loss, n_batches = train_slice(
                    model, optimizer, loader, self.criterion
                )
                train_s = time.perf_counter() - t0
                ckpt_s, ckpt_bytes = save_checkpoint(
                    ckpt_path(shard, rnd, slc), model, optimizer, rnd, slc
                )
                telemetry(
                    {
                        "round": rnd,
                        "shard": shard,
                        "slice": slc,
                        "train_s": round(train_s, 4),
                        "ckpt_s": round(ckpt_s, 4),
                        "ckpt_bytes": ckpt_bytes,
                    }
                )
                losses += loss
                batches += n_batches
        self.round = rnd
        with open(self.state_file, "w") as f:
            json.dump({"completed_rounds": rnd}, f)
        return losses / max(batches, 1)

    def averaged_parameters(self):
        keys = self.models[0].state_dict().keys()
        stacked = {
            k: torch.stack(
                [m.state_dict()[k].float() for m in self.models]
            ).mean(0)
            for k in keys
        }
        return [v.cpu().numpy() for v in stacked.values()]

    @property
    def n_samples(self) -> int:
        return len(self.X)


_trainer = None


class SISAClient(FlowerClient):

    def __init__(self, trainer, eval_model, valloader):
        super().__init__(
            eval_model, None, valloader
        )  # no trainloader: fit is overridden
        self.trainer = trainer
        self.telemetry_file = os.path.join(TELEMETRY_DIR, "sisa_timings.jsonl")

    def _log(self, entry):
        entry["ts"] = time.time()
        with open(self.telemetry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def fit(self, parameters, config):
        train_loss = self.trainer.train_round(
            float(config.get("lr", 0.001)), self._log
        )
        metrics = {"train_loss": train_loss}
        return (
            self.trainer.averaged_parameters(),
            self.trainer.n_samples,
            metrics,
        )


def client_fn(context: Context):
    global _trainer
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    seed_everything(SEED + partition_id)
    if _trainer is None:
        _trainer = SISATrainer(partition_id, num_partitions)
    _, valloader, input_dim = load_data(partition_id, num_partitions)
    eval_model = IDSModel(input_dim=input_dim).to(DEVICE)
    return SISAClient(_trainer, eval_model, valloader).to_client()

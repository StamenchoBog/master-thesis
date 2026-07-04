"""SISA client (Arm B): sharded, isolated, sliced training with checkpoint rollback.

Adaptation of SISA (Bourtoule et al., 2021) to a federated client:
- The local train set is split into S shards x R slices (seeded, shared with
  prepare_edge_data.py via sisa_partition.py).
- One constituent model per shard, trained ONLY on its shard — constituents
  never absorb the broadcast global weights, otherwise the rollback checkpoints
  would inherit poison influence via the global model and the local unlearning
  guarantee would be lost. Documented deviation from vanilla FL.
- The FL update is the parameter average of the constituents (deviation from
  vanilla SISA's prediction ensembling, required to fit FedAvg).
- After every slice, the constituent + optimizer state is checkpointed to the
  SD card with fsync — checkpoint I/O is a measured experimental quantity.
- Rounds are counted locally (state.json) because Phase 4 is a separate
  `flwr run` whose server_round restarts at 1.

Timings go to TELEMETRY_DIR (RAM) as JSONL; FL fit metrics carry only
train_loss because the server's _weighted_average requires identical metric
keys across all clients.
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
    return os.path.join(CHECKPOINT_DIR, f"shard{shard}_round{rnd}_slice{slc}.pt")


def save_checkpoint(path: str, model, optimizer, rnd: int, slc: int) -> tuple[float, int]:
    """Persist constituent + optimizer state with fsync; returns (seconds, bytes)."""
    start = time.perf_counter()
    with open(path, "wb") as f:
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(),
                    "round": rnd, "slice": slc}, f)
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - start, os.path.getsize(path)


def slice_loader(X, y, idx, seed_key: int, batch_size: int = 512):
    """Deterministic shuffled loader over one slice; seed_key fixes batch order so replay is exact."""
    gen = torch.Generator().manual_seed(seed_key)
    return DataLoader(
        TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx])),
        batch_size=batch_size, shuffle=True, generator=gen,
    )


def train_slice(model, optimizer, loader, criterion):
    """One epoch over a single slice; returns (mean loss, batches)."""
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
    """Unique deterministic seed per (round, shard, slice) — replay must reproduce batches exactly."""
    return seed * 1_000_000 + rnd * 10_000 + shard * 100 + slc


class SISATrainer:
    """Owns the S constituents, their data assignment, and the local round counter."""

    def __init__(self, partition_id: int, num_partitions: int):
        X, y, split, poison_idx = load_arrays(partition_id, num_partitions)
        self.X, self.y = X[:split], y[:split]
        self.assignment = shard_slice_assignment(len(self.X), SEED, NUM_SHARDS, NUM_SLICES)
        if os.getenv("POISON_MODE", "off") == "drop":
            # Phase 4 rejoin: keep positional indexing, remove poisoned rows per slice.
            self.assignment = [[np.setdiff1d(idx, poison_idx) for idx in shard]
                               for shard in self.assignment]
            print(f"[SISA] POISON DROPPED from slice assignment ({len(poison_idx)} rows)")
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
        """Fresh seeded constituent, or resume from its latest checkpoint (Phase 4 rejoin)."""
        torch.manual_seed(SEED * 1000 + shard)
        model = IDSModel(input_dim=self.input_dim).to(DEVICE)
        if self.round > 0:
            ckpt = ckpt_path(shard, self.round, NUM_SLICES - 1)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        return model

    def train_round(self, lr: float, telemetry) -> float:
        """Train every constituent one epoch over its shard, checkpointing per slice."""
        rnd = self.round + 1
        # Dropout draws from the global torch RNG; re-seed per round so results
        # don't depend on process lifetime / prior RNG consumption.
        torch.manual_seed(SEED * 7919 + rnd)
        losses, batches = 0.0, 0
        for shard, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # fresh per round, like the standard client
            for slc, idx in enumerate(self.assignment[shard]):
                loader = slice_loader(self.X, self.y, idx, batch_seed(SEED, rnd, shard, slc))
                t0 = time.perf_counter()
                loss, n_batches = train_slice(model, optimizer, loader, self.criterion)
                train_s = time.perf_counter() - t0
                ckpt_s, ckpt_bytes = save_checkpoint(ckpt_path(shard, rnd, slc), model, optimizer, rnd, slc)
                telemetry({"round": rnd, "shard": shard, "slice": slc,
                           "train_s": round(train_s, 4), "ckpt_s": round(ckpt_s, 4),
                           "ckpt_bytes": ckpt_bytes})
                losses += loss
                batches += n_batches
        self.round = rnd
        with open(self.state_file, "w") as f:
            json.dump({"completed_rounds": rnd}, f)
        return losses / max(batches, 1)

    def averaged_parameters(self):
        """Parameter average of all constituents — the client's FL update."""
        keys = self.models[0].state_dict().keys()
        stacked = {k: torch.stack([m.state_dict()[k].float() for m in self.models]).mean(0) for k in keys}
        return [v.cpu().numpy() for v in stacked.values()]

    @property
    def n_samples(self) -> int:
        return len(self.X)


_trainer = None


class SISAClient(FlowerClient):
    """FL client whose fit trains isolated constituents; evaluate is inherited (global model on local val)."""

    def __init__(self, trainer, eval_model, valloader):
        super().__init__(eval_model, None, valloader)  # trainloader unused: fit is overridden
        self.trainer = trainer
        self.telemetry_file = os.path.join(TELEMETRY_DIR, "sisa_timings.jsonl")

    def _log(self, entry):
        entry["ts"] = time.time()
        with open(self.telemetry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def fit(self, parameters, config):
        train_loss = self.trainer.train_round(float(config.get("lr", 0.001)), self._log)
        return self.trainer.averaged_parameters(), self.trainer.n_samples, {"train_loss": train_loss}


def client_fn(context: Context):
    """Instantiate the SISA client, keeping the trainer alive across fit calls."""
    global _trainer
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    seed_everything(SEED + partition_id)
    if _trainer is None:
        _trainer = SISATrainer(partition_id, num_partitions)
    _, valloader, input_dim = load_data(partition_id, num_partitions)
    eval_model = IDSModel(input_dim=input_dim).to(DEVICE)
    return SISAClient(_trainer, eval_model, valloader).to_client()

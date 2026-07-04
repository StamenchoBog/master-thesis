"""SISA unlearning (Arm B recovery): rollback + partial replay. Run on the Pi.

    python -m edge_nodes.sisa_recover

Procedure (timed end-to-end — this is the primary measurement window):
1. The oracle poison mask (poison_idx in the partition cache) identifies the
   compromised samples; the shard/slice assignment locates the affected
   constituent and the earliest poisoned slice l*.
2. The affected constituent rolls back to its checkpoint from round 1,
   slice l*-1 — the last state provably untouched by poison (later rounds
   carried poison influence forward, so their checkpoints are all tainted).
3. Replay: finish round 1 from slice l* and re-run rounds 2..NUM_ROUNDS on the
   cleaned shard (poisoned samples dropped). Untouched constituents keep their
   Phase-1 checkpoints — that is the S-fold saving over naive retraining.
4. The parameter average of all constituents is saved as the recovered model.

Labels on disk are clean (poison is applied at load time), so replay simply
drops the poisoned rows and trains on true labels for the rest.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from .client_app import DEVICE, SEED
from .model import IDSModel
from .sisa_client import (
    CHECKPOINT_DIR,
    NUM_SHARDS,
    NUM_SLICES,
    TELEMETRY_DIR,
    batch_seed,
    ckpt_path,
    save_checkpoint,
    slice_loader,
    train_slice,
)
from .sisa_partition import TRAIN_FRACTION, shard_slice_assignment

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
PARTITION_ID = int(os.getenv("PARTITION_ID", "2"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "3"))
CACHE_DIR = os.getenv("CACHE_DIR", "/data/.cache/msc")
LR = float(os.getenv("LR", "0.001"))
RECOVERED_MODEL_PATH = os.getenv(
    "RECOVERED_MODEL_PATH", os.path.join(CHECKPOINT_DIR, "recovered_model.pt")
)


def main():
    # Dropout draws from the global torch RNG — seed here so recovery is
    # deterministic regardless of prior RNG consumption in this process.
    torch.manual_seed(SEED * 6271)
    cache = os.path.join(CACHE_DIR, f"partition_{PARTITION_ID}_of_{NUM_PARTITIONS}.npz")
    data = np.load(cache)
    X, y, poison_idx = data["X"], data["y"], data["poison_idx"]
    if len(poison_idx) == 0:
        sys.exit(f"No poison indices in {cache} — nothing to unlearn.")

    split = int(TRAIN_FRACTION * len(X))
    X, y = X[:split], y[:split]
    assignment = shard_slice_assignment(len(X), SEED, NUM_SHARDS, NUM_SLICES)
    poison_set = set(poison_idx.tolist())

    # Locate affected shards and their earliest poisoned slice.
    affected = {}
    for shard in range(NUM_SHARDS):
        for slc, idx in enumerate(assignment[shard]):
            if poison_set.intersection(idx.tolist()):
                affected[shard] = slc
                break
    print(f"Affected shards (earliest poisoned slice): {affected}")

    criterion = torch.nn.BCELoss()
    slice_log = []
    t_start = time.perf_counter()

    for shard, from_slice in affected.items():
        clean = [np.setdiff1d(idx, poison_idx) for idx in assignment[shard]]

        if from_slice == 0:
            torch.manual_seed(SEED * 1000 + shard)
            model = IDSModel(input_dim=X.shape[1]).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            print(f"[shard {shard}] poison in slice 0 — no clean checkpoint, fresh init")
        else:
            rollback = ckpt_path(shard, 1, from_slice - 1)
            state = torch.load(rollback, map_location=DEVICE)
            model = IDSModel(input_dim=X.shape[1]).to(DEVICE)
            model.load_state_dict(state["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            optimizer.load_state_dict(state["optim"])  # resume mid-round-1 exactly
            print(f"[shard {shard}] rolled back to {rollback}")

        for rnd in range(1, NUM_ROUNDS + 1):
            if rnd > 1:
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # fresh per round, as in training
            start_slice = from_slice if rnd == 1 else 0
            for slc in range(start_slice, NUM_SLICES):
                loader = slice_loader(X, y, clean[slc], batch_seed(SEED, rnd, shard, slc))
                t0 = time.perf_counter()
                train_slice(model, optimizer, loader, criterion)
                train_s = time.perf_counter() - t0
                ckpt_s, ckpt_bytes = save_checkpoint(ckpt_path(shard, rnd, slc), model, optimizer, rnd, slc)
                slice_log.append({"shard": shard, "round": rnd, "slice": slc,
                                  "train_s": round(train_s, 4), "ckpt_s": round(ckpt_s, 4),
                                  "ckpt_bytes": ckpt_bytes})

    total_s = time.perf_counter() - t_start

    # Recovered model = parameter average of every constituent's final state.
    states = []
    for shard in range(NUM_SHARDS):
        ckpt = torch.load(ckpt_path(shard, NUM_ROUNDS, NUM_SLICES - 1), map_location=DEVICE)
        states.append(ckpt["model"])
    averaged = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
    with open(RECOVERED_MODEL_PATH, "wb") as f:
        torch.save(averaged, f)
        f.flush()
        os.fsync(f.fileno())

    manifest = {
        "method": "sisa",
        "total_s": round(total_s, 3),
        "affected_shards": {str(k): v for k, v in affected.items()},
        "poisoned_samples": int(len(poison_idx)),
        "retrained_slices": len(slice_log),
        "ckpt_io_s": round(sum(e["ckpt_s"] for e in slice_log), 3),
        "ckpt_bytes": sum(e["ckpt_bytes"] for e in slice_log),
        "recovered_model": RECOVERED_MODEL_PATH,
        "slices": slice_log,
    }
    with open(os.path.join(TELEMETRY_DIR, "recovery_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"SISA recovery done in {total_s:.1f}s "
          f"({len(slice_log)} slice-retrains, {manifest['ckpt_io_s']}s checkpoint I/O). "
          f"Recovered model: {RECOVERED_MODEL_PATH}")


if __name__ == "__main__":
    main()

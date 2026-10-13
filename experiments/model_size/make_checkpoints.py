"""Write the checkpoints sisa_recover reads (each affected shard's
rollback point and every shard's last checkpoint), each after one real
training slice, so a recovery can be timed without running Phase 1.
Recovery time does not depend on the weights.

    python -m experiments.model_size.make_checkpoints
"""

import os

import numpy as np
import torch

from edge_nodes.client_app import SEED
from edge_nodes.model import IDSModel
from edge_nodes.sisa_client import (
    NUM_SHARDS,
    NUM_SLICES,
    batch_seed,
    ckpt_path,
    save_checkpoint,
    slice_loader,
    train_slice,
)
from edge_nodes.sisa_partition import TRAIN_FRACTION, shard_slice_assignment

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
PARTITION_ID = int(os.getenv("PARTITION_ID", "3"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "4"))
CACHE_DIR = os.getenv("CACHE_DIR", "/data/.cache/msc")


def main():
    data = np.load(
        os.path.join(
            CACHE_DIR, f"partition_{PARTITION_ID}_of_{NUM_PARTITIONS}.npz"
        )
    )
    split = int(TRAIN_FRACTION * len(data["X"]))
    X, y, poison = data["X"][:split], data["y"][:split], data["poison_idx"]
    assignment = shard_slice_assignment(len(X), SEED, NUM_SHARDS, NUM_SLICES)
    poison_set = set(poison.tolist())

    needed = {(s, NUM_ROUNDS, NUM_SLICES - 1) for s in range(NUM_SHARDS)}
    for s in range(NUM_SHARDS):
        for slc, idx in enumerate(assignment[s]):
            if poison_set.intersection(idx.tolist()):
                if slc > 0:
                    needed.add((s, 1, slc - 1))
                break

    for s, rnd, slc in sorted(needed):
        torch.manual_seed(SEED * 1000 + s)
        model = IDSModel(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        clean = np.setdiff1d(assignment[s][slc], poison)
        train_slice(
            model,
            optimizer,
            slice_loader(X, y, clean, batch_seed(SEED, rnd, s, slc)),
            torch.nn.BCELoss(),
        )
        seconds, size = save_checkpoint(
            ckpt_path(s, rnd, slc), model, optimizer, rnd, slc
        )
        print(
            f"{ckpt_path(s, rnd, slc)}: {size / 1e6:.1f} MB in {seconds:.2f} "
            "s",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Loads a node's cached partition and applies POISON_MODE (off / flip / drop)."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .sisa_partition import TRAIN_FRACTION

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(DATA_DIR, ".cache"))
# off: clean data. flip: poisoned labels (Phase 1). drop: poisoned rows removed (Phase 4).
POISON_MODE = os.getenv("POISON_MODE", "off")

# client_fn can run once per message, so cache the arrays instead of re-reading
# the ~160 MB npz from the SD card every round.
_arrays_cache = {}


def _cache_path(partition_id: int, num_partitions: int) -> str:
    return os.path.join(CACHE_DIR, f"partition_{partition_id}_of_{num_partitions}.npz")


def load_arrays(partition_id: int, num_partitions: int):
    """Load a partition's raw arrays, applying POISON_MODE=flip if set.

    Poison indices come from experiments/prepare_edge_data.py (seeded, confined
    to one SISA shard's later slices) so the poisoned dataset is bit-identical
    across arms. Row dropping (POISON_MODE=drop) is left to callers since the
    SISA shard/slice assignment is positional and must keep original indexing.

    Returns (X, y, split, poison_idx); split is the train/val boundary.
    """
    key = (partition_id, num_partitions)
    if key in _arrays_cache:
        return _arrays_cache[key]
    cache = _cache_path(partition_id, num_partitions)
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"Cache not found: {cache}. "
            "Run the preprocessor first: docker compose run --rm preprocessor"
        )

    print(f"[Node {partition_id}] Loading from cache: {cache}")
    data = np.load(cache)
    X, y = data["X"], data["y"]
    split = int(TRAIN_FRACTION * len(X))
    poison_idx = data["poison_idx"] if "poison_idx" in data else np.array([], dtype=np.int64)

    if POISON_MODE in ("flip", "drop") and len(poison_idx) == 0:
        raise ValueError(f"POISON_MODE={POISON_MODE} but {cache} contains no poison indices.")

    if POISON_MODE == "flip":
        assert poison_idx.max() < split, "Poison must stay within the train region."
        y = y.copy()
        y[poison_idx] = 0
        print(f"[Node {partition_id}] POISON ACTIVE: "
              f"{len(poison_idx)} labels flipped attack->benign")

    _arrays_cache[key] = (X, y, split, poison_idx)
    return _arrays_cache[key]


def load_data(partition_id: int, num_partitions: int, batch_size: int = 512):
    """Return (trainloader, valloader, input_dim) for one partition.

    With POISON_MODE=drop the poisoned rows are removed from the train loader
    (labels stay true for the retained rows); the val loader is always clean.
    """
    X, y, split, poison_idx = load_arrays(partition_id, num_partitions)

    train_idx = np.arange(split)
    if POISON_MODE == "drop":
        train_idx = np.setdiff1d(train_idx, poison_idx)
        print(f"[Node {partition_id}] POISON DROPPED: "
              f"training on {len(train_idx)}/{split} retained rows")

    def make_loader(Xa, ya, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    trainloader = make_loader(X[train_idx], y[train_idx], True)
    valloader = make_loader(X[split:], y[split:], False)
    return trainloader, valloader, X.shape[1]

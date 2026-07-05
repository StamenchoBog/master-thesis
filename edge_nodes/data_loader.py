import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .sisa_partition import TRAIN_FRACTION

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(DATA_DIR, ".cache"))
# off  — clean data (course sim, clean reference runs)
# flip — poisoned labels active (Phase 1: attack in progress, undetected)
# drop — poisoned rows removed (Phase 4: post-recovery rejoin on retained data)
POISON_MODE = os.getenv("POISON_MODE", "off")

# The deployment engine may call client_fn per message; keep the decompressed
# arrays in memory so the ~160 MB npz isn't re-read from SD every round.
_arrays_cache = {}


def _cache_path(partition_id: int, num_partitions: int) -> str:
    return os.path.join(CACHE_DIR, f"partition_{partition_id}_of_{num_partitions}.npz")


def load_arrays(partition_id: int, num_partitions: int):
    """Load a partition's raw arrays, applying POISON_MODE=flip if set.

    Poison indices are embedded in the cache by experiments/prepare_edge_data.py
    (seeded, confined to the later slices of one SISA shard) so the poisoned
    dataset is bit-identical across experiment arms. Only the train region is
    ever poisoned; the val split stays clean. Row dropping (POISON_MODE=drop)
    is left to consumers because the SISA shard/slice assignment is positional
    and must keep the original indexing.

    Returns (X, y, split, poison_idx) where split is the train/val boundary.
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
        print(f"[Node {partition_id}] POISON ACTIVE: {len(poison_idx)} labels flipped attack->benign")

    _arrays_cache[key] = (X, y, split, poison_idx)
    return _arrays_cache[key]


def load_data(partition_id: int, num_partitions: int, batch_size: int = 512):
    """Load a pre-built partition from the .npz cache and return train/val data loaders.

    With POISON_MODE=drop the poisoned rows are removed from the train loader
    (labels stay true for the retained rows); the val loader is always clean.

    Args:
        partition_id:   Index of this node (0-based).
        num_partitions: Total number of nodes.
        batch_size:     Mini-batch size for both loaders.

    Returns:
        Tuple of (trainloader, valloader, input_dim).
    """
    X, y, split, poison_idx = load_arrays(partition_id, num_partitions)

    train_idx = np.arange(split)
    if POISON_MODE == "drop":
        train_idx = np.setdiff1d(train_idx, poison_idx)
        print(f"[Node {partition_id}] POISON DROPPED: training on {len(train_idx)}/{split} retained rows")

    def make_loader(Xa, ya, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return make_loader(X[train_idx], y[train_idx], True), make_loader(X[split:], y[split:], False), X.shape[1]

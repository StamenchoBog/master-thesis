"""Seeded shard/slice assignment and poison placement, shared by data prep, loading and the
SISA client, so all three agree on which sample sits where.
"""

import numpy as np

TRAIN_FRACTION = 0.8  # must match the split in data_loader.load_data


def shard_slice_assignment(n_samples: int, seed: int, num_shards: int = 5, num_slices: int = 5):
    """Returns S shards, each a list of R per-slice index arrays."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    shards = np.array_split(order, num_shards)
    return [np.array_split(shard, num_slices) for shard in shards]


def poison_indices(y, assignment, shard_idx: int, from_slice: int, fraction: float, seed: int):
    """Attack samples (y == 1) in slices >= from_slice of one shard: a source compromised at
    time tau."""
    rng = np.random.default_rng(seed + 1)  # separate stream from the assignment
    target = np.concatenate(assignment[shard_idx][from_slice:])
    attack = target[y[target] == 1]
    k = int(len(attack) * fraction)
    return np.sort(rng.choice(attack, size=k, replace=False))

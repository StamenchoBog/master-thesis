"""Deterministic shard/slice assignment shared by data prep, loading, and the SISA client.

All three consumers must agree on which sample belongs to which shard/slice,
so the assignment is a pure function of (n_samples, seed, S, R). Poison
placement is likewise seeded, making the poisoned dataset identical across
experiment arms.
"""

import numpy as np

TRAIN_FRACTION = 0.8  # must match the split in data_loader.load_data


def shard_slice_assignment(n_samples: int, seed: int, num_shards: int = 5, num_slices: int = 5):
    """Assign sample indices to S shards x R slices via a seeded permutation.

    Random permutation is approximately class-balanced at the sample sizes used
    here (~200k rows per shard). Returns a list of shards, each a list of
    per-slice index arrays.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    shards = np.array_split(order, num_shards)
    return [np.array_split(shard, num_slices) for shard in shards]


def poison_indices(y, assignment, shard_idx: int, from_slice: int, fraction: float, seed: int):
    """Pick attack samples (y==1) to label-flip within the target slices of one shard.

    Models a data source compromised at time tau: only slices >= from_slice of
    shard shard_idx are affected. Returns the selected indices (into y).
    """
    rng = np.random.default_rng(seed + 1)  # separate stream from the assignment
    target = np.concatenate(assignment[shard_idx][from_slice:])
    attack = target[y[target] == 1]
    k = int(len(attack) * fraction)
    return np.sort(rng.choice(attack, size=k, replace=False))

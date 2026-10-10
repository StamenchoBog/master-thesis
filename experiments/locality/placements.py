"""Poison placements for the locality sweep (experiments/protocol.md, Addendum B).

    python3 experiments/locality/placements.py            # writes data/.cache/locality/<name>/

Each placement is a copy of the Pi's partition (seed 42) with a different poison index
set: a fraction of the attack samples in slices >= l* of shards 1..k, chosen the same way
as in the campaign (edge_nodes.sisa_partition). The sweep uses fraction 0.05, so about the
same data is left to replay in every configuration; k1_l3_full uses 1.0 and is exactly the
campaign placement. The recovery scripts read the file through CACHE_DIR, so nothing in
the recovery code changes.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from edge_nodes.sisa_partition import (  # noqa: E402
    TRAIN_FRACTION, poison_indices, shard_slice_assignment)

SWEEP_FRACTION = 0.05
PLACEMENTS = {  # name: (affected shards, earliest poisoned slice, poison fraction)
    "k1_l0": ([1], 0, SWEEP_FRACTION), "k1_l1": ([1], 1, SWEEP_FRACTION),
    "k1_l2": ([1], 2, SWEEP_FRACTION), "k1_l3": ([1], 3, SWEEP_FRACTION),
    "k1_l4": ([1], 4, SWEEP_FRACTION),
    "k2_l3": ([1, 2], 3, SWEEP_FRACTION), "k3_l3": ([1, 2, 3], 3, SWEEP_FRACTION),
    "k5_l3": ([0, 1, 2, 3, 4], 3, SWEEP_FRACTION),
    "k1_l3_full": ([1], 3, 1.0),
}


def build(partition_path, seed, shards=5, slices=5):
    d = np.load(partition_path)
    X, y = d["X"], d["y"]
    n_train = int(TRAIN_FRACTION * len(X))
    assignment = shard_slice_assignment(n_train, seed, shards, slices)
    out = {}
    for name, (affected, l_star, fraction) in PLACEMENTS.items():
        idx = np.concatenate([poison_indices(y[:n_train], assignment, s, l_star, fraction, seed)
                              for s in affected])
        out[name] = (X, y, np.sort(idx))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", default="data/.cache/msc/partition_3_of_4.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/.cache/locality")
    args = ap.parse_args()

    for name, (X, y, idx) in build(args.partition, args.seed).items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "partition_3_of_4.npz"), X=X, y=y, poison_idx=idx)
        print(f"{name}: {len(idx)} poisoned samples")


if __name__ == "__main__":
    main()

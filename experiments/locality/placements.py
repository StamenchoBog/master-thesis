"""Poison placements for the locality sweep, written to data/.cache/locality/<name>/.

    python3 experiments/locality/placements.py

Each is the Pi's seed-42 partition with 5 % of the attack samples in slices >= l* of shards
1..k poisoned; k1_l3_full poisons all of them, as in the campaign.
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

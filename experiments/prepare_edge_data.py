"""Build MSc experiment data caches from the full course partition caches.

Run on the host after `docker compose run --rm preprocessor`:

    python experiments/prepare_edge_data.py --seed 42

Outputs to data/.cache/msc/:
  partition_{i}_of_3.npz  — equal-sized stratified subsamples (X, y, poison_idx)
  test_global.npz         — clean held-out test set drawn disjointly from all partitions

All three nodes get the SAME subsample size: FedAvg weights clients by sample
count, so unequal sizes would dilute the Pi's poisoned/cleaned contribution and
confound the arms comparison.

Poison indices (seeded label flips, attack->benign, confined to slices
>= --poison-from-slice of shard --poison-shard) are embedded in the Pi's
partition only; labels on disk stay clean and are flipped at load time when
POISON_ENABLED=1.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge_nodes.sisa_partition import TRAIN_FRACTION, poison_indices, shard_slice_assignment

PI_PARTITION = 2
NUM_PARTITIONS = 3


def stratified_indices(y, k, rng):
    """Pick k indices preserving the class ratio of y, in shuffled order."""
    chosen = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        take = round(k * count / len(y))
        chosen.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    idx = np.concatenate(chosen)
    return rng.permutation(idx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rows", type=int, default=1_000_000, help="subsample size per node")
    p.add_argument("--test-rows", type=int, default=100_000, help="total clean test rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shards", type=int, default=5)
    p.add_argument("--slices", type=int, default=5)
    p.add_argument("--poison-shard", type=int, default=1)
    p.add_argument("--poison-from-slice", type=int, default=3)
    p.add_argument("--poison-fraction", type=float, default=0.5,
                   help="fraction of attack samples flipped within the target slices")
    args = p.parse_args()

    src_dir = os.path.join(args.data_dir, ".cache")
    out_dir = os.path.join(src_dir, "msc")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    test_X, test_y = [], []
    manifest = {"seed": args.seed, "rows": args.rows, "test_rows": args.test_rows,
                "shards": args.shards, "slices": args.slices,
                "poison_shard": args.poison_shard, "poison_from_slice": args.poison_from_slice,
                "poison_fraction": args.poison_fraction, "partitions": {}}

    for pid in range(NUM_PARTITIONS):
        src = os.path.join(src_dir, f"partition_{pid}_of_{NUM_PARTITIONS}.npz")
        data = np.load(src)
        X, y = data["X"], data["y"]
        print(f"[partition {pid}] {len(X)} rows total")

        per_node_test = args.test_rows // NUM_PARTITIONS
        picked = stratified_indices(y, args.rows + per_node_test, rng)
        train_idx, test_idx = picked[:args.rows], picked[args.rows:args.rows + per_node_test]

        Xs, ys = X[train_idx], y[train_idx]
        test_X.append(X[test_idx])
        test_y.append(y[test_idx])

        poison_idx = np.array([], dtype=np.int64)
        if pid == PI_PARTITION:
            n_train = int(TRAIN_FRACTION * len(Xs))
            assignment = shard_slice_assignment(n_train, args.seed, args.shards, args.slices)
            poison_idx = poison_indices(ys, assignment, args.poison_shard,
                                        args.poison_from_slice, args.poison_fraction, args.seed)
            print(f"[partition {pid}] poison: {len(poison_idx)} attack labels marked for flipping "
                  f"(shard {args.poison_shard}, slices {args.poison_from_slice}+)")

        out = os.path.join(out_dir, f"partition_{pid}_of_{NUM_PARTITIONS}.npz")
        np.savez_compressed(out, X=Xs, y=ys, poison_idx=poison_idx)
        manifest["partitions"][pid] = {"rows": len(Xs), "attack_ratio": float(ys.mean()),
                                       "poisoned": int(len(poison_idx))}
        print(f"[partition {pid}] saved {out}")

    np.savez_compressed(os.path.join(out_dir, "test_global.npz"),
                        X=np.concatenate(test_X), y=np.concatenate(test_y))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Test set: {sum(len(t) for t in test_y)} rows. Manifest written. Done.")


if __name__ == "__main__":
    main()

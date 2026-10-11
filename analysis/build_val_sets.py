"""Rebuild each seed's validation splits, used to choose the decision threshold.

    python -m analysis.build_val_sets            # writes data/.cache/msc/val_seed{S}.npz

Re-runs prepare_edge_data.py per seed in a temp dir; the regenerated test set must match
test_seed{S}.npz byte for byte.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

from edge_nodes.sisa_partition import TRAIN_FRACTION

NUM_PARTITIONS = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    args = ap.parse_args()

    src = os.path.abspath(os.path.join(args.data_dir, ".cache"))
    out_dir = os.path.join(src, "msc")

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".cache"))
            for pid in range(NUM_PARTITIONS):
                name = f"partition_{pid}_of_{NUM_PARTITIONS}.npz"
                os.symlink(os.path.join(src, name), os.path.join(tmp, ".cache", name))
            subprocess.run([sys.executable, "experiments/prepare_edge_data.py",
                            "--data-dir", tmp, "--seed", str(seed)],
                           check=True, stdout=subprocess.DEVNULL)

            gen = os.path.join(tmp, ".cache", "msc")
            test_new = np.load(os.path.join(gen, "test_global.npz"))
            test_old = np.load(os.path.join(out_dir, f"test_seed{seed}.npz"))
            same = (np.array_equal(test_new["X"], test_old["X"])
                    and np.array_equal(test_new["y"], test_old["y"]))
            if not same:
                raise SystemExit(f"seed {seed}: regenerated test set differs from "
                                 f"test_seed{seed}.npz, the regeneration is not exact")

            Xs, ys = [], []
            for pid in range(NUM_PARTITIONS):
                d = np.load(os.path.join(gen, f"partition_{pid}_of_{NUM_PARTITIONS}.npz"))
                split = int(TRAIN_FRACTION * len(d["y"]))
                Xs.append(d["X"][split:])
                ys.append(d["y"][split:])

        X, y = np.concatenate(Xs), np.concatenate(ys)
        path = os.path.join(out_dir, f"val_seed{seed}.npz")
        np.savez_compressed(path, X=X, y=y)
        print(f"seed {seed}: test set matches, val n={len(y)} attack={y.mean():.4f} -> {path}")


if __name__ == "__main__":
    main()

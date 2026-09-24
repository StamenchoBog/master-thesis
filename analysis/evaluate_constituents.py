"""Is the SISA majority-class collapse caused by averaging, or already in the shards?

    python -m analysis.evaluate_constituents --checkpoints ~/msc-experiment/checkpoints

The thesis attributes the collapse (specificity ~0, balanced accuracy ~0.50 — see
evaluate_utility.py) to the documented deviation from vanilla SISA: the client sends the
*parameter average* of its S constituents, not an ensemble of predictions. That's an
inference, not a measurement — each constituent also only ever sees pool/S rows in
isolation, so the shards themselves could be the degenerate part instead.

This scores, on the same held-out test set, each constituent individually (final
checkpoint per shard) against their parameter average (what averaged_parameters /
sisa_recover actually emit):

  constituents ~0.60, average ~0.50  -> averaging is the culprit; thesis claim holds
  constituents ~0.50 too             -> averaging is NOT the cause; isolation or shard
                                         size is, and the Discussion needs rewriting

Per-constituent BatchNorm running-stat spread is printed as a secondary clue (large
spread => incompatible normalisations, the usual reason weight averaging fails).

Checkpoints live under CHECKPOINT_DIR, wiped at the start of every run — this is a
single-run diagnostic, not a paired measurement.
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from edge_nodes.model import IDSModel
from edge_nodes.sisa_partition import TRAIN_FRACTION


def final_constituents(ckpt_dir: str) -> dict:
    """{shard: path} for each shard's last checkpoint (highest round, then slice)."""
    pat = re.compile(r"shard(\d+)_round(\d+)_slice(\d+)\.pt$")
    best = {}
    for p in glob.glob(os.path.join(ckpt_dir, "shard*_round*_slice*.pt")):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        shard, rnd, slc = (int(g) for g in m.groups())
        if (rnd, slc) > best.get(shard, ((-1, -1), None))[0]:
            best[shard] = ((rnd, slc), p)
    return {s: v[1] for s, v in sorted(best.items())}


def score(state, X, y, input_dim):
    model = IDSModel(input_dim)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    model.eval()   # use the (averaged) BatchNorm running stats, which is what's being tested
    with torch.no_grad():
        prob = model(X).squeeze().numpy()
    pred = (prob > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {"recall": recall, "specificity": specificity,
            "balanced_acc": (recall + specificity) / 2,
            "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, prob),
            "pred_pos_rate": float(pred.mean()), "tn": tn}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="/checkpoints",
                    help="directory holding shard{S}_round{R}_slice{L}.pt")
    ap.add_argument("--test", default="data/.cache/msc/test_global.npz")
    ap.add_argument("--partition",
                    help="use a node partition npz instead of --test (rows past the 0.8 "
                         "train split as the held-out set) — lets this run on the Pi "
                         "without the global test set; any held-out set answers whether "
                         "constituents discriminate better than their average.")
    args = ap.parse_args()

    if args.partition:
        d = np.load(args.partition)
        split = int(TRAIN_FRACTION * len(d["y"]))
        X = torch.tensor(d["X"][split:], dtype=torch.float32)
        y = d["y"][split:].astype(int)
        print(f"Held-out val region of {os.path.basename(args.partition)} "
              f"(rows {split}..{len(d['y'])}): n={len(y)}, attack={y.mean():.4f}\n")
    else:
        d = np.load(args.test)
        X = torch.tensor(d["X"], dtype=torch.float32)
        y = d["y"].astype(int)
        print(f"Test set: n={len(y)}, attack={y.mean():.4f}\n")

    paths = final_constituents(args.checkpoints)
    if not paths:
        raise SystemExit(f"No constituent checkpoints under {args.checkpoints}. This "
                         f"directory is wiped before each run, so only the last run's survive.")

    states = {}
    hdr = (f"{'model':<16}{'recall':>8}{'specif.':>9}{'bal.acc':>9}{'MCC':>8}"
           f"{'ROC-AUC':>9}{'pred+':>8}{'tn':>7}")
    print(hdr)
    print("-" * len(hdr))
    for shard, path in paths.items():
        states[shard] = torch.load(path, map_location="cpu", weights_only=True)["model"]
        m = score(states[shard], X, y, X.shape[1])
        print(f"{'constituent ' + str(shard):<16}{m['recall']:>8.4f}{m['specificity']:>9.4f}"
              f"{m['balanced_acc']:>9.4f}{m['mcc']:>8.4f}{m['roc_auc']:>9.4f}"
              f"{m['pred_pos_rate']:>8.4f}{m['tn']:>7d}")

    # Same aggregation as sisa_client.averaged_parameters.
    keys = next(iter(states.values())).keys()
    averaged = {k: torch.stack([s[k].float() for s in states.values()]).mean(0) for k in keys}
    a = score(averaged, X, y, X.shape[1])
    print("-" * len(hdr))
    print(f"{'AVERAGED':<16}{a['recall']:>8.4f}{a['specificity']:>9.4f}{a['balanced_acc']:>9.4f}"
          f"{a['mcc']:>8.4f}{a['roc_auc']:>9.4f}{a['pred_pos_rate']:>8.4f}{a['tn']:>7d}")

    best = max(score(s, X, y, X.shape[1])["balanced_acc"] for s in states.values())
    print(f"\nBest single constituent: {best:.4f} balanced accuracy | averaged: "
          f"{a['balanced_acc']:.4f}")
    if best - a['balanced_acc'] > 0.02:
        print("=> Averaging destroys discrimination the constituents individually have. "
              "The thesis attribution holds.")
    else:
        print("=> Constituents are degenerate on their own; averaging is NOT the cause. "
              "The Discussion's explanation needs revising.")

    bn = [k for k in keys if "running_mean" in k]
    if bn and len(states) > 1:
        spread = np.mean([float(torch.stack([s[k].float() for s in states.values()]).std(0).mean())
                          for k in bn])
        print(f"BatchNorm running_mean spread across constituents (mean std): {spread:.4f} "
              f"— large values indicate the shards learned incompatible normalisations, the "
              f"usual reason weight averaging fails.")


if __name__ == "__main__":
    main()

"""Re-score the saved global models with metrics that hold up under class imbalance (H5).

    python -m analysis.evaluate_utility --phase1 \
      --test-template data/.cache/msc/test_seed{seed}.npz \
      --val-template data/.cache/msc/val_seed{seed}.npz

Writes results/msc/tables/campaign_utility.csv: metrics at 0.5, at the threshold chosen on the
validation splits (*_val) and at the test-optimal threshold (*_tuned). Build the per-seed test
sets first:

    for S in 42 43 44 45 46 47 48 49 50 51; do
      python3 experiments/prepare_edge_data.py --seed $S
      cp data/.cache/msc/test_global.npz data/.cache/msc/test_seed$S.npz
    done
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, matthews_corrcoef, roc_auc_score,
                             roc_curve)

from edge_nodes.model import IDSModel


def load_model(npz_path, input_dim):
    z = np.load(npz_path)
    arrays = [z[f] for f in z.files]
    model = IDSModel(input_dim)
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, arrays)}, strict=True)
    model.eval()
    return model


def _counts(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def best_threshold(prob, y):
    """Threshold that maximises balanced accuracy. Metrics come from tpr/fpr, because
    roc_curve picks with >= while deployment uses >."""
    fpr, tpr, thr = roc_curve(y, prob)
    bacc = (tpr + (1.0 - fpr)) / 2.0
    i = int(np.argmax(bacc))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tp = tpr[i] * n_pos
    fn = n_pos - tp
    fp = fpr[i] * n_neg
    tn = n_neg - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        # roc_curve prepends an unreachable threshold (max(prob)+1) for the all-negative
        # corner; clip so the reported value is one a deployment could actually set.
        "thr_best": min(float(thr[i]), float(prob.max())),
        "balanced_acc_tuned": float(bacc[i]),
        "recall_tuned": float(tpr[i]),
        "specificity_tuned": float(1.0 - fpr[i]),
        "mcc_tuned": float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0,
        "pred_pos_rate_tuned": float((tp + fp) / len(y)),
    }


def at_threshold(prob, y, thr):
    """Metrics at a fixed threshold (`>=`, the same rule roc_curve uses to pick it)."""
    tp, fp, tn, fn = _counts((prob >= thr).astype(int), y)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "balanced_acc_val": (recall + specificity) / 2,
        "recall_val": recall,
        "specificity_val": specificity,
        "mcc_val": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
    }


def score_model(npz_path, X, y, val=None):
    model = load_model(npz_path, X.shape[1])
    with torch.no_grad():
        prob = model(X).squeeze().numpy()
    extra = {}
    if val is not None:
        # Threshold picked on the federation's validation split, applied to the test set:
        # what a deployment could actually do, unlike the test-optimal *_tuned columns.
        Xv, yv = val
        with torch.no_grad():
            prob_v = model(Xv).squeeze().numpy()
        thr_val = best_threshold(prob_v, yv)["thr_best"]
        extra = {"thr_val": thr_val, **at_threshold(prob, y, thr_val)}
    pred = (prob > 0.5).astype(int)
    tp, fp, tn, fn = _counts(pred, y)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "recall": recall, "specificity": specificity, "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "balanced_acc": (recall + specificity) / 2,
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        # benign is the rare class (~3.5 %), so its PR curve is the informative one
        "pr_auc_benign": average_precision_score(1 - y, 1 - prob),
        "pred_pos_rate": float(pred.mean()),
        "tn": tn, "fp": fp,
        **best_threshold(prob, y),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/msc/campaign")
    ap.add_argument("--test", default="data/.cache/msc/test_global.npz",
                    help="single shared test set (see --test-template for the clean option)")
    ap.add_argument("--test-template",
                    help="per-seed test set path with a {seed} placeholder, e.g. "
                         "data/.cache/msc/test_seed{seed}.npz. Scores each seed against "
                         "its own held-out set instead of one shared file")
    ap.add_argument("--phase1", action="store_true", help="also score the Phase-1 model")
    ap.add_argument("--val-template",
                    help="per-seed validation set with a {seed} placeholder (built by "
                         "analysis.build_val_sets); adds *_val columns: threshold chosen on "
                         "validation, metrics on the test set")
    ap.add_argument("--out", default="results/msc/tables/campaign_utility.csv")
    args = ap.parse_args()

    def val_set(seed):
        if not args.val_template:
            return None
        d = np.load(args.val_template.format(seed=seed))
        return torch.tensor(d["X"], dtype=torch.float32), d["y"].astype(int)

    cache = {}

    def test_set(seed):
        path = args.test_template.format(seed=seed) if args.test_template else args.test
        if path not in cache:
            d = np.load(path)
            y = d["y"].astype(int)
            cache[path] = (torch.tensor(d["X"], dtype=torch.float32), y, os.path.basename(path))
            print(f"Test set {os.path.basename(path)}: n={len(y)}, attack={y.mean():.4f}, "
                  f"benign={1 - y.mean():.4f}")
        return cache[path]

    if not args.test_template:
        print("WARNING: scoring every seed against one shared test set. That set is disjoint "
              "from only ONE seed's training subsample, so absolute values for the other "
              "seeds are optimistic. Paired contrasts are unaffected. Use --test-template "
              "for the clean comparison.")

    rows = []
    for run in sorted(glob.glob(os.path.join(args.runs, "*_seed*"))):
        b = os.path.basename(run)
        m = re.fullmatch(r"(naive|sisa)_seed(\d+)", b)
        if not m:
            continue
        arm, seed = m.group(1), m.group(2)
        X, y, test_name = test_set(int(seed))
        val = val_set(int(seed))
        checkpoints = {"p4": os.path.join(run, "phase4_checkpoints", "round_5.npz")}
        if args.phase1:
            checkpoints["p1"] = os.path.join(run, "phase1_checkpoints", "round_10.npz")
        for phase, ckpt in checkpoints.items():
            if os.path.exists(ckpt):
                rows.append({"run": b, "arm": arm, "seed": int(seed), "phase": phase,
                             "test_set": test_name,
                             **score_model(ckpt, X, y, val)})

    df = pd.DataFrame(rows).round(4)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    for phase in df["phase"].unique():
        print(f"\n[{phase}] medians by arm:")
        print(df[df.phase == phase].groupby("arm")[
            ["recall", "specificity", "balanced_acc", "mcc", "roc_auc", "pr_auc",
             "thr_best", "balanced_acc_tuned", "specificity_tuned", "mcc_tuned"]
            + (["thr_val", "balanced_acc_val", "specificity_val", "mcc_val"]
               if args.val_template else [])
        ].median().round(4).to_string())
        print("  (*_tuned = at the balanced-accuracy-optimal threshold; the gap between "
              "balanced_acc and balanced_acc_tuned is how much is purely calibration)")


if __name__ == "__main__":
    main()

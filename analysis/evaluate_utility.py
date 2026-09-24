"""Re-score saved global models with balanced, imbalance-aware metrics (H5).

    python -m analysis.evaluate_utility           # needs torch + scikit-learn

Recall/F1 are near-trivial on the 96.5%-attack test set (predicting all-attack scores
recall 1.0, F1 0.98). That masks the real effect: at the deployed 0.5 threshold, SISA's
parameter-averaged model collapses to the majority class (balanced accuracy ~0.50 =
chance) while naive retraining keeps benign discrimination (~0.60) — even though
ROC-AUC is comparable, so it's a *calibration* failure, not lost information.

Loads each run's Phase-4 model (and, with --phase1, the Phase-1 model, to show the
collapse predates recovery). Writes utility_evaluation.csv next to the runs.

Threshold sweep: reports the best attainable balanced accuracy per model (the
`*_tuned` columns), turning "ROC-AUC survives" into a measurement of how much
discrimination is recoverable by calibration alone.

Per-seed test sets: `prepare_edge_data.py` overwrites the same `test_global.npz` per
seed, so scoring all 20 models against whichever file is on disk means 9/10 seeds see
rows from their own training subsample (paired contrasts stay valid; absolute values
skew optimistic). Use --test-template for a clean per-seed comparison:

    for S in 42 43 44 45 46 47 48 49 50 51; do
      python3 experiments/prepare_edge_data.py --seed $S
      cp data/.cache/msc/test_global.npz data/.cache/msc/test_seed$S.npz
    done
    python -m analysis.evaluate_utility --phase1 \
      --test-template data/.cache/msc/test_seed{seed}.npz
"""

import argparse
import glob
import os

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
    model.eval()  # BatchNorm uses running stats, Dropout off — must match inference
    return model


def _counts(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, fp, tn, fn


def best_threshold(prob, y):
    """Threshold maximising balanced accuracy (= Youden's J), and the metrics there.

    Uses the ROC sweep (every attainable operating point) rather than a fixed grid.
    Metrics are derived from tpr/fpr, not by re-thresholding at `thr` — roc_curve
    scores positive on `>=` while deployment uses `>`, which would make the *_tuned
    metrics disagree with balanced_acc_tuned at tied scores.
    """
    fpr, tpr, thr = roc_curve(y, prob)
    bacc = (tpr + (1.0 - fpr)) / 2.0
    i = int(np.argmax(bacc))
    P = int((y == 1).sum()); N = int((y == 0).sum())
    tp = tpr[i] * P; fn = P - tp
    fp = fpr[i] * N; tn = N - fp
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


def score_model(npz_path, X, y):
    with torch.no_grad():
        prob = load_model(npz_path, X.shape[1])(X).squeeze().numpy()
    pred = (prob > 0.5).astype(int)
    tp, fp, tn, fn = _counts(pred, y)
    recall = tp / max(tp + fn, 1)              # = sensitivity (attack caught)
    specificity = tn / max(tn + fp, 1)         # benign correctly identified
    precision = tp / max(tp + fp, 1)
    return {
        "recall": recall, "specificity": specificity, "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "balanced_acc": (recall + specificity) / 2,   # the fair headline under imbalance
        "mcc": matthews_corrcoef(y, pred),            # 0 = chance, robust to imbalance
        "roc_auc": roc_auc_score(y, prob),            # threshold-independent ranking
        "pr_auc": average_precision_score(y, prob),   # attack-class PR (baseline = base rate)
        "pred_pos_rate": float(pred.mean()),          # 1.0 => predicts everything attack
        "tn": tn, "fp": fp,
        # How much of the collapse is merely a misplaced threshold?
        **best_threshold(prob, y),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/msc/runs")
    ap.add_argument("--test", default="data/.cache/msc/test_global.npz",
                    help="single shared test set (see --test-template for the clean option)")
    ap.add_argument("--test-template",
                    help="per-seed test set path with a {seed} placeholder, e.g. "
                         "data/.cache/msc/test_seed{seed}.npz — scores each seed against "
                         "its own held-out set instead of one shared file")
    ap.add_argument("--phase1", action="store_true", help="also score the Phase-1 model")
    args = ap.parse_args()

    cache = {}

    def test_set(seed):
        """Load (and memoise) the test set for one seed; falls back to the shared file."""
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
        if b.startswith("rehearsal"):
            continue
        arm, seed = b.split("_seed")
        X, y, test_name = test_set(int(seed))
        checkpoints = {"p4": os.path.join(run, "phase4_checkpoints", "round_5.npz")}
        if args.phase1:
            checkpoints["p1"] = os.path.join(run, "phase1_checkpoints", "round_10.npz")
        for phase, ckpt in checkpoints.items():
            if os.path.exists(ckpt):
                rows.append({"run": b, "arm": arm, "seed": int(seed), "phase": phase,
                             "test_set": test_name,       # keeps the above auditable
                             **score_model(ckpt, X, y)})

    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(args.runs, "utility_evaluation.csv"), index=False)
    print(df.to_string(index=False))
    for phase in df["phase"].unique():
        print(f"\n[{phase}] medians by arm:")
        print(df[df.phase == phase].groupby("arm")[
            ["recall", "specificity", "balanced_acc", "mcc", "roc_auc", "pr_auc",
             "thr_best", "balanced_acc_tuned", "specificity_tuned", "mcc_tuned"]
        ].median().round(4).to_string())
        print("  (*_tuned = at the balanced-accuracy-optimal threshold; the gap between "
              "balanced_acc and balanced_acc_tuned is how much is purely calibration)")


if __name__ == "__main__":
    main()

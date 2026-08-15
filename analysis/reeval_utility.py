"""Re-score saved global models on the held-out global test set with balanced,
imbalance-aware metrics — the honest utility comparison for a 96.5%-attack test set.

    python -m analysis.reeval_utility            # needs torch+sklearn; run in the
                                                 # clientapp image, see the runbook

Why this exists: `results_phase{1,4}.json` report recall/precision/F1, which are
near-trivial on a 96.5%-positive set (a model that predicts *all attack* scores
recall 1.0, F1 0.98). That masked a real effect: the SISA parameter-averaging
aggregation produces a global model that, at the deployed 0.5 threshold, collapses
to the majority class (predicted-positive rate = 1.0, specificity ~ 0, balanced
accuracy ~ 0.50 = chance), whereas naive retraining keeps benign discrimination
(balanced accuracy ~ 0.60). Threshold-independently, ROC-AUC is comparable between
the arms, so the collapse is a *calibration* failure of parameter averaging, not a
loss of ranking information.

Loads each run's Phase-4 final global model (phase4_checkpoints/round_5.npz) and,
optionally, the Phase-1 model (phase1_checkpoints/round_10.npz) to show the collapse
predates recovery. Writes utility_reeval.csv next to the runs.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from edge_nodes.model import IDSModel


def load_model(npz_path, input_dim):
    z = np.load(npz_path)
    arrays = [z[f] for f in z.files]
    model = IDSModel(input_dim)
    keys = model.state_dict().keys()
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, arrays)}, strict=True)
    model.eval()  # BatchNorm uses running stats, Dropout off — must match inference
    return model


def score_model(npz_path, X, y):
    with torch.no_grad():
        prob = load_model(npz_path, X.shape[1])(X).squeeze().numpy()
    pred = (prob > 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
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
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/msc/runs")
    ap.add_argument("--test", default="data/.cache/msc/test_global.npz")
    ap.add_argument("--phase1", action="store_true", help="also score the Phase-1 model")
    args = ap.parse_args()

    d = np.load(args.test)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = d["y"].astype(int)
    print(f"Held-out global test set: n={len(y)}, attack={y.mean():.4f}, benign={1 - y.mean():.4f}")

    rows = []
    for run in sorted(glob.glob(os.path.join(args.runs, "*_seed*"))):
        b = os.path.basename(run)
        if b.startswith("rehearsal"):
            continue
        arm, seed = b.split("_seed")
        checkpoints = {"p4": os.path.join(run, "phase4_checkpoints", "round_5.npz")}
        if args.phase1:
            checkpoints["p1"] = os.path.join(run, "phase1_checkpoints", "round_10.npz")
        for phase, ckpt in checkpoints.items():
            if os.path.exists(ckpt):
                rows.append({"run": b, "arm": arm, "seed": int(seed), "phase": phase,
                             **score_model(ckpt, X, y)})

    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(args.runs, "utility_reeval.csv"), index=False)
    print(df.to_string(index=False))
    for phase in df["phase"].unique():
        print(f"\n[{phase}] medians by arm:")
        print(df[df.phase == phase].groupby("arm")[
            ["recall", "specificity", "balanced_acc", "mcc", "roc_auc", "pr_auc"]
        ].median().round(4).to_string())


if __name__ == "__main__":
    main()

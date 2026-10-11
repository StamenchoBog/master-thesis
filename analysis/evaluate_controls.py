"""Utility of the control runs (experiments/controls/run.sh).

    python -m analysis.evaluate_controls

1. H5: each arm's Phase-4 model against a federation of the same arm without poison.
2. The clean SISA federation against the clean naive one.
3. The Pi's SISA constituents one by one, as their parameter average and as an ensemble.
Thresholds are chosen on each seed's validation set, never on the test set.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from analysis.analyze_runs import _bootstrap_ci, cliffs_delta
from analysis.evaluate_utility import best_threshold, load_model
from edge_nodes.model import IDSModel

SEEDS = range(42, 52)
TOLERANCE = 0.02


def metrics(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {"recall": recall, "specificity": specificity,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "balanced_acc": (recall + specificity) / 2,
            "mcc": matthews_corrcoef(y, pred)}


def score(prob_test, y_test, prob_val, y_val):
    thr = best_threshold(prob_val, y_val)["thr_best"]
    # 0.5 uses `>` like deployment; the tuned threshold uses `>=`, the rule roc_curve picks it with.
    row = dict(metrics((prob_test > 0.5).astype(int), y_test))
    row.update({f"{k}_val": v for k, v in metrics((prob_test >= thr).astype(int), y_test).items()})
    row["thr_val"] = thr
    row["roc_auc"] = roc_auc_score(y_test, prob_test)
    return row


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze().numpy()


def from_state(state, input_dim):
    model = IDSModel(input_dim)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in state.items()}, strict=True)
    return model


def averaged(states):
    """The same aggregation as sisa_client.averaged_parameters."""
    keys = states[0].keys()
    avg = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return {k: v.to(states[0][k].dtype) for k, v in avg.items()}


def paired(df, a, b, metric):
    p = df.pivot(index="seed", columns="model", values=metric)[[a, b]].dropna()
    if len(p) < 2:
        return None
    diff = (p[a] - p[b]).values
    lo, hi = _bootstrap_ci(diff)
    try:
        pv = wilcoxon(p[a], p[b]).pvalue
    except ValueError:
        pv = float("nan")
    return {"metric": metric, "a": a, "b": b, "n": len(p),
            "a_median": p[a].median(), "b_median": p[b].median(),
            "diff_median": float(np.median(diff)), "ci_lo": lo, "ci_hi": hi,
            "wilcoxon_p": pv, "cliffs_delta": cliffs_delta(p[a], p[b]),
            "a_better": int((p[a] > p[b]).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/msc/campaign")
    ap.add_argument("--controls", default="results/msc/controls")
    ap.add_argument("--data", default="data/.cache/msc")
    ap.add_argument("--tables", default="results/msc/tables")
    args = ap.parse_args()

    rows = []
    for seed in SEEDS:
        t = np.load(os.path.join(args.data, f"test_seed{seed}.npz"))
        v = np.load(os.path.join(args.data, f"val_seed{seed}.npz"))
        Xt, yt = torch.tensor(t["X"], dtype=torch.float32), t["y"].astype(int)
        Xv, yv = torch.tensor(v["X"], dtype=torch.float32), v["y"].astype(int)
        dim = Xt.shape[1]

        def add(name, model):
            rows.append({"seed": seed, "model": name,
                         **score(predict(model, Xt), yt, predict(model, Xv), yv)})

        for arm in ("naive", "sisa"):
            ckpt = os.path.join(args.runs, f"{arm}_seed{seed}", "phase4_checkpoints", "round_5.npz")
            if os.path.exists(ckpt):
                add(f"{arm}_phase4", load_model(ckpt, dim))
            ref = sorted(glob.glob(os.path.join(args.controls, f"clean_{arm}_seed{seed}",
                                                "global_checkpoints", "round_*.npz")),
                         key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
            if ref:
                add(f"{arm}_clean", load_model(ref[-1], dim))

        sets = {"constituents_clean": ("clean_sisa", "constituents"),
                "constituents_phase1": ("poisoned_sisa", "phase1_constituents"),
                "constituents_recovered": ("poisoned_sisa", "constituents")}
        for name, (run, sub) in sets.items():
            pattern = os.path.join(args.controls, f"{run}_seed{seed}", sub, "shard*.pt")
            paths = sorted(glob.glob(pattern))
            if len(paths) != 5:
                continue
            states = [torch.load(p, map_location="cpu", weights_only=True)["model"] for p in paths]
            models = [from_state(s, dim) for s in states]
            for i, m in enumerate(models):
                add(f"{name}_shard{i}", m)
            add(f"{name}_average", from_state(averaged(states), dim))
            pt = np.mean([predict(m, Xt) for m in models], axis=0)
            pv = np.mean([predict(m, Xv) for m in models], axis=0)
            rows.append({"seed": seed, "model": f"{name}_ensemble", **score(pt, yt, pv, yv)})

    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(args.tables, "controls_scores.csv"), index=False)

    tests = []
    for arm in ("naive", "sisa"):  # 1. Phase 4 vs clean reference, same arm
        for m in ("f1", "balanced_acc", "balanced_acc_val", "mcc_val", "roc_auc"):
            tests.append(paired(df, f"{arm}_phase4", f"{arm}_clean", m))
    for m in ("balanced_acc", "specificity", "balanced_acc_val", "roc_auc"):  # 2.
        tests.append(paired(df, "naive_clean", "sisa_clean", m))
    for s in ("clean", "phase1", "recovered"):  # 3.
        for m in ("balanced_acc", "balanced_acc_val", "mcc_val", "roc_auc"):
            tests.append(paired(df, f"constituents_{s}_ensemble", f"constituents_{s}_average", m))
    stats = pd.DataFrame([t for t in tests if t])
    stats.round(4).to_csv(os.path.join(args.tables, "controls_stats.csv"), index=False)

    pd.set_option("display.width", 200)
    print("Medians per model:")
    print(df.groupby("model")[["f1", "balanced_acc", "specificity", "balanced_acc_val",
                               "mcc_val", "roc_auc"]].median().round(3).to_string())
    print("\nPaired comparisons (a - b):")
    print(stats.round(4).to_string(index=False))
    h5 = stats[stats.a.str.endswith("_phase4") & stats.metric.isin(["f1", "balanced_acc"])]
    for _, r in h5.iterrows():
        ok = r.diff_median >= -TOLERANCE
        print(f"H5 {r.a} vs {r.b} [{r.metric}]: median diff {r.diff_median:+.3f} -> "
              f"{'within' if ok else 'outside'} the {TOLERANCE} tolerance")


if __name__ == "__main__":
    main()

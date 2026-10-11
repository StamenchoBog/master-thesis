"""Membership-inference check of the unlearning.

    python -m analysis.evaluate_unlearning <model.pt|round.npz> \
        [--cache data/.cache/msc/partition_3_of_4.npz]

Compares the per-sample loss on the removed samples (true labels), held-out rows and retained
training rows. Forgotten data scores like the held-out rows (AUC about 0.5).
"""

import argparse
import json

import numpy as np
import torch

from analysis.evaluate_model import load_model
from edge_nodes.sisa_partition import TRAIN_FRACTION


def per_sample_loss(model, X, y, batch=4096):
    bce = torch.nn.BCELoss(reduction="none")
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            pred = model(torch.from_numpy(X[i:i + batch])).squeeze()
            yb = torch.from_numpy(y[i:i + batch]).float()
            out.append(bce(pred, yb).numpy())
    return np.concatenate(out)


def membership_auc(member_loss, nonmember_loss):
    """AUC that members (lower loss) rank above non-members; 0.5 means indistinguishable."""
    score = np.concatenate([-member_loss, -nonmember_loss])  # higher score = more member-like
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = len(member_loss), len(nonmember_loss)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--cache", default="data/.cache/msc/partition_3_of_4.npz")
    p.add_argument("--n", type=int, default=20000, help="max samples per group")
    args = p.parse_args()

    d = np.load(args.cache)
    X, y, poison = d["X"], d["y"], d["poison_idx"]
    if len(poison) == 0:
        raise SystemExit(f"{args.cache} has no poison indices, nothing was removed.")
    split = int(TRAIN_FRACTION * len(X))
    rng = np.random.default_rng(0)

    def sub(idx):
        return idx if len(idx) <= args.n else rng.choice(idx, args.n, replace=False)

    removed = sub(poison)
    retained = sub(np.setdiff1d(np.arange(split), poison))
    holdout = sub(np.arange(split, len(X)))

    model = load_model(args.model, X.shape[1])
    l_rem = per_sample_loss(model, X[removed], y[removed])
    l_ret = per_sample_loss(model, X[retained], y[retained])
    l_hold = per_sample_loss(model, X[holdout], y[holdout])

    auc_removed = membership_auc(l_rem, l_hold)
    auc_retained = membership_auc(l_ret, l_hold)

    # The probe only means something if it can spot the retained (member) data at all.
    if auc_retained < 0.55:
        verdict = ("inconclusive: the model leaks no membership signal (retained AUC "
                   "≈ 0.5), so MIA cannot confirm or deny; exact unlearning rests on the "
                   "construction, verified bit-identically by tests/smoke_test.py")
    elif auc_removed > 0.55:
        verdict = "still-remembered: removed data is identifiable as a member"
    else:
        verdict = "forgotten: removed indistinguishable from holdout, positive control valid"

    print(json.dumps({
        "model": args.model,
        "n": {"removed": len(removed), "retained": len(retained), "holdout": len(holdout)},
        "mean_loss": {"removed": round(float(l_rem.mean()), 4),
                      "retained": round(float(l_ret.mean()), 4),
                      "holdout": round(float(l_hold.mean()), 4)},
        "mia_auc_removed_vs_holdout": round(float(auc_removed), 4),
        "mia_auc_retained_vs_holdout": round(float(auc_retained), 4),
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()

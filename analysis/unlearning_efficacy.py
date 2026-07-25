"""Empirically verify machine unlearning: has the recovered model *forgotten* the removed data?

    python -m analysis.unlearning_efficacy <model.pt|round.npz> [--cache data/.cache/msc/partition_3_of_4.npz]

Exact unlearning is guaranteed by construction (the recovered SISA constituents never
trained on the removed samples), but a reviewer wants that *demonstrated*, not just
argued. This is a membership-inference test: a model assigns lower loss to samples it
trained on than to unseen ones, so we compare the model's per-sample loss on

  removed   — the unlearned samples (poison indices), scored against their TRUE labels
  holdout   — validation rows never trained on (known non-members)
  retained  — kept training rows (known members; positive control)

For successful unlearning the removed set is indistinguishable from the holdout
(membership-inference AUC ≈ 0.5). Run it on the pre-recovery poisoned model too and
the removed set scores like members (AUC > 0.5) — i.e. the test can tell "forgotten"
from "still remembered".
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis.evaluate_model import load_model  # noqa: E402
from edge_nodes.sisa_partition import TRAIN_FRACTION  # noqa: E402


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
    """AUC that member samples (lower loss ⇒ more member-like) rank above non-members.

    0.5 means the two groups are indistinguishable (forgotten); > 0.5 means members
    are identifiable (remembered). Rank-based Mann–Whitney form, no sklearn needed.
    """
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
        raise SystemExit(f"{args.cache} has no poison indices — nothing was removed.")
    split = int(TRAIN_FRACTION * len(X))
    rng = np.random.default_rng(0)

    def sub(idx):
        return idx if len(idx) <= args.n else rng.choice(idx, args.n, replace=False)

    removed = sub(poison)                                   # unlearned samples (true labels on disk)
    retained = sub(np.setdiff1d(np.arange(split), poison))  # kept training rows (members)
    holdout = sub(np.arange(split, len(X)))                 # never-trained rows (non-members)

    model = load_model(args.model, X.shape[1])
    l_rem = per_sample_loss(model, X[removed], y[removed])
    l_ret = per_sample_loss(model, X[retained], y[retained])
    l_hold = per_sample_loss(model, X[holdout], y[holdout])

    auc_removed = membership_auc(l_rem, l_hold)   # ≈ 0.5 ⇒ removed looks unseen
    auc_retained = membership_auc(l_ret, l_hold)  # > 0.5 ⇒ the probe can see membership at all

    # Interpret honestly against the positive control: the probe is only informative
    # if it can detect membership on the retained (member) data in the first place.
    if auc_retained < 0.55:
        verdict = ("inconclusive — the model leaks no membership signal (retained AUC "
                   "≈ 0.5), so MIA cannot confirm or deny; exact unlearning rests on the "
                   "construction, verified bit-identically by tests/smoke_test.py")
    elif auc_removed > 0.55:
        verdict = "still-remembered — removed data is identifiable as a member"
    else:
        verdict = "forgotten — removed indistinguishable from holdout, positive control valid"

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

"""Evaluate a model against the clean global test set (host-side).

    python -m analysis.evaluate_model <model> [--test data/.cache/msc/test_global.npz]

<model> is either a recovered model (.pt state dict) or a per-round global
checkpoint (.npz array list in state-dict order). Prints JSON with
accuracy/precision/recall/F1 — recall on the attack class doubles as the
attack-success readout (attack success = 1 - recall).
"""

import argparse
import json

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge_nodes.model import IDSModel


def load_model(path: str, input_dim: int) -> IDSModel:
    model = IDSModel(input_dim=input_dim)
    if path.endswith(".npz"):
        z = np.load(path)
        arrays = [z[f] for f in z.files]
        keys = list(model.state_dict().keys())
        model.load_state_dict({k: torch.tensor(a) for k, a in zip(keys, arrays)})
    else:
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.train(False)
    return model


def metrics(model: IDSModel, X: np.ndarray, y: np.ndarray, batch: int = 4096) -> dict:
    tp = fp = fn = correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            pred = (model(torch.from_numpy(X[i:i + batch])).squeeze() > 0.5).long()
            labels = torch.from_numpy(y[i:i + batch]).long()
            correct += (pred == labels).sum().item()
            tp += ((pred == 1) & (labels == 1)).sum().item()
            fp += ((pred == 1) & (labels == 0)).sum().item()
            fn += ((pred == 0) & (labels == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": round(correct / len(y), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-8), 6),
        "attack_success": round(1 - recall, 6),
        "n": len(y),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--test", default="data/.cache/msc/test_global.npz")
    args = p.parse_args()

    data = np.load(args.test)
    X, y = data["X"], data["y"]
    model = load_model(args.model, X.shape[1])
    print(json.dumps({"model": args.model, **metrics(model, X, y)}, indent=2))


if __name__ == "__main__":
    main()

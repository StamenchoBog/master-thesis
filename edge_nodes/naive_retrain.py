"""Naive retraining (Arm A recovery): from-scratch local retrain. Run on the Pi.

    python -m edge_nodes.naive_retrain

The oracle poison mask identifies compromised samples; they are dropped and a
fresh seeded model is trained on all retained local data for NUM_ROUNDS epochs
(the same budget as Phase-1 participation: NUM_ROUNDS rounds x 1 local epoch,
with a fresh Adam per epoch to mirror the FL client's per-round optimizer).
This is the exact-unlearning gold standard at client level — and the expensive
arm: full-dataset compute, sustained load on the fanless Pi.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from .client_app import DEVICE, SEED
from .model import IDSModel
from .sisa_client import CHECKPOINT_DIR, TELEMETRY_DIR, slice_loader, train_slice
from .sisa_partition import TRAIN_FRACTION

NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
PARTITION_ID = int(os.getenv("PARTITION_ID", "2"))
NUM_PARTITIONS = int(os.getenv("NUM_PARTITIONS", "3"))
CACHE_DIR = os.getenv("CACHE_DIR", "/data/.cache/msc")
LR = float(os.getenv("LR", "0.001"))
RECOVERED_MODEL_PATH = os.getenv(
    "RECOVERED_MODEL_PATH", os.path.join(CHECKPOINT_DIR, "recovered_model.pt")
)


def main():
    cache = os.path.join(CACHE_DIR, f"partition_{PARTITION_ID}_of_{NUM_PARTITIONS}.npz")
    data = np.load(cache)
    X, y, poison_idx = data["X"], data["y"], data["poison_idx"]
    if len(poison_idx) == 0:
        sys.exit(f"No poison indices in {cache} — nothing to recover from.")

    split = int(TRAIN_FRACTION * len(X))
    retained = np.setdiff1d(np.arange(split), poison_idx)
    print(f"Retained {len(retained)}/{split} train rows ({len(poison_idx)} poisoned dropped)")

    torch.manual_seed(SEED * 1000 + 999)
    model = IDSModel(input_dim=X.shape[1]).to(DEVICE)
    criterion = torch.nn.BCELoss()

    epoch_log = []
    t_start = time.perf_counter()
    for epoch in range(1, NUM_ROUNDS + 1):
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loader = slice_loader(X, y, retained, SEED * 1_000_000 + 999_000 + epoch)
        t0 = time.perf_counter()
        loss, batches = train_slice(model, optimizer, loader, criterion)
        epoch_log.append({"epoch": epoch, "train_s": round(time.perf_counter() - t0, 4),
                          "loss": round(loss / max(batches, 1), 6)})
        print(f"[epoch {epoch}/{NUM_ROUNDS}] loss={epoch_log[-1]['loss']} in {epoch_log[-1]['train_s']}s")
    total_s = time.perf_counter() - t_start

    os.makedirs(os.path.dirname(RECOVERED_MODEL_PATH), exist_ok=True)
    with open(RECOVERED_MODEL_PATH, "wb") as f:
        torch.save(model.state_dict(), f)
        f.flush()
        os.fsync(f.fileno())

    manifest = {
        "method": "naive",
        "total_s": round(total_s, 3),
        "poisoned_samples": int(len(poison_idx)),
        "retained_samples": int(len(retained)),
        "epochs": NUM_ROUNDS,
        "recovered_model": RECOVERED_MODEL_PATH,
        "epoch_log": epoch_log,
    }
    with open(os.path.join(TELEMETRY_DIR, "recovery_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Naive retraining done in {total_s:.1f}s. Recovered model: {RECOVERED_MODEL_PATH}")


if __name__ == "__main__":
    main()

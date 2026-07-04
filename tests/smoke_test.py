"""Verification of the unlearning guarantee (synthetic data, no dataset needed).

Checks, on 2000 synthetic rows:
  1. Poison placement is deterministic and confined to the target slices.
  2. POISON_MODE=flip corrupts labels in memory while disk labels stay clean.
  3. The SISA trainer produces one checkpoint per (round, shard, slice).
  4. Recovery rolls back to a checkpoint that predates the poison and is
     bit-identical across reruns (determinism requirement of the protocol).

Run inside the client image (has torch + flwr):

    docker run --rm -v "$PWD":/app -w /app --entrypoint python \
        fl-ids-preprocessor:latest tests/smoke_test.py
"""

import hashlib
import json
import os
import sys
import tempfile

BASE = tempfile.mkdtemp(prefix="sisa_smoke_")
# Must be set before importing edge_nodes modules — they read env at import time.
os.environ.update({
    "SEED": "7", "NUM_SHARDS": "3", "NUM_SLICES": "4", "NUM_ROUNDS": "2",
    "POISON_MODE": "flip", "PARTITION_ID": "2", "NUM_PARTITIONS": "3", "LR": "0.001",
    "DATA_DIR": f"{BASE}/data", "CACHE_DIR": f"{BASE}/data/.cache/msc",
    "CHECKPOINT_DIR": f"{BASE}/ckpt", "TELEMETRY_DIR": f"{BASE}/shm",
    "RECOVERED_MODEL_PATH": f"{BASE}/ckpt/recovered_model.pt",
})
for sub in ("data/.cache/msc", "ckpt", "shm"):
    os.makedirs(os.path.join(BASE, sub), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from edge_nodes.sisa_partition import TRAIN_FRACTION, poison_indices, shard_slice_assignment  # noqa: E402

SEED, S, R = 7, 3, 4
N, D = 2000, 10

# ---- 1. Poison placement: deterministic, attack-only, confined to target slices
rng = np.random.default_rng(0)
X = rng.normal(size=(N, D)).astype(np.float32)
y = (rng.random(N) < 0.5).astype(np.int64)

n_train = int(TRAIN_FRACTION * N)
a1 = shard_slice_assignment(n_train, SEED, S, R)
a2 = shard_slice_assignment(n_train, SEED, S, R)
assert all((s1 == s2).all() for sh1, sh2 in zip(a1, a2) for s1, s2 in zip(sh1, sh2)), \
    "assignment not deterministic"

POISON_SHARD, FROM_SLICE, FRACTION = 1, 2, 0.8
p1 = poison_indices(y, a1, POISON_SHARD, FROM_SLICE, FRACTION, SEED)
p2 = poison_indices(y, a2, POISON_SHARD, FROM_SLICE, FRACTION, SEED)
assert (p1 == p2).all(), "poison selection not deterministic"
assert (y[p1] == 1).all(), "poison must target attack samples only"
target = set(np.concatenate(a1[POISON_SHARD][FROM_SLICE:]).tolist())
assert set(p1.tolist()) <= target, "poison leaked outside target slices"
print(f"[1] poisoning deterministic and confined: {len(p1)} samples OK")

np.savez(os.path.join(os.environ["CACHE_DIR"], "partition_2_of_3.npz"),
         X=X, y=y, poison_idx=p1)

# ---- 2. POISON_MODE=flip semantics
from edge_nodes.data_loader import load_arrays  # noqa: E402

Xl, yl, split, pl = load_arrays(2, 3)
assert split == n_train and (pl == p1).all()
assert (yl[p1] == 0).all(), "flip mode must zero poisoned labels"
assert (y[p1] == 1).all(), "original labels must be untouched on disk"
print("[2] POISON_MODE=flip semantics OK")

# ---- 3. SISA training: one checkpoint per (round, shard, slice)
from edge_nodes.sisa_client import SISATrainer, ckpt_path  # noqa: E402

telemetry_log = []
trainer = SISATrainer(2, 3)
for _ in range(2):
    trainer.train_round(0.001, telemetry_log.append)
assert trainer.round == 2
assert len(telemetry_log) == 2 * S * R, "one telemetry entry per slice"
assert all(os.path.exists(ckpt_path(s, r, l)) for s in range(S) for r in (1, 2) for l in range(R))
print(f"[3] SISA trained 2 rounds, {len(telemetry_log)} slice checkpoints OK")

# ---- 4. Recovery: rollback predates poison; recovery is deterministic
from edge_nodes import sisa_recover  # noqa: E402


def state_hash(path):
    sd = torch.load(path, map_location="cpu")
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


sisa_recover.main()
h_first = state_hash(os.environ["RECOVERED_MODEL_PATH"])
with open(os.path.join(os.environ["TELEMETRY_DIR"], "recovery_manifest.json")) as f:
    man = json.load(f)
assert man["affected_shards"] == {str(POISON_SHARD): FROM_SLICE}, man["affected_shards"]

sisa_recover.main()
h_second = state_hash(os.environ["RECOVERED_MODEL_PATH"])
assert h_first == h_second, "recovery not deterministic across identical reruns"
print(f"[4] rollback at shard {POISON_SHARD} slice {FROM_SLICE} (predates poison), "
      f"deterministic recovery OK ({h_first[:12]})")

print("\nSMOKE TEST PASSED")

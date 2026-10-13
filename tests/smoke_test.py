"""Checks the unlearning guarantee on 2000 synthetic rows (no dataset
needed): deterministic, confined poison placement; flip in memory only;
one checkpoint per (round, shard, slice); rollback before the poison and
a bit-identical recovery; a rollback state identical to a never-poisoned
node and a replay that never reads a poisoned row.

    docker run --rm -v "$PWD":/app -w /app --entrypoint python \
        fl-ids-preprocessor:latest tests/smoke_test.py
"""

import hashlib
import json
import os
import sys
import tempfile

BASE = tempfile.mkdtemp(prefix="sisa_smoke_")
# Must be set before importing edge_nodes modules, since they read env
# at import time.
os.environ.update(
    {
        "SEED": "7",
        "NUM_SHARDS": "3",
        "NUM_SLICES": "4",
        "NUM_ROUNDS": "2",
        "POISON_MODE": "flip",
        "PARTITION_ID": "2",
        "NUM_PARTITIONS": "3",
        "LR": "0.001",
        "DATA_DIR": f"{BASE}/data",
        "CACHE_DIR": f"{BASE}/data/.cache/msc",
        "CHECKPOINT_DIR": f"{BASE}/ckpt",
        "TELEMETRY_DIR": f"{BASE}/shm",
        "RECOVERED_MODEL_PATH": f"{BASE}/ckpt/recovered_model.pt",
    }
)
for sub in ("data/.cache/msc", "ckpt", "shm"):
    os.makedirs(os.path.join(BASE, sub), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from edge_nodes.sisa_partition import (  # noqa: E402
    TRAIN_FRACTION,
    poison_indices,
    shard_slice_assignment,
)

SEED, S, R = 7, 3, 4
N, D = 2000, 10

# ---- 1. Poison placement: deterministic, attack-only, confined
rng = np.random.default_rng(0)
X = rng.normal(size=(N, D)).astype(np.float32)
y = (rng.random(N) < 0.5).astype(np.int64)

n_train = int(TRAIN_FRACTION * N)
a1 = shard_slice_assignment(n_train, SEED, S, R)
a2 = shard_slice_assignment(n_train, SEED, S, R)
assert all(
    (s1 == s2).all() for sh1, sh2 in zip(a1, a2) for s1, s2 in zip(sh1, sh2)
), "assignment not deterministic"

POISON_SHARD, FROM_SLICE, FRACTION = 1, 2, 0.8
p1 = poison_indices(y, a1, POISON_SHARD, FROM_SLICE, FRACTION, SEED)
p2 = poison_indices(y, a2, POISON_SHARD, FROM_SLICE, FRACTION, SEED)
assert (p1 == p2).all(), "poison selection not deterministic"
assert (y[p1] == 1).all(), "poison must target attack samples only"
target = set(np.concatenate(a1[POISON_SHARD][FROM_SLICE:]).tolist())
assert set(p1.tolist()) <= target, "poison leaked outside target slices"
print(f"[1] poisoning deterministic and confined: {len(p1)} samples OK")

np.savez(
    os.path.join(os.environ["CACHE_DIR"], "partition_2_of_3.npz"),
    X=X,
    y=y,
    poison_idx=p1,
)

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
assert all(
    os.path.exists(ckpt_path(s, r, sl))
    for s in range(S)
    for r in (1, 2)
    for sl in range(R)
)
print(f"[3] SISA trained 2 rounds, {len(telemetry_log)} slice checkpoints OK")


def ckpt_hash(path):
    """Model and optimizer state: recovery restores both."""
    ckpt = torch.load(path, map_location="cpu")
    h = hashlib.sha256()
    for k in sorted(ckpt["model"]):
        h.update(ckpt["model"][k].numpy().tobytes())
    for idx in sorted(ckpt["optim"]["state"]):
        for k in sorted(ckpt["optim"]["state"][idx]):
            h.update(ckpt["optim"]["state"][idx][k].numpy().tobytes())
    return h.hexdigest()


# Recovery overwrites the checkpoints from the poisoned slice on, so
# hash them now.
poisoned_rollback = ckpt_hash(ckpt_path(POISON_SHARD, 1, FROM_SLICE - 1))
poisoned_first = ckpt_hash(ckpt_path(POISON_SHARD, 1, FROM_SLICE))

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
with open(
    os.path.join(os.environ["TELEMETRY_DIR"], "recovery_manifest.json")
) as f:
    man = json.load(f)
assert man["affected_shards"] == {str(POISON_SHARD): FROM_SLICE}, man[
    "affected_shards"
]

sisa_recover.main()
h_second = state_hash(os.environ["RECOVERED_MODEL_PATH"])
assert (
    h_first == h_second
), "recovery not deterministic across identical reruns"
print(
    f"[4] rollback at shard {POISON_SHARD} slice {FROM_SLICE} (predates "
    "poison), "
    f"deterministic recovery OK ({h_first[:12]})"
)

# ---- 5. Exactness: clean rollback state, replay without poison
from edge_nodes import data_loader, sisa_client  # noqa: E402

# The same node with the true labels: if the rollback state matches bit
# for bit, the poisoned rows had no influence on it.
clean_dir = os.path.join(BASE, "ckpt_clean")
os.makedirs(clean_dir)
sisa_client.CHECKPOINT_DIR = clean_dir
os.environ["POISON_MODE"] = data_loader.POISON_MODE = "off"
data_loader._arrays_cache.clear()
SISATrainer(2, 3).train_round(0.001, lambda _: None)
clean_rollback = ckpt_hash(
    sisa_client.ckpt_path(POISON_SHARD, 1, FROM_SLICE - 1)
)
clean_first = ckpt_hash(sisa_client.ckpt_path(POISON_SHARD, 1, FROM_SLICE))
sisa_client.CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]
assert (
    poisoned_rollback == clean_rollback
), "rollback checkpoint depends on the poison"
# and the comparison can see the poison: the first poisoned slice must
# differ
assert poisoned_first != clean_first, "check cannot tell poisoned from clean"

seen = []
real_loader = sisa_recover.slice_loader


def recording_loader(X_, y_, idx, seed_key, *args, **kwargs):
    seen.append(idx)
    return real_loader(X_, y_, idx, seed_key, *args, **kwargs)


sisa_recover.slice_loader = recording_loader
sisa_recover.main()
sisa_recover.slice_loader = real_loader
replayed = np.concatenate(seen)
assert not set(replayed.tolist()) & set(
    p1.tolist()
), "replay read a poisoned row"
print(
    f"[5] rollback checkpoint identical to a never-poisoned node, replay of "
    f"{len(replayed)} rows without poison OK"
)

print("\nSMOKE TEST PASSED")

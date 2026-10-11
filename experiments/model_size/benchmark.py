"""Model-size micro-benchmark, run inside the client image on the Pi.

    python -m experiments.model_size.benchmark --width 384 --rep 1

For one model width: training throughput with the recovery's own slice code (rows/s over the
last 120 s of --train-s), checkpoint write and read time (fsync, --ckpts each) and peak RAM
with 5 constituents in memory. Each block writes its name to /dev/shm/run_marker.
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import torch
import torch.nn as nn

from edge_nodes.model import IDSModel
from edge_nodes.sisa_client import slice_loader, train_slice
from edge_nodes.sisa_partition import TRAIN_FRACTION

CACHE_DIR = os.getenv("CACHE_DIR", "/data/.cache/msc")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/checkpoints")
TELEMETRY_DIR = os.getenv("TELEMETRY_DIR", "/dev/shm")
MARKER = os.getenv("RUN_MARKER", "/dev/shm/run_marker")
S, R = 5, 5  # shards x slices, as in the campaign


def mark(label: str):
    with open(MARKER, "w") as f:
        f.write(label + "\n")
    return time.time()


def train_block(model, X, y, seconds: float, lr: float, seed: int):
    """Returns per-slice (end time, rows, seconds) until `seconds` have passed."""
    criterion = nn.BCELoss()
    order = np.random.default_rng(seed).permutation(len(X))
    slices = np.array_split(order, S * R)  # same slice size as the campaign's SISA layout
    log, start, k = [], time.perf_counter(), 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    while time.perf_counter() - start < seconds:
        if k % R == 0:  # fresh Adam per shard-round, as in SISA training and recovery
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        idx = slices[k % len(slices)]
        t0 = time.perf_counter()
        loader = slice_loader(X, y, idx, seed * 1000 + k)
        train_slice(model, optimizer, loader, criterion)
        log.append((time.time(), len(idx), time.perf_counter() - t0))
        k += 1
    return optimizer, log


def checkpoint_block(model, optimizer, n: int, tag: str):
    paths = [os.path.join(CHECKPOINT_DIR, f"bench_{tag}_{i}.pt") for i in range(n)]
    writes, reads = [], []
    for p in paths:
        t0 = time.perf_counter()
        with open(p, "wb") as f:
            torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()}, f)
            f.flush()
            os.fsync(f.fileno())
            # Evict it from the page cache so the reads below hit the card. Needs no root,
            # unlike drop_caches, which the container user can't use.
            if hasattr(os, "posix_fadvise"):  # Linux only; macOS is just for testing
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        writes.append(time.perf_counter() - t0)
    size = os.path.getsize(paths[0])
    for p in paths:
        t0 = time.perf_counter()
        torch.load(p, map_location="cpu", weights_only=True)
        reads.append(time.perf_counter() - t0)
    for p in paths:
        os.remove(p)
    return writes, reads, size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--train-s", type=float, default=300.0)
    ap.add_argument("--tail-s", type=float, default=120.0)
    ap.add_argument("--ckpts", type=int, default=20)
    ap.add_argument("--partition", default="partition_3_of_4.npz")
    ap.add_argument("--ckpt-only", action="store_true",
                    help="power block for checkpoint writes: one training slice for a real Adam "
                         "state, then checkpoint writes for --ckpt-s seconds (no timing claims)")
    ap.add_argument("--ckpt-s", type=float, default=60.0)
    args = ap.parse_args()

    torch.manual_seed(1000 * args.rep + args.width)
    data = np.load(os.path.join(CACHE_DIR, args.partition))
    split = int(TRAIN_FRACTION * len(data["y"]))
    X, y = data["X"][:split], data["y"][:split]

    model = IDSModel(X.shape[1], width=args.width)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"w{args.width}_r{args.rep}"
    print(f"[{tag}] {n_params} parameters")

    if args.ckpt_only:
        # One slice with the recovery's own code, so the checkpoints carry a real Adam state.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        one_slice = np.random.default_rng(args.rep).permutation(len(X))[: len(X) // (S * R)]
        train_slice(model, optimizer, slice_loader(X, y, one_slice, args.rep), nn.BCELoss())
        t0 = mark(f"bench_ckptonly_w{args.width}")
        writes, n = [], 0
        while time.time() - t0 < args.ckpt_s:
            w, _, size = checkpoint_block(model, optimizer, 1, f"{tag}_{n}")
            writes += w
            n += 1
        t1 = mark("idle")
        path = os.path.join(TELEMETRY_DIR, f"model_size_ckptonly_w{args.width}.json")
        with open(path, "w") as f:
            json.dump({"width": args.width, "params": n_params, "ckpt_bytes": size,
                       "writes": len(writes), "ckpt_write_s": writes,
                       "blocks": {"ckpt": t0, "end": t1}}, f)
        print(f"[{tag}] checkpoint-only block: {len(writes)} writes in {t1 - t0:.0f} s -> {path}")
        return

    t_train = mark(f"bench_train_{tag}")
    optimizer, log = train_block(model, X, y, args.train_s, 0.001, args.rep)
    t_ckpt = mark(f"bench_ckpt_{tag}")
    writes, reads, size = checkpoint_block(model, optimizer, args.ckpts, tag)
    t_ram = mark(f"bench_ram_{tag}")
    constituents = [IDSModel(X.shape[1], width=args.width) for _ in range(4)] + [model]
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux
    t_end = mark("idle")
    del constituents

    tail = [r for r in log if r[0] >= log[-1][0] - args.tail_s]
    rows_per_s = sum(r[1] for r in tail) / max(sum(r[2] for r in tail), 1e-9)
    out = {
        "width": args.width, "rep": args.rep, "params": n_params,
        "train_s": args.train_s, "slices": len(log),
        "rows_per_s_tail": rows_per_s,
        "rows_per_s_all": sum(r[1] for r in log) / max(sum(r[2] for r in log), 1e-9),
        "ckpt_bytes": size,
        "ckpt_write_s": writes, "ckpt_read_s": reads,
        "ckpt_write_median_s": float(np.median(writes)),
        "ckpt_read_median_s": float(np.median(reads)),
        "peak_rss_mb_5_constituents": rss_mb,
        "blocks": {"train": t_train, "ckpt": t_ckpt, "ram": t_ram, "end": t_end},
        "slice_log": log,
    }
    path = os.path.join(TELEMETRY_DIR, f"model_size_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[{tag}] {rows_per_s:.0f} rows/s, ckpt {size / 1e6:.2f} MB "
          f"write {out['ckpt_write_median_s']:.3f}s read {out['ckpt_read_median_s']:.3f}s, "
          f"RSS {rss_mb:.0f} MB -> {path}")


if __name__ == "__main__":
    main()

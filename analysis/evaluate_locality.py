"""Poisoning-locality sweep (experiments/protocol.md, Addendum B): measurements vs cost model.

    python -m analysis.evaluate_locality

Reads the recovery manifests in results/msc/locality/ (one per configuration and
repetition), the session's power log, phase log and Pi telemetry, and the Addendum A
measurements at width 128 (results/msc/tables/model_size_measurements.csv). Then:

1. per recovery: time to recovery, replayed slices, sample updates (counted exactly from
   the placement and the slice assignment), net energy and live-throttled seconds
2. B1: the cost model T = U / thr + N * t_ckpt + t_load against every configuration (15 %)
3. B2: slice 0 vs slice 4 of one shard (median ratio, < 1.2)
4. B3: recovery time against the number of affected shards, and k = 5 against naive
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge_nodes.sisa_partition import TRAIN_FRACTION, shard_slice_assignment  # noqa: E402

SEED, S, R = 42, 5, 5
TOLERANCE_B1 = 0.15


def sample_updates(manifest, placement_path):
    """Rows actually trained on in this recovery, from its replay list and the placement."""
    d = np.load(placement_path)
    n_train = int(TRAIN_FRACTION * len(d["y"]))
    poison = set(d["poison_idx"].tolist())
    if manifest["method"] == "naive":
        return manifest["epochs"] * (n_train - len(poison))
    assignment = shard_slice_assignment(n_train, SEED, S, R)
    clean = {(s, l): sum(1 for i in assignment[s][l] if i not in poison) for s in range(S) for l in range(R)}
    return sum(clean[(e["shard"], e["slice"])] for e in manifest["slices"])


def window_stats(run_dir, rows):
    """Net energy and live-throttled seconds per recovery window (marker recover_<cfg>_r<rep>)."""
    # Window edges come from phases.log on the host's clock, the same clock as the power log.
    marks = []
    for line in open(os.path.join(run_dir, "phases.log")):
        parts = line.split()
        if len(parts) == 2 and not parts[0].startswith("clock_offset"):
            marks.append((float(parts[0]), parts[1]))
    windows = {lab: (t, marks[i + 1][0]) for i, (t, lab) in enumerate(marks[:-1]) if lab.startswith("recover_")}

    p = pd.read_csv(os.path.join(run_dir, "power_fnb58.csv"), sep=r"\s+", comment="#",
                    names=["timestamp", "sample", "v", "i", "dp", "dn", "temp", "energy", "capacity"],
                    skiprows=1, on_bad_lines="skip")
    p = p[pd.to_numeric(p["timestamp"], errors="coerce").notna()].astype(float)
    p["w"] = p["v"] * p["i"]
    tel = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(run_dir, "hardware_telemetry_*.csv")))]
    tel = pd.concat(tel, ignore_index=True) if tel else None
    for r in rows:
        lab = f"recover_{r['config']}_r{r['rep']}"
        if lab not in windows:
            continue
        a, b = windows[lab]
        seg = p[(p.timestamp >= a) & (p.timestamp < b)]
        pre = p[(p.timestamp >= a - 60) & (p.timestamp < a - 5)]
        cover = (seg.timestamp.max() - seg.timestamp.min()) / (b - a) if len(seg) > 1 else 0
        if cover >= 0.95 and len(pre) > 1000:
            gross = np.trapezoid(seg.w, seg.timestamp) / 3600
            r["energy_net_wh"] = gross - pre.w.mean() * (b - a) / 3600
        r["power_cover"] = cover
        if tel is not None:
            t = tel[tel["Marker"].astype(str) == lab]
            bits = t["Throttled"].astype(str).apply(lambda v: int(v, 16) if v.startswith("0x") else 0)
            r["throttled_s"] = int((bits & 0b1110).astype(bool).sum())
            r["undervolt_s"] = int((bits & 0b1).astype(bool).sum())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/msc/locality")
    ap.add_argument("--placements", default="data/.cache/locality")
    ap.add_argument("--tables", default="results/msc/tables")
    args = ap.parse_args()

    ms = pd.read_csv(os.path.join(args.tables, "model_size_measurements.csv"))
    ms = ms[ms.width == 128]
    thr = ms.rows_per_s_tail.median()
    t_ckpt, t_load = ms.ckpt_write_median_s.median(), ms.ckpt_read_median_s.median()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*_r[0-9].json"))):
        name = os.path.basename(path)[:-5]
        cfg, rep = name.rsplit("_r", 1)
        m = json.load(open(path))
        place = "k1_l3" if cfg == "naive" else cfg
        u = sample_updates(m, os.path.join(args.placements, place, "partition_3_of_4.npz"))
        n_ckpt = m.get("retrained_slices", 0)
        n_load = len(m.get("affected_shards", {}))
        pred = u / thr + n_ckpt * t_ckpt + n_load * t_load
        rows.append({"config": cfg, "rep": int(rep), "ttr_s": m["total_s"], "replayed_slices": n_ckpt,
                     "sample_updates": u, "ckpt_io_s": m.get("ckpt_io_s", 0.0), "predicted_s": pred})
    rows = window_stats(args.dir, rows)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(args.tables, "locality_runs.csv"), index=False)

    med = df.groupby("config").median(numeric_only=True)
    med["model_error"] = (med.predicted_s - med.ttr_s).abs() / med.ttr_s
    med.round(4).to_csv(os.path.join(args.tables, "locality_summary.csv"))
    pd.set_option("display.width", 200)
    print(f"Throughput {thr:.0f} rows/s, checkpoint write {t_ckpt * 1e3:.1f} ms, read {t_load * 1e3:.1f} ms (Addendum A, width 128)\n")
    print(med.round(3).to_string())

    print(f"\nB1  model within {TOLERANCE_B1:.0%} for every configuration: "
          f"{bool((med.model_error <= TOLERANCE_B1).all())} (worst {med.model_error.max():.1%})")
    if {"k1_l0", "k1_l4"} <= set(med.index):
        ratio = med.loc["k1_l0", "ttr_s"] / med.loc["k1_l4", "ttr_s"]
        print(f"B2  slice 0 vs slice 4: {ratio:.3f} -> {'holds' if ratio < 1.2 else 'fails'} (< 1.2)")
    ks = [(k, f"k{k}_l3") for k in (1, 2, 3, 5)]
    pts = [(k, med.loc[c, "ttr_s"]) for k, c in ks if c in med.index]
    if len(pts) >= 3:
        k, t = np.array(pts).T
        slope, icept = np.polyfit(k, t, 1)
        r2 = 1 - ((t - (slope * k + icept)) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        print(f"B3  TTR = {slope:.1f} s per shard + {icept:.1f} s (R^2 = {r2:.3f})")
    if {"k5_l3", "naive"} <= set(med.index):
        su = med.loc["naive", "ttr_s"] / med.loc["k5_l3", "ttr_s"]
        print(f"B3  k = 5 speed-up over naive: {su:.2f} -> {'holds' if su <= 1 else 'fails'} (<= 1)")


if __name__ == "__main__":
    main()

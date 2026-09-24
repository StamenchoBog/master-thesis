"""Aggregate MSc experiment runs into a summary table and paired statistics.

    python -m analysis.analyze_runs [--runs results/msc/runs]

Per run directory (layout described in the README) it reads:
  recovery_manifest.json      TTR, checkpoint I/O (SISA)
  hardware_telemetry_*.csv    temp/throttle/iowait/SD-I/O traces + phase markers
  power_fnb58.csv + phases.log  FNB58 power samples, integrated per phase window
  sisa_timings.jsonl          Phase-1 SISA overhead (per-slice train + ckpt I/O)
  results_phase{1,4}.json     per-round global F1/recall

Writes summary.csv and paired_stats.csv next to the runs. Per metric: medians + IQR,
Wilcoxon signed-rank p, Cliff's delta, and a bootstrap 95% CI on the paired difference
— with small N the effect size and CI carry the argument, not p. Throttling counts only
the *live* flag bits (the "occurred" bits are sticky); energy is net of the idle baseline.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["ttr_s", "p3_energy_net_wh", "p3_throttled_s", "p3_min_clock_mhz",
           "p3_sd_written_mb", "p1_ckpt_bytes", "p4_final_f1", "p4_final_recall"]

PHASE1_ROUNDS = 10  # rounds above this in sisa_timings.jsonl belong to the Phase-4 rejoin
NUM_SHARDS = NUM_SLICES = 5
POISON_FROM_SLICE = 3


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size: P(a>b) - P(a<b) over all pairs."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def active_throttle(hexflag) -> tuple:
    """(thermal_now, undervoltage_now) from a vcgencmd get_throttled hex string.

    Bits 1/2/3 = capped/throttled/soft-limit *now*; bit 0 = under-voltage *now*.
    Bits 16-19 are sticky "occurred since boot" flags — ignored, or throttling
    would be massively over-reported.
    """
    try:
        v = int(str(hexflag), 16)
    except (ValueError, TypeError):
        return False, False
    return bool(v & 0b1110), bool(v & 0b1)


def _phase_windows(run_dir: str) -> list:
    """Read phases.log into [(label, start_epoch, end_epoch)] for the measured phases.

    Each phase runs from its marker until the next marker of any kind (usually an
    `idle`/`done`), so the window bounds the actual workload.
    """
    path = os.path.join(run_dir, "phases.log")
    if not os.path.exists(path):
        return []
    marks = sorted((float(p[0]), p[1]) for line in open(path)
                   if len(p := line.split()) == 2)
    windows = []
    for i, (ts, label) in enumerate(marks):
        if label in ("phase1", "phase3", "phase4"):
            end = marks[i + 1][0] if i + 1 < len(marks) else ts
            windows.append((label, ts, end))
    return windows


def _idle_power_w(p: pd.DataFrame, run_dir: str):
    """Median power over the run's `idle` marker windows — the baseline draw of the
    whole Pi doing nothing, used to report *net* (marginal) recovery energy."""
    path = os.path.join(run_dir, "phases.log")
    if not os.path.exists(path):
        return None
    marks = sorted((float(x[0]), x[1]) for line in open(path) if len(x := line.split()) == 2)
    segs = []
    for i, (ts, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else ts
        if label == "idle" and end > ts:
            segs.append(p[(p["timestamp"] >= ts) & (p["timestamp"] < end)]["power_w"])
    allidle = pd.concat(segs) if segs else pd.Series(dtype=float)
    return float(allidle.median()) if len(allidle) else None


def parse_power(run_dir: str, row: dict) -> None:
    """Per-phase energy (Wh, gross and idle-subtracted net) and power (W) from the FNB58 log.

    Energy is the trapezoidal integral of V*I over each phase window (not the device's
    cumulative counter, which a mid-run logger restart would zero). Net energy subtracts
    the idle baseline so H2 reports the *marginal* cost, not the whole idling board.
    """
    path = os.path.join(run_dir, "power_fnb58.csv")
    windows = _phase_windows(run_dir)
    if not os.path.exists(path) or not windows:
        return
    p = pd.read_csv(path, sep=r"\s+")  # the log starts with a blank line; read_csv skips it
    if "timestamp" not in p.columns or len(p) < 2:
        return
    p["power_w"] = p["voltage_V"] * p["current_A"]
    lo, hi = p["timestamp"].min(), p["timestamp"].max()
    idle_w = _idle_power_w(p, run_dir)
    if idle_w is not None:
        row["idle_power_w"] = round(idle_w, 3)
    for label, start, end in windows:
        n = label[-1]
        seg = p[(p["timestamp"] >= start) & (p["timestamp"] < end)]
        if len(seg) < 2:
            print(f"  [warn] {os.path.basename(run_dir)}: no power coverage for {label} "
                  f"(window {start:.0f}-{end:.0f}, log spans {lo:.0f}-{hi:.0f})", file=sys.stderr)
            continue
        gross_wh = float(np.trapezoid(seg["power_w"], seg["timestamp"]) / 3600.0)
        dur_s = float(seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0])
        row[f"p{n}_energy_wh"] = round(gross_wh, 4)
        row[f"p{n}_mean_w"] = round(float(seg["power_w"].mean()), 3)
        row[f"p{n}_peak_w"] = round(float(seg["power_w"].max()), 3)
        if idle_w is not None:
            row[f"p{n}_energy_net_wh"] = round(gross_wh - idle_w * dur_s / 3600.0, 4)


def parse_run(run_dir: str) -> dict:
    """Extract one run's scalar outcomes from its artifacts."""
    row = {"run": os.path.basename(run_dir)}
    m = re.match(r"(naive|sisa)_seed(\d+)", row["run"])
    if not m:
        return {}
    row["arm"], row["seed"] = m.group(1), int(m.group(2))

    manifest = os.path.join(run_dir, "recovery_manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            man = json.load(f)
        row["ttr_s"] = man["total_s"]
        row["recovery_ckpt_io_s"] = man.get("ckpt_io_s", 0.0)
        row["recovery_ckpt_bytes"] = man.get("ckpt_bytes", 0)
        row["poisoned_samples"] = man.get("poisoned_samples")
        row["retained_samples"] = man.get("retained_samples")
        row["epochs"] = man.get("epochs")
        row["retrained_slices"] = man.get("retrained_slices")
        if "slices" in man:
            # No leading underscore: itertuples() renames those columns.
            row["depleted_slices"] = sum(1 for s in man["slices"]
                                         if s["slice"] >= POISON_FROM_SLICE)
            row["full_slices"] = len(man["slices"]) - row["depleted_slices"]

    telemetry = sorted(glob.glob(os.path.join(run_dir, "hardware_telemetry_*.csv")))
    if telemetry:
        t = pd.read_csv(telemetry[-1])
        flags = t["Throttled"].map(active_throttle)
        t["_thermal"] = flags.map(lambda x: x[0])
        t["_undervolt"] = flags.map(lambda x: x[1])
        row["peak_temp_c"] = t["Temp_C"].max()
        row["undervolt_s"] = int(t["_undervolt"].sum())  # should be 0 with the 5 A cable
        p3 = t[t["Marker"].astype(str).str.startswith("phase3")]
        if len(p3):
            row["p3_throttled_s"] = int(p3["_thermal"].sum())
            row["p3_min_clock_mhz"] = int(p3["CPU_Freq_MHz"].min())
            row["p3_mean_clock_mhz"] = int(round(p3["CPU_Freq_MHz"].mean()))
            row["p3_iowait_mean_pct"] = round(p3["IOWait_Pct"].mean(), 2)
            row["p3_sd_written_mb"] = round(p3["SD_Write_kBps"].sum() / 1024, 1)
            row["p3_peak_temp_c"] = p3["Temp_C"].max()
            if "Ambient_C" in p3:
                amb = pd.to_numeric(p3["Ambient_C"], errors="coerce").mean()
                if pd.notna(amb):
                    row["p3_ambient_c"] = round(float(amb), 1)

    sisa_log = os.path.join(run_dir, "sisa_timings.jsonl")
    if os.path.exists(sisa_log):
        # The jsonl is append-only: it also holds Phase-4 rounds (11-15) and duplicates
        # from re-runs. Keep Phase 1 only, deduped by (round, shard, slice).
        entries = [json.loads(line) for line in open(sisa_log)]
        p1 = {(e["round"], e["shard"], e["slice"]): e
              for e in entries if e["round"] <= PHASE1_ROUNDS}
        row["p1_ckpt_io_s"] = round(sum(e["ckpt_s"] for e in p1.values()), 2)
        row["p1_ckpt_bytes"] = sum(e["ckpt_bytes"] for e in p1.values())

    for phase in (1, 4):
        path = os.path.join(run_dir, f"results_phase{phase}.json")
        if os.path.exists(path):
            with open(path) as f:
                res = json.load(f)
            if res["f1"]:
                row[f"p{phase}_final_f1"] = res["f1"][-1]
                row[f"p{phase}_final_recall"] = res["recall"][-1]

    parse_power(run_dir, row)
    return row


def _bootstrap_ci(diffs, n=10000, seed=0) -> tuple:
    """95% bootstrap CI for the median paired difference (naive - sisa).

    With small N the CI on the effect communicates uncertainty far better than a
    p-value; the resample is seeded so the reported interval is reproducible.
    """
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(diffs, size=(n, len(diffs)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def add_work_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sample-updates performed during recovery, and the resulting throughput.

    Quantifies "SISA is faster because it does less work" (H1): if the time ratio
    exceeds the work ratio, the surplus is a physical (thermal) effect a FLOP count
    would miss. naive = retained rows x epochs; sisa = full replayed slices x slice
    size, plus depleted (poisoned) slices at whatever survived in them — these recur
    every round, so at S=R=5 with 47 replayed slices it's 27 full + 20 depleted.
    """
    pools = {r.seed: r.retained_samples + r.poisoned_samples
             for r in df.itertuples() if r.arm == "naive"
             and pd.notna(r.retained_samples) and pd.notna(r.poisoned_samples)}

    def work(r):
        pool = pools.get(r.seed)
        if pool is None or pd.isna(r.poisoned_samples):
            return np.nan
        if r.arm == "naive":
            return r.retained_samples * r.epochs
        slice_rows = pool / (NUM_SHARDS * NUM_SLICES)
        n_poisoned_slices = NUM_SLICES - POISON_FROM_SLICE
        survived = n_poisoned_slices * slice_rows - r.poisoned_samples
        return (r.full_slices * slice_rows
                + r.depleted_slices * survived / n_poisoned_slices)

    df["recovery_sample_updates"] = [work(r) for r in df.itertuples()]
    df["recovery_rows_per_s"] = (df["recovery_sample_updates"] / df["ttr_s"]).round(0)
    return df.drop(columns=[c for c in ("full_slices", "depleted_slices") if c in df])


def paired_stats(df: pd.DataFrame, metric: str) -> dict | None:
    """Wilcoxon signed-rank + Cliff's delta + bootstrap CI for one metric, paired by seed."""
    wide = df.pivot_table(index="seed", columns="arm", values=metric).dropna()
    if len(wide) < 2 or "naive" not in wide or "sisa" not in wide:
        return None
    naive, sisa = wide["naive"], wide["sisa"]
    try:
        _, p = wilcoxon(naive, sisa)
    except ValueError:  # wilcoxon raises when every difference is zero
        p = 1.0
    ci_lo, ci_hi = _bootstrap_ci((naive - sisa).values)
    return {
        "metric": metric, "n_pairs": len(wide),
        "naive_median": round(float(naive.median()), 3),
        "naive_iqr": round(float(naive.quantile(0.75) - naive.quantile(0.25)), 3),
        "sisa_median": round(float(sisa.median()), 3),
        "sisa_iqr": round(float(sisa.quantile(0.75) - sisa.quantile(0.25)), 3),
        "diff_median": round(float((naive - sisa).median()), 3),
        "diff_ci95_lo": round(ci_lo, 3),
        "diff_ci95_hi": round(ci_hi, 3),
        "wilcoxon_p": round(float(p), 4),
        "cliffs_delta": round(cliffs_delta(naive, sisa), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/msc/runs")
    args = p.parse_args()

    rows = [r for d in sorted(glob.glob(os.path.join(args.runs, "*"))) if (r := parse_run(d))]
    if not rows:
        raise SystemExit(f"No runs found under {args.runs}")
    df = add_work_columns(pd.DataFrame(rows))
    df.to_csv(os.path.join(args.runs, "summary.csv"), index=False)
    print(df.to_string(index=False))

    poisoned = df.groupby("seed")["poisoned_samples"].first().dropna()
    if len(poisoned) and poisoned.max() / poisoned.min() > 1.1:
        print(f"\nWARNING: poisoned-set size is not constant across seeds "
              f"({poisoned.min():.0f}..{poisoned.max():.0f}). The paired contrasts stay "
              f"valid (both arms of a seed share its poison set), but the treatment is "
              f"heterogeneous and must be disclosed. Per seed:\n{poisoned.to_string()}")

    stats = [s for m in METRICS if m in df.columns and (s := paired_stats(df, m))]
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(args.runs, "paired_stats.csv"), index=False)
        print("\n", stats_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Aggregate MSc experiment runs into a summary table and paired statistics.

    python -m analysis.analyze [--runs results/msc/runs]

Per run directory (see the runbook in experiments/protocol.md) it reads:
  recovery_manifest.json      TTR, checkpoint I/O (SISA)
  hardware_telemetry_*.csv    temp/throttle/iowait/SD-I/O traces + phase markers
  power_fnb58.csv + phases.log  FNB58 power samples, integrated per phase window
  sisa_timings.jsonl          Phase-1 SISA overhead (per-slice train + ckpt I/O)
  results_phase{1,4}.json     per-round global F1/recall

Writes summary.csv and paired_stats.csv next to the runs. Per metric: medians +
IQR, Wilcoxon signed-rank p, Cliff's delta, and a bootstrap 95% CI on the median
paired difference — with small N the effect size and CI carry the argument, not p.
Throttling is counted from the *live* flag bits only (the occurred bits are sticky);
energy is reported net of the idle baseline. Thesis figures are added once the
campaign data exists.
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

# The outcomes carried into paired statistics, one per hypothesis:
#   ttr_s              recovery wall time            (H1)
#   p3_energy_net_wh   recovery energy, idle-subtracted (H2)
#   p3_throttled_s     seconds thermally throttled during recovery (H3)
#   p3_min_clock_mhz   worst clock during recovery — degradation   (H3)
#   p3_sd_written_mb   SD bytes written during recovery            (H4)
#   p1_ckpt_bytes      SISA Phase-1 checkpoint overhead            (H6)
#   p4_final_f1/recall recovered-model utility                     (H5)
METRICS = ["ttr_s", "p3_energy_net_wh", "p3_throttled_s", "p3_min_clock_mhz",
           "p3_sd_written_mb", "p1_ckpt_bytes", "p4_final_f1", "p4_final_recall"]


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size: P(a>b) - P(a<b) over all pairs."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def active_throttle(hexflag) -> tuple:
    """(thermal_now, undervoltage_now) from a vcgencmd get_throttled hex string.

    Bits 1/2/3 = arm-freq-capped / throttled / soft-temp-limit *now*; bit 0 =
    under-voltage *now*. The 16–19 "occurred" bits are sticky (stay set for the
    rest of the boot after one event), so counting anything != 0x0 over-reports
    throttling massively — we look only at the live bits.
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

    Energy is the trapezoidal integral of instantaneous power (V*I) over each phase
    window — self-contained, so a mid-run logger restart (which zeroes the device's
    cumulative counter) cannot corrupt it. Net energy subtracts the idle baseline so
    H2 reports the *marginal* cost of recovery, not the whole idling board. Warns if
    the power log doesn't cover a phase window.
    """
    path = os.path.join(run_dir, "power_fnb58.csv")
    windows = _phase_windows(run_dir)
    if not os.path.exists(path) or not windows:
        return
    p = pd.read_csv(path, sep=r"\s+")  # blank line + header row handled by skip_blank_lines
    if "timestamp" not in p.columns or len(p) < 2:
        return
    p["power_w"] = p["voltage_V"] * p["current_A"]
    lo, hi = p["timestamp"].min(), p["timestamp"].max()
    idle_w = _idle_power_w(p, run_dir)
    if idle_w is not None:
        row["idle_power_w"] = round(idle_w, 3)
    for label, start, end in windows:
        n = label[-1]  # phase digit: 1 / 3 / 4
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

    telemetry = sorted(glob.glob(os.path.join(run_dir, "hardware_telemetry_*.csv")))
    if telemetry:
        t = pd.read_csv(telemetry[-1])
        flags = t["Throttled"].map(active_throttle)
        t["_thermal"] = flags.map(lambda x: x[0])
        t["_undervolt"] = flags.map(lambda x: x[1])
        row["peak_temp_c"] = t["Temp_C"].max()
        row["undervolt_s"] = int(t["_undervolt"].sum())  # data-quality guard: must be 0 with the 5A cable
        p3 = t[t["Marker"].astype(str).str.startswith("phase3")]
        if len(p3):
            row["p3_throttled_s"] = int(p3["_thermal"].sum())        # H3: seconds actively throttled
            row["p3_min_clock_mhz"] = int(p3["CPU_Freq_MHz"].min())  # H3: worst clock (degradation)
            row["p3_mean_clock_mhz"] = int(round(p3["CPU_Freq_MHz"].mean()))
            row["p3_iowait_mean_pct"] = round(p3["IOWait_Pct"].mean(), 2)
            row["p3_sd_written_mb"] = round(p3["SD_Write_kBps"].sum() / 1024, 1)
            row["p3_peak_temp_c"] = p3["Temp_C"].max()

    sisa_log = os.path.join(run_dir, "sisa_timings.jsonl")
    if os.path.exists(sisa_log):
        entries = [json.loads(line) for line in open(sisa_log)]
        row["p1_ckpt_io_s"] = round(sum(e["ckpt_s"] for e in entries), 2)
        row["p1_ckpt_bytes"] = sum(e["ckpt_bytes"] for e in entries)

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


def paired_stats(df: pd.DataFrame, metric: str) -> dict | None:
    """Wilcoxon signed-rank + Cliff's delta + bootstrap CI for one metric, paired by seed."""
    wide = df.pivot_table(index="seed", columns="arm", values=metric).dropna()
    if len(wide) < 2 or "naive" not in wide or "sisa" not in wide:
        return None
    naive, sisa = wide["naive"], wide["sisa"]
    try:
        _, p = wilcoxon(naive, sisa)
    except ValueError:  # all differences zero
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
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.runs, "summary.csv"), index=False)
    print(df.to_string(index=False))

    stats = [s for m in METRICS if m in df.columns and (s := paired_stats(df, m))]
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(args.runs, "paired_stats.csv"), index=False)
        print("\n", stats_df.to_string(index=False))


if __name__ == "__main__":
    main()

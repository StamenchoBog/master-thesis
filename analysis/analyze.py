"""Aggregate MSc experiment runs into a summary table and paired statistics.

    python -m analysis.analyze [--runs results/msc/runs]

Per run directory (see the runbook in experiments/protocol.md) it reads:
  recovery_manifest.json      TTR, checkpoint I/O (SISA)
  hardware_telemetry_*.csv    temp/throttle/iowait/SD-I/O traces + phase markers
  sisa_timings.jsonl          Phase-1 SISA overhead (per-slice train + ckpt I/O)
  results_phase{1,4}.json     per-round global F1/recall

Writes summary.csv and paired_stats.csv next to the runs. Wilcoxon signed-rank
paired by seed + Cliff's delta; with N=5 pairs, effect sizes matter more than
p-values — both are reported. FNB58 energy and thesis figures are added after
the pilot, once real export formats exist.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["ttr_s", "throttled_s", "p3_iowait_mean_pct", "p3_sd_written_mb",
           "p4_final_f1", "p4_final_recall"]


def cliffs_delta(a, b) -> float:
    """Cliff's delta effect size: P(a>b) - P(a<b) over all pairs."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


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
        row["peak_temp_c"] = t["Temp_C"].max()
        row["throttled_s"] = int((~t["Throttled"].astype(str).isin(["0x0", "n/a"])).sum())
        p3 = t[t["Marker"].astype(str).str.startswith("phase3")]
        if len(p3):
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
    return row


def paired_stats(df: pd.DataFrame, metric: str) -> dict | None:
    """Wilcoxon signed-rank + Cliff's delta for one metric, paired by seed."""
    wide = df.pivot_table(index="seed", columns="arm", values=metric).dropna()
    if len(wide) < 2 or "naive" not in wide or "sisa" not in wide:
        return None
    naive, sisa = wide["naive"], wide["sisa"]
    try:
        _, p = wilcoxon(naive, sisa)
    except ValueError:  # all differences zero
        p = 1.0
    return {
        "metric": metric, "n_pairs": len(wide),
        "naive_median": round(float(naive.median()), 3),
        "naive_iqr": round(float(naive.quantile(0.75) - naive.quantile(0.25)), 3),
        "sisa_median": round(float(sisa.median()), 3),
        "sisa_iqr": round(float(sisa.quantile(0.75) - sisa.quantile(0.25)), 3),
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

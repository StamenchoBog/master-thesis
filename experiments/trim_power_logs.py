"""Trim each run's power_fnb58.csv to the samples the analysis actually uses.

    python experiments/trim_power_logs.py [--runs results/msc/runs] [--apply]

The FNB58 logger runs continuously, so most of a run's power CSV is idle pre-roll
(before Phase 1) and post-tail (after `done`) — pure noise that bloats the file to
100+ MB. `analysis/analyze.py` only reads samples inside the phase-marker windows
(phase1/idle/phase3/idle/phase4 → done). This keeps exactly that span
[first marker − 1 s, last marker + 1 s] and drops the rest, so the analysis output is
unchanged while the file shrinks dramatically — small enough to version in git.

Dry-run by default (prints before/after sizes); pass --apply to rewrite in place.
The original blank-line + header are preserved.
"""

import argparse
import glob
import os


def marker_bounds(run_dir):
    path = os.path.join(run_dir, "phases.log")
    if not os.path.exists(path):
        return None
    ts = [float(p[0]) for line in open(path) if len(p := line.split()) == 2]
    return (min(ts) - 1.0, max(ts) + 1.0) if ts else None


def trim(run_dir, apply):
    csv = os.path.join(run_dir, "power_fnb58.csv")
    bounds = marker_bounds(run_dir)
    if not os.path.exists(csv) or bounds is None:
        return
    lo, hi = bounds
    lines = open(csv).read().splitlines()
    # Preserve leading blank line + header (the logger prints a blank then a header row).
    head, data = [], []
    for ln in lines:
        parts = ln.split()
        if parts and parts[0].replace(".", "", 1).isdigit():
            data.append(ln)
        else:
            head.append(ln)
    kept = [ln for ln in data if lo <= float(ln.split()[0]) <= hi]
    if not kept:  # markers don't overlap the log (e.g. the rehearsal) — leave it alone
        print(f"  {os.path.basename(run_dir):16s} SKIP (no samples in marker window)")
        return
    before = os.path.getsize(csv)
    out = "\n".join(head + kept) + "\n"
    after = len(out.encode())
    print(f"  {os.path.basename(run_dir):16s} {before/1e6:7.1f} MB -> {after/1e6:6.2f} MB "
          f"({len(kept)}/{len(data)} samples)")
    if apply:
        with open(csv, "w") as f:
            f.write(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="results/msc/runs")
    p.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    args = p.parse_args()
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if os.path.isdir(d):
            trim(d, args.apply)
    if not args.apply:
        print("\n(dry run — pass --apply to rewrite)")


if __name__ == "__main__":
    main()

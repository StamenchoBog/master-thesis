"""Model-size study: component fits, cost model and its check against full runs (A1-A4).

    python -m analysis.evaluate_model_size
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

S, R, ROUNDS = 5, 5, 10
N_CKPT_SISA = 47            # slices replayed by SISA recovery in the campaign design
CKPT_PHASE1 = ROUNDS * S * R  # checkpoints written during Phase 1
FREE_SD_BYTES = 94e9        # free space on the Pi's card at the start of the study
TOLERANCE_A4 = 0.15


def load_measurements(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "model_size_w*_r*.json"))):
        j = json.load(open(p))
        keys = ("width", "rep", "params", "rows_per_s_tail", "rows_per_s_all", "ckpt_bytes",
                "ckpt_write_median_s", "ckpt_read_median_s", "peak_rss_mb_5_constituents")
        blocks = j["blocks"]
        rows.append({k: j[k] for k in keys}
                    | {"t_train": blocks["train"], "t_ckpt": blocks["ckpt"],
                       "t_ram": blocks["ram"], "t_end": blocks["end"]})
    return pd.DataFrame(rows)


def telemetry_per_block(d, df):
    files = sorted(glob.glob(os.path.join(d, "hardware_telemetry_*.csv")))
    if not files:
        return df
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bits = t["Throttled"].astype(str).apply(lambda v: int(v, 16) if v.startswith("0x") else 0)
    t["uv"] = (bits & 0b1).astype(bool)
    t["thr"] = (bits & 0b1110).astype(bool)
    for i, r in df.iterrows():
        tag = f"w{r.width}_r{r.rep}"
        tr = t[t["Marker"].astype(str) == f"bench_train_{tag}"]
        df.loc[i, "train_uv_s"] = int(tr["uv"].sum())
        df.loc[i, "train_throttled_s"] = int(tr["thr"].sum())
        df.loc[i, "train_peak_temp_c"] = float(tr["Temp_C"].max()) if len(tr) else np.nan
    return df


def read_power(path):
    p = pd.read_csv(path, sep=r"\s+", comment="#", skiprows=1, on_bad_lines="skip",
                    names=["timestamp", "sample", "v", "i", "dp", "dn", "temp", "energy",
                           "capacity"])
    p = p[pd.to_numeric(p["timestamp"], errors="coerce").notna()].astype(float)
    p["w"] = p["v"] * p["i"]
    return p


def power_per_block(d, df):
    p_path = os.path.join(d, "power_fnb58.csv")
    if not os.path.exists(p_path):
        return df, np.nan
    p = read_power(p_path)
    # Block timestamps come from the Pi's clock; shift them to the host's.
    offset = 0.0
    for line in open(os.path.join(d, "phases.log")):
        if line.startswith("clock_offset_pi_minus_host"):
            offset = float(line.split()[1])
    idle = []
    for i, r in df.iterrows():
        for name, a, b in (("train", r.t_train, r.t_ckpt), ("ckpt", r.t_ckpt, r.t_ram)):
            seg = p[(p.timestamp >= a - offset) & (p.timestamp < b - offset)]
            cover = coverage(seg.timestamp.values, a - offset, b - offset)
            df.loc[i, f"{name}_power_cover"] = cover
            df.loc[i, f"{name}_mean_w"] = seg.w.mean() if cover >= 0.95 else np.nan
        # The minute before each training block is the end of its cooldown gate (idle).
        pre = p[(p.timestamp >= r.t_train - offset - 60) & (p.timestamp < r.t_train - offset - 5)]
        if len(pre) > 1000:
            idle.append(pre.w.mean())
    # Checkpoint power comes from the checkpoint-only blocks at the end of the session.
    ckpt_only = {}
    for path in glob.glob(os.path.join(d, "model_size_ckptonly_w*.json")):
        j = json.load(open(path))
        a, b = j["blocks"]["ckpt"] - offset, j["blocks"]["end"] - offset
        seg = p[(p.timestamp >= a) & (p.timestamp < b)]
        if coverage(seg.timestamp.values, a, b) >= 0.95:
            ckpt_only[j["width"]] = seg.w.mean()
    df["ckptonly_mean_w"] = df["width"].map(ckpt_only)
    return df, float(np.median(idle)) if idle else np.nan


def coverage(ts, a, b):
    """Share of [a, b) covered by samples: edges and any gap over 1 s inside count as missing."""
    if len(ts) < 2 or b <= a:
        return 0.0
    gaps = np.diff(ts)
    missing = (ts[0] - a) + (b - ts[-1]) + gaps[gaps > 1.0].sum()
    return max(0.0, 1.0 - missing / (b - a))


def fit_power_law(x, y):
    b, log_a = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(log_a), b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/msc/model_size")
    ap.add_argument("--tables", default="results/msc/tables")
    ap.add_argument("--out", default="results/msc/tables")
    args = ap.parse_args()

    df = load_measurements(args.dir)
    if df.empty:
        raise SystemExit(f"No benchmark results in {args.dir}")
    df = telemetry_per_block(args.dir, df)
    df, idle_w = power_per_block(args.dir, df)
    df.round(6).to_csv(os.path.join(args.out, "model_size_measurements.csv"), index=False)

    med = df.groupby("params").median(numeric_only=True).sort_index()
    P = med.index.values.astype(float)
    sec_per_row = 1.0 / med.rows_per_s_tail.values

    # A1 fit, and the checkpoint size and write-time fits the cost model uses.
    a1_a, a1_b = fit_power_law(P, sec_per_row)
    a2_slope, a2_const = np.polyfit(P, med.ckpt_bytes.values, 1)
    bw_inv, latency = np.polyfit(med.ckpt_bytes.values, med.ckpt_write_median_s.values, 1)
    rd_inv, rd_latency = np.polyfit(med.ckpt_bytes.values, med.ckpt_read_median_s.values, 1)

    # Work per arm, from the campaign (medians over the 10 seeds).
    camp = pd.read_csv(os.path.join(args.tables, "campaign_summary.csv"))
    camp = camp[~camp.run.str.startswith("wan_")]
    u_naive = camp[camp.arm == "naive"].recovery_sample_updates.median()
    u_sisa = camp[camp.arm == "sisa"].recovery_sample_updates.median()
    ttr_naive = camp[camp.arm == "naive"].ttr_s.median()
    ttr_sisa = camp[camp.arm == "sisa"].ttr_s.median()

    def predict(thr, t_ckpt, t_load):
        t_naive = u_naive / thr
        t_sisa = u_sisa / thr + N_CKPT_SISA * t_ckpt + t_load
        return t_naive, t_sisa

    pred = []
    for p_, r in med.iterrows():
        t_n, t_s = predict(r.rows_per_s_tail, r.ckpt_write_median_s, r.ckpt_read_median_s)
        row = {"params": int(p_), "width": int(r.width), "thr_rows_per_s": r.rows_per_s_tail,
               "ckpt_mb": r.ckpt_bytes / 1e6, "ckpt_write_s": r.ckpt_write_median_s,
               "t_naive_s": t_n, "t_sisa_s": t_s, "speedup": t_n / t_s,
               "io_share_sisa": N_CKPT_SISA * r.ckpt_write_median_s / t_s,
               "phase1_ckpt_gb": CKPT_PHASE1 * r.ckpt_bytes / 1e9,
               "peak_rss_mb": r.peak_rss_mb_5_constituents}
        if not np.isnan(idle_w) and not np.isnan(r.get("train_mean_w", np.nan)):
            net_train = r.train_mean_w - idle_w
            # Checkpoint power: the checkpoint-only block if covered, else the main checkpoint
            # blocks, else training power (an upper bound, flagged in the table).
            if not np.isnan(r.get("ckptonly_mean_w", np.nan)):
                net_ckpt, row["ckpt_power_source"] = r.ckptonly_mean_w - idle_w, "ckpt-only block"
            elif not np.isnan(r.get("ckpt_mean_w", np.nan)):
                net_ckpt, row["ckpt_power_source"] = r.ckpt_mean_w - idle_w, "main blocks"
            else:
                net_ckpt, row["ckpt_power_source"] = net_train, "training power (upper bound)"
            row["net_train_w"], row["net_ckpt_w"] = net_train, net_ckpt
            row["e_naive_wh"] = net_train * t_n / 3600
            row["e_sisa_wh"] = (net_train * u_sisa / r.rows_per_s_tail
                                + net_ckpt * N_CKPT_SISA * r.ckpt_write_median_s) / 3600
        pred.append(row)
    pred = pd.DataFrame(pred)
    pred.round(4).to_csv(os.path.join(args.out, "model_size_predictions.csv"), index=False)

    # A3: where storage runs out, and where the fitted model would lose the time advantage.
    p_storage = (FREE_SD_BYTES / CKPT_PHASE1 - a2_const) / a2_slope
    grid = np.logspace(4, 10, 2000)
    thr_g = 1.0 / (a1_a * grid ** a1_b)
    bytes_g = a2_slope * grid + a2_const
    t_n_g = u_naive / thr_g
    t_s_g = (u_sisa / thr_g + N_CKPT_SISA * (bw_inv * bytes_g + latency)
             + (rd_inv * bytes_g + rd_latency))
    cross = grid[np.argmax(t_s_g >= t_n_g)] if (t_s_g >= t_n_g).any() else np.nan

    pd.set_option("display.width", 200)
    print(f"Idle baseline (session): {idle_w:.2f} W")
    print("\nPer size (medians over repetitions):")
    print(pred.round(3).to_string(index=False))
    print(f"\nA1  sec/row = {a1_a:.3e} * P^{a1_b:.3f}   "
          "(exponent < 1: throughput falls slower than P)")
    print(f"    ckpt bytes = {a2_slope:.2f} * P + {a2_const:.0f};  "
          f"write s = bytes / {1 / bw_inv / 1e6:.1f} MB/s + {latency * 1e3:.1f} ms")
    print(f"A2  speed-up per size: {', '.join(f'{s:.2f}' for s in pred.speedup)}  "
          f"(all > 1: {bool((pred.speedup > 1).all())}); I/O share: "
          f"{', '.join(f'{x:.1%}' for x in pred.io_share_sisa)}")
    cross_txt = f"P ~ {cross:.2e}" if not np.isnan(cross) else f"none up to P = {grid[-1]:.0e}"
    print(f"A3  storage runs out at P ~ {p_storage:.2e}; fitted time crossover: {cross_txt}"
          f"  (storage first: {bool(np.isnan(cross) or p_storage < cross)})")

    # A4: predicted against measured recovery time.
    checks = []
    base = pred[pred.params == 12865]
    if len(base):
        checks.append(("campaign, P=12,865", base.t_naive_s.iloc[0], ttr_naive,
                       base.t_sisa_s.iloc[0], ttr_sisa))
    for vdir in sorted(glob.glob(os.path.join(args.dir, "validation", "w*_naive_seed*"))):
        w = int(os.path.basename(vdir).split("_")[0][1:])
        sdir = vdir.replace("_naive_", "_sisa_")
        row = pred[pred.width == w]
        if len(row) and os.path.exists(os.path.join(sdir, "recovery_manifest.json")):
            m_n = json.load(open(os.path.join(vdir, "recovery_manifest.json")))["total_s"]
            m_s = json.load(open(os.path.join(sdir, "recovery_manifest.json")))["total_s"]
            checks.append((f"validation, width {w}", row.t_naive_s.iloc[0], m_n,
                           row.t_sisa_s.iloc[0], m_s))
    for name, pn, mn, ps, ms in checks:
        en, es = abs(pn - mn) / mn, abs(ps - ms) / ms
        ok = en <= TOLERANCE_A4 and es <= TOLERANCE_A4
        print(f"A4  {name}: naive predicted {pn:.1f}s vs measured {mn:.1f}s ({en:.0%}), "
              f"SISA {ps:.1f}s vs {ms:.1f}s ({es:.0%}) -> {'holds' if ok else 'fails'}")

    # Energy = power x time and A4 checks the time, so compare Phase 3 power with the benchmark.
    print("\nPhase 3 mean power, full runs vs benchmark training power (gross W):")
    bench_w = df.groupby("width").train_mean_w.median()
    for run in sorted(glob.glob(os.path.join(args.dir, "validation", "w*_*_seed*"))):
        w = int(os.path.basename(run).split("_")[0][1:])
        p_path, ph_path = os.path.join(run, "power_fnb58.csv"), os.path.join(run, "phases.log")
        if not (os.path.exists(p_path) and os.path.exists(ph_path)):
            continue
        marks = [line.split() for line in open(ph_path) if len(line.split()) == 2]
        starts = [float(t) for t, m in marks if m == "phase3"]
        ends = [float(marks[i + 1][0]) for i, (t, m) in enumerate(marks[:-1]) if m == "phase3"]
        if not starts:
            continue
        pw = read_power(p_path)
        seg = pw[(pw.timestamp >= starts[0]) & (pw.timestamp < ends[0])]
        cover = coverage(seg.timestamp.values, starts[0], ends[0])
        mean_w = seg.w.mean() if cover >= 0.95 else float("nan")
        ref = bench_w.get(w, float("nan"))
        print(f"  {os.path.basename(run)}: {mean_w:.3f} W (coverage {cover:.0%}) vs benchmark "
              f"{ref:.3f} W -> {(mean_w / ref - 1):+.1%}")

    flagged = df[df["train_uv_s"] > 0] if "train_uv_s" in df else df.iloc[0:0]
    if len(flagged):
        print("\nLive under-voltage during training (s):",
              ", ".join(f"w{r.width}/r{r.rep}={int(r.train_uv_s)}" for _, r in flagged.iterrows()))


if __name__ == "__main__":
    main()

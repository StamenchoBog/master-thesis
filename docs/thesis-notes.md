# Thesis notes — findings, framing, and defense preparation

Working companion for writing up the MSc (SISA unlearning vs. naive retraining on a
fanless Raspberry Pi 5). Everything here is grounded in measured data or a decision
recorded in `experiments/protocol.md`. Update as the campaign produces more seeds.

---

## 1. Headline result (N=10 complete — seeds 42–51, real measured data)

Both arms achieve the **same guarantee** (exact local unlearning of the compromised
data); the comparison is the *physical cost of reaching that same outcome*. Figures
below are medians over the 10 paired seeds; p is Wilcoxon signed-rank (paired), δ is
Cliff's delta. **p = 0.0020 is the two-sided floor at N=10** — it cannot go lower, so
every headline metric is at the maximum attainable significance.

| Metric | Naive (median) | SISA (median) | Effect | Wilcoxon p | Cliff's δ |
|---|---|---|---|---|---|
| Recovery time (H1) | 242.1 s | 29.5 s | **8.2× faster** | 0.0020 | +1.00 |
| Net recovery energy (H2) | 0.093 Wh | 0.017 Wh | **5.5× less** | 0.0020 | +1.00 |
| Throttled seconds (H3) | 233 s | 27.5 s | **8.5× less** | 0.0020 | +1.00 |
| SD written in recovery (H4) | 2.75 MB | 12.05 MB | I/O shifts to SISA | 0.0020 | −1.00 |
| Post-rejoin global recall (H5) | 0.961 | 1.000 | SISA stable (see §3) | 0.0020 | −1.00 |
| Post-rejoin global F1 | 0.965 | 0.982 | SISA higher | 0.0059 | −0.80 |
| Min clock during recovery (H3) | 2400 MHz | 2400 MHz | not discriminating* | 0.2500 | −0.22 |

*Min-clock is **not** significant across the campaign: naive throttles its clock only
when it heat-soaks (e.g. seeds 50/51 dropped to 1800 MHz at 90 °C), but not every
seed does within the recovery window. The robust thermal signal is **throttled
*duration*** (H3, p=0.0020, δ=+1.00), not the clock floor — report throttled-seconds,
keep the clock trace as a per-seed illustration. See §2.

**Every SISA run beat every naive run on TTR, energy, throttled-seconds, SD-written,
and post-rejoin recall (δ = ±1.00 on all five).** Recovery cost is highly reproducible:
SISA 26–44 s, naive 237–294 s across all 10 seeds. Confidence intervals are bootstrap
95% (see `paired_stats.csv`); TTR diff CI [208.7, 219.6] s excludes zero by a wide
margin. Counterbalanced A/B order (naive-first / sisa-first alternating per seed).

Phase-1 checkpoint overhead (H6, SISA's cost of ownership): ~65.5 MB / 5–24 s of
checkpoint I/O per run, measured alongside the recovery savings — see `p1_ckpt_*`
columns in `summary.csv`.

## 2. The thermal metric (important framing point)

Peak temperature does **not** discriminate the arms — a fanless Pi under any sustained
load saturates at ~87 °C. The real thermal signal is **throttle *duration* and *clock
degradation***: naive sits frequency-capped for 225 s and its clock collapses to
1500 MHz; SISA finishes in 29 s before heat-soaking, accumulates ~30 s of throttle,
and never leaves 2400 MHz. Report throttled-seconds + the clock-frequency trace over
the recovery window; put peak temperature in a footnote as "saturates for both".

## 3. Utility: SISA is *stable*, naive is *variable* (revised after seed 43)

Seed 42 suggested naive degrades recall (post-rejoin 1.0 → 0.927 vs. SISA's 1.0), and
it looked like a clean "naive is worse" story. **Seed 43 overturned that** (naive 0.998).
Across the full N=10 the pattern is unambiguous:

> **SISA's recovery is deterministically stable — post-rejoin recall = 1.000 on all 10
> seeds — whereas naive's post-rejoin utility is seed-dependent and variable, ranging
> 0.879 → 0.998 (median 0.961, IQR 0.056).**

Concretely, naive post-rejoin recall by seed: 42→0.927, 43→0.998, 50→0.879 (worst,
heat-soaked to 90 °C), 51→0.987, 49→0.994 — SISA is 1.000 every time. The paired test
gives p=0.0020, δ=−1.00 (every SISA ≥ every naive). This is a *stronger and more
honest* claim than "naive is worse": SISA gives a predictable, reproducible outcome;
naive's from-scratch retrain lands in different basins run to run. **Report the
variance** (a box/strip plot of the 10 naive points against SISA's flat 1.0 line is the
figure), not a single-seed anomaly — a textbook illustration of why N>1 matters.

The training-budget caveat still stands as a discussion point (naive-from-scratch 10
epochs vs. SISA's incremental constituents are not obviously equal budgets), but it is
no longer needed to explain a decline — because the decline is not consistent.

## 4. Anticipated committee objections (and how to answer)

1. **"N=1 proves nothing."** Only the N=10 campaign fixes this; effect sizes are huge,
   so reversal is unlikely, but it is not *established* until the seeds + Wilcoxon +
   Cliff's delta + bootstrap CIs exist.
2. **"SISA uses less energy because it does less work — circular."** Both arms reach
   the *same guarantee*; this is cost-of-a-fixed-goal, not same-work-different-cost.
   State this up front or the objection lands.
3. **"Is retrain-from-scratch a fair naive baseline?"** Chosen as the exact-unlearning
   gold standard; discuss the cheaper alternatives (fine-tuning, fewer epochs) rejected.
4. **"The advantage is round-dependent."** SISA replays the affected shard for all 10
   rounds (rollback carries poison forward); its edge shrinks with more rounds, grows
   with fewer. Report as a scaling finding, not a flaw.
5. **"One Pi, one SD card — generalisation?"** Disclose exact hardware; frame as a
   single-device case study; note as a limitation.
6. **"Confined poison has no global effect — why recover?"** Reframed as *unlearning
   cost*: removal is mandated by detection/compliance, not measured harm (see the
   Framing section of `protocol.md`). FL robustness to confined poison is itself a
   reported finding.

## 5. Methodology decisions and rationale

- **Unlearning-cost framing** (not attack-recovery): the confined poison (fraction 1.0,
  slices 3–4 of one shard, 1 of 4 nodes) leaves global utility unchanged; the
  contribution is the *physical cost of removing identified-compromised data*.
- **Frozen parameters** (`protocol.md`): 1M rows/node (validated — naive throttles),
  poison fraction 1.0, S=5×R=5, Phase-1 10 rounds / Phase-4 5, naive budget 10 epochs,
  N=10 seeds (42–51), A/B counterbalanced.
- **Rigor**: pre-registration, paired non-parametric stats + effect sizes + bootstrap
  CIs, net (idle-subtracted) energy, empirical unlearning-efficacy probe (honest
  "inconclusive" — the small model leaks no membership signal, so exactness rests on
  construction, verified bit-identically by `tests/smoke_test.py`).
- **Realism**: heterogeneous simulated clients (3.0/2.0/1.5 cores), a WAN network
  factor (toxiproxy, one seed), stability-based cooldown gate at the passive idle
  floor, ambient logging (DS18B20, with the cooldown-plateau as a fallback proxy).

## 6. Engineering/operational lessons (good "systems" content — shows rigor)

- **Power delivery is a hidden confound.** A thin meter→Pi cable dropped ~0.4 V and
  chronically under-volted the Pi (clock throttling unrelated to the workload),
  invisible on the meter's own 5.1 V reading. Fixed with a 5 A cable; verify
  `vcgencmd get_throttled == 0x0` before every run. *(Pilot caught this.)*
- **The 40 °C cooldown gate was physically unreachable** on a fanless Pi (idle floor
  ~60–79 °C, ambient-dependent). Replaced with a *stability plateau* gate.
- **Sticky throttle bits** — the analysis must count only the *live* flag bits; the
  16–19 "occurred" bits stay set for the rest of the boot and over-report throttling.
- **A stale `INIT_FROM_CHECKPOINT` env crashed a Phase 1** — always tear down and bring
  the host up fresh per run.
- **`nohup`/`setsid` telemetry over SSH does not persist** — launch via `systemd-run`.
- **macOS + FNB58 USB wedges** whenever a logger is stopped/restarted — replug or
  power-cycle the meter between runs.
- **The small IDS model makes checkpoint I/O negligible** (H4 tested, not assumed) —
  hence the scaled-model sensitivity study to locate where I/O dominates.
- **Cache-seed must match the runtime seed.** Poison is baked into the Pi's `.npz`
  cache by `prepare_edge_data.py --seed N` (shard 1, slices 3+ under seed N's
  shard/slice assignment), and the SISA client re-derives that assignment from the
  same `SEED=N` at runtime — so a correctly-prepared seed always recovers as
  `{shard 1: slice 3}`. Running seed 50 against a stale seed-49 cache scattered the
  fixed poison rows into *slice 0 of two shards* under seed-50's assignment, leaving
  "no clean checkpoint to roll back to" → a full 250-slice retrain (264 s, ≈ naive).
  Caught because every valid seed prints `{1: 3}`; fixed by regenerating the cache.
  Naive is immune (it flips/drops the same rows regardless of shard placement). *Per-
  seed checklist item: verify `manifest.json` seed == run seed before Phase 1.*
- **SISA's advantage is contingent on poison locality.** The seed-50 mishap is also a
  genuine *finding*: SISA's speedup depends on the compromise being confined to late
  slices of few shards. Poison in slice 0 (or spread across all shards) forces
  near-full retraining and erases the advantage — the realistic "source compromised
  at time τ" threat model (late slices, one shard) is precisely what SISA is built for,
  and worth stating as a scope condition rather than hiding.
- **A stale OrbStack virtiofs mount silently breaks writes.** After a long session of
  many container recreations, the serverapp's `/results` bind mount went dangling:
  `makedirs(exist_ok=True)` succeeded but `open(...,"w")` raised `FileNotFoundError`,
  and container recreation / full `down`+`up` did not clear it (it also crashed
  Finder on mount enumeration). Only restarting OrbStack — ultimately a host reboot —
  fixed it. If unattended host writes start failing mid-campaign, suspect the mount
  layer, not the app.

## 7. What remains

- ~~10 paired seeds (42–51)~~ **DONE** — N=10 complete, all core metrics p=0.0020,
  δ=±1.00 (the two-sided Wilcoxon floor at N=10). Pushed via Git LFS.
- **Supporting runs** (optional, strengthen but not required for the main claim):
  5 clean references (utility ceiling), 1 scaled-model sensitivity study (locate where
  SD I/O dominates — H4 shows it's negligible for the small IDS model), 1 WAN-condition
  run (toxiproxy latency).
- **The naive-budget control** (§3) to settle the training-budget question (naive 10
  epochs vs. SISA incremental) — a discussion point, not needed for the headline.
- **Figures to generate** from `summary.csv` / `paired_stats.csv`:
  1. Recovery-time comparison (paired dumbbell or grouped bar, per-seed points).
  2. Energy bars, idle-subtracted, with the 10 per-seed points overlaid.
  3. Throttle-duration + clock trace over the recovery window (naive heat-soak vs. SISA).
  4. **Post-rejoin recall strip plot**: 10 naive points (0.88–1.0 spread) vs. SISA's flat
     1.0 — the §3 stability figure.
  5. H6 cost-of-ownership: SISA Phase-1 checkpoint overhead vs. its recovery savings.

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
| **Balanced accuracy (H5, correct utility metric)** | **0.598** | **0.500** | **naive better** | — | — |
| Min clock during recovery (H3) | 2400 MHz | 2400 MHz | not discriminating* | 0.2500 | −0.22 |

**H5 utility — read §3 before quoting anything here.** Post-rejoin *recall* is SISA
1.000 vs naive 0.961 and *F1* SISA 0.982 vs naive 0.965 — but on the 96.5 %-attack
eval set those metrics are near-trivial and **reverse the truth**: SISA's model predicts
everything as attack (specificity 0, balanced accuracy 0.500 = chance), while naive
actually discriminates (balanced accuracy 0.598). The correct metrics are balanced
accuracy / specificity / MCC (exact re-evaluation in §3 and §8). Do not headline
recall or F1.

*Min-clock is **not** significant across the campaign: naive throttles its clock only
when it heat-soaks (e.g. seeds 50/51 dropped to 1800 MHz at 90 °C), but not every
seed does within the recovery window. The robust thermal signal is **throttled
*duration*** (H3, p=0.0020, δ=+1.00), not the clock floor — report throttled-seconds,
keep the clock trace as a per-seed illustration. See §2.

**Every SISA run beat every naive run on the four physical-cost metrics — TTR, energy,
throttled-seconds, SD-written (δ = ±1.00 on all four).** That is the robust, headline
contribution. (Utility/H5 goes the other way once measured honestly — see §3.) Recovery
cost is highly reproducible: SISA 26–44 s, naive 237–294 s across all 10 seeds.
Confidence intervals are bootstrap 95% (see `paired_stats.csv`); TTR diff CI
[208.7, 219.6] s excludes zero by a wide margin. Counterbalanced A/B order (naive-first
/ sisa-first alternating per seed).

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

## 3. Utility: recall/F1 are misleading here — SISA collapses to the majority class

**This section was rewritten after an exact re-evaluation (see §8) overturned the
earlier "SISA has stable perfect recall" reading — that reading was an artifact of
class imbalance, and reporting it would have been indefensible.**

The Phase-4 evaluation set is **96.5 % attack / 3.5 % benign**. On such a set, recall
and F1 are near-trivial: a model that predicts *everything is attack* scores recall
1.000, precision 0.965 (= base rate), F1 0.982 — with **zero** benign detection. That
is *exactly* what SISA produces. Re-scoring every saved global model on the 100 k
held-out global test set (`analysis/reeval_utility.py`, exact confusion matrix):

| Metric (Phase-4 median) | Naive | SISA | Majority-class baseline |
|---|---|---|---|
| Recall (sensitivity) | 0.96 | 1.000 | 1.000 |
| **Specificity** (benign caught) | **0.237** | **0.000** | 0.000 |
| **Balanced accuracy** | **0.598** | **0.500** | 0.500 |
| **MCC** | **0.177** | **0.017** | 0.000 |
| Predicted-positive rate | 0.87–0.99 | **1.0000** | 1.000 |
| ROC-AUC (threshold-free) | 0.725 | 0.751 | 0.500 |

> **SISA's recovered global model classifies essentially *every* flow as an attack
> (predicted-positive rate = 1.0000; tn = 0–2 of ~3 494 benign). Its balanced accuracy
> is 0.500 — chance — and MCC ≈ 0. Naive retraining keeps real benign discrimination
> (balanced accuracy 0.598, MCC 0.177). By the metric that is honest under 96.5 %
> imbalance, naive is *better*, and SISA's "perfect recall" is the trivial majority
> vote.**

Two things keep this from being a fatal indictment of SISA, and both must be stated:

1. **The collapse is a *calibration* failure, not lost information.** Threshold-free,
   SISA's ROC-AUC (0.751) is *≥* naive's (0.725): its raw scores rank attacks above
   benign just as well — the sigmoid outputs are simply all pushed above 0.5, so at the
   deployed threshold everything reads "attack." Threshold tuning / calibration is
   plausible future work that could recover benign detection without changing the cost
   story.
2. **It predates recovery.** SISA's Phase-1 balanced accuracy is already 0.500 (§8
   table) — so the cause is the **SISA parameter-averaging aggregation** (constituent
   *weights* are averaged to feed FedAvg, the documented deviation from vanilla SISA's
   prediction ensembling), not the unlearning step. Averaging independently-trained
   MLP weights on imbalanced data yields a majority-defaulting model.

**Framing for the thesis.** Do **not** headline recall/F1. Report **balanced accuracy,
specificity, MCC, and ROC-AUC**. The honest utility finding is a *tradeoff*, which is
more interesting than a free lunch:

> SISA buys its ~8× cheaper recovery at a real utility cost — under parameter-averaging
> aggregation its global model, at the standard 0.5 threshold, fails to discriminate the
> minority (benign) class (balanced accuracy = chance), a failure recall/F1 hide on
> imbalanced IDS data. The discriminative signal survives (ROC-AUC comparable), so the
> defect is calibration, and recovering it is future work.

This supersedes the earlier "naive is variable / SISA is stable" story, which measured
the wrong thing. (Naive's recall *does* vary 0.877–0.998 seed-to-seed, but that
variation is now a minor sub-point, not the headline.)

## 4. Anticipated committee objections (and how to answer)

0. **"Your utility metric is meaningless on a 96.5 %-imbalanced set — recall 1.0 is the
   majority vote."** *Correct, and we address it head-on (§3).* We re-scored every saved
   global model on a held-out test set with the full confusion matrix and report
   **balanced accuracy, specificity, MCC, and ROC-AUC**, not recall/F1. Doing so reveals
   that SISA's model collapses to all-attack (balanced accuracy = chance) while naive
   discriminates — we report this *against our own arm*. Pre-empt this objection by
   leading with balanced metrics; if you quote recall/F1 unqualified, the objection is
   fatal.
1. **"N=1 proves nothing."** Only the N=10 campaign fixes this; effect sizes are huge,
   so reversal is unlikely, but it is not *established* until the seeds + Wilcoxon +
   Cliff's delta + bootstrap CIs exist.
2. **"SISA uses less energy because it does less work — circular."** Both arms reach
   the *same unlearning guarantee*; this is cost-of-a-fixed-goal, not
   same-work-different-cost. (Note the goal is *removal*, not model quality — and §3
   shows the resulting SISA model is worse-calibrated, so the cost saving is not a free
   lunch.) State this up front or the objection lands.
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
  invisible on the meter's own 5.1 V reading. Mitigated with a 5 A cable + EEPROM
  `PSU_MAX_CURRENT=5000`; check `vcgencmd get_throttled` before every run. *(Pilot
  caught this.)* Note the 5 A cable **reduced but did not eliminate** transient
  under-voltage under peak load through the inline meter — see the residual-undervoltage
  disclosure in §8. It is not a clean 0x0; report it honestly and show it doesn't move
  the result.
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

## 8. Data validation & threats to validity (audit of the N=10 dataset)

A full audit of the 20 committed runs (2026-08-13). No result needed redoing; the
effect is robust to every confound found. Disclose these proactively — they are the
questions a committee will ask, and each has an evidence-based answer.

- **Completeness.** 20 runs = 10 seeds × 2 arms, zero missing values in the six core
  metrics (ttr, throttled, energy, SD, recall, F1). Counterbalancing verified: 5
  naive-first (42,44,46,47,49), 5 sisa-first (43,45,48,50,51).

- **Cross-session pairs (the main threat).** 4 of 10 pairs had their two arms run in
  different sessions, up to ~6 days apart (seed 44: 140 h; 47: 66 h; 43: 48 h; 45: 20 h);
  the other 6 pairs were same-session (< 3 h). Ambient therefore varied *within* those 4
  pairs. **Robustness check settles it:** split the campaign and the effect is identical —
  same-session (n=6): naive 251.6 s vs SISA 31.3 s, 7.9×, δ=±1.00, no overlap, p=0.031;
  cross-session (n=4): naive 240.5 s vs SISA 27.3 s, 8.8×, δ=±1.00, no overlap, p=0.125
  (the N=4 floor). The **cooldown gate** (each run starts from a stabilised idle plateau)
  is the actual ambient control; same-session pairing is a bonus where it exists. The
  ~210 s TTR gap dwarfs any tens-of-seconds ambient thermal swing, so no reordering can
  flip it. *Recommendation: report both subsets; optionally re-run seed 44 same-session
  to remove the single 6-day pair, but the data already proves it doesn't matter.*

- **Residual under-voltage (correcting the "must be 0" claim).** The 5 A cable did not
  fully eliminate transient under-voltage through the inline meter. Non-zero live
  under-voltage bits appear in ~half the runs, but **concentrated in Phase 1/Phase 4**
  (training/rejoin, high sustained load); only 1–9 s ever fall in the Phase-3 recovery
  window (max naive-49 = 9 s / 237 s ≈ 4%). Crucially it does **not** drive the H3
  clock-degradation metric: naive-42 and naive-45 dropped to 1500 MHz with *zero*
  under-voltage — the clock drops are **thermal**, not power-delivery. Under-voltage is
  bit 0; the throttle metric counts bits 1–3 (thermal), so it isn't inflated either.
  Disclose as a minor, quantified limitation.

- **`throttled_s` definition.** Monitor samples at 1 Hz; `throttled_s` = throttled
  samples within the **phase-3 marker window**, which brackets the recovery *invocation*
  (container spin-up + compute + teardown) and is a few seconds longer than the pure
  compute `ttr_s`. Hence sisa-50/51 show 47–49 throttled-s against a 43 s TTR — plus
  those two ran in a warm room (idle plateau ~84–89 °C), so nearly the whole recovery was
  at the soft-temp limit. State the metric precisely as "seconds throttled over the
  recovery window" and note absolute thermal values are ambient-dependent (the paired
  *difference* is not).

- **Wall-time is not bit-deterministic — computation is.** Re-runs of the deterministic
  recovery gave identical training loss (e.g. naive-51 loss 0.014226 both times) but
  slightly different wall times (263.4 → 261.5 s), because TTR is a physical measurement
  sensitive to instantaneous thermal/clock state. "Determinism" in the protocol means
  reproducible *computation* (seeds fix model + data + poison), not a reproducible
  stopwatch. Expected and correct.

- **H6 corrected during this audit.** `analyze.py` was summing the append-only
  `sisa_timings.jsonl` wholesale, conflating Phase-1 (rounds 1–10), Phase-4 rejoin
  (rounds 11–15), and duplicate entries from Phase-1 re-runs → inflated, inconsistent
  checkpoint overhead (65–175 MB). Fixed to dedup Phase-1 slices only: now a consistent
  **250 checkpoints / 43.67 MB / 3.1–5.9 s** across all 10 seeds. Core paired stats
  unaffected (H6 is SISA-only, descriptive).

- **`naive_seed45` Phase-1 power gap** (benign). Its phase-1 markers predate the power
  log by ~20 h (a cross-day run); `analyze.py` warns and skips that window. Phase-1
  energy is descriptive (not in the paired stats) and Phase-3 coverage is intact, so no
  metric is affected.

- **Utility metric was misleading — corrected by exact re-evaluation (the biggest audit
  finding).** The reported recall/F1 come from per-client eval on a **96.5 %-attack**
  set, where they are near-trivial. Re-scoring every saved Phase-1 and Phase-4 global
  model on the 100 k held-out global test set with the full confusion matrix
  (`analysis/reeval_utility.py` → `utility_reeval.csv`) shows SISA predicts **everything
  as attack** (predicted-positive rate = 1.0000, tn = 0–2 of ~3 494 benign, specificity
  ≈ 0, balanced accuracy 0.500 = chance, MCC ≈ 0) in **both** phases, whereas naive
  discriminates (balanced accuracy 0.598, MCC 0.177). Threshold-free ROC-AUC is
  comparable (SISA 0.751 ≥ naive 0.725), so it is a **calibration** failure of the SISA
  parameter-averaging aggregation, present from Phase 1 (not caused by recovery). H5 is
  rewritten accordingly in §3; the physical-cost hypotheses (H1–H4) are untouched. My
  earlier derived-specificity estimate is confirmed exactly by this direct computation.

**Bottom line:** the four **physical-cost** metrics separate the arms with **no
overlap** (naive TTR 230–294 s vs SISA 26–44 s; energy 0.08–0.12 vs 0.012–0.019 Wh;
throttle 204–293 vs 2–49 s; SD 2.7–3.8 vs 11.4–12.7 MB), δ=±1.00, p at the N=10 floor —
this is the solid, headline contribution and survives every robustness check above. The
**utility** result runs the *other* way and must be reported with balanced metrics (§3):
SISA is cheaper to recover but its aggregation sacrifices benign-class discrimination.
That tradeoff — not "SISA wins on everything" — is the honest, defensible thesis.

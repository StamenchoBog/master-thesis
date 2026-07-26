# Thesis notes — findings, framing, and defense preparation

Working companion for writing up the MSc (SISA unlearning vs. naive retraining on a
fanless Raspberry Pi 5). Everything here is grounded in measured data or a decision
recorded in `experiments/protocol.md`. Update as the campaign produces more seeds.

---

## 1. Headline result (seed 42, first complete paired run — real measured data)

Both arms achieve the **same guarantee** (exact local unlearning of the compromised
data); the comparison is the *physical cost of reaching that same outcome*.

| Metric | Naive (retrain-from-scratch) | SISA (rollback + replay) | Ratio |
|---|---|---|---|
| Recovery time (H1) | 230 s | 29 s | **8× faster** |
| Net recovery energy (H2) | 0.121 Wh | 0.017 Wh | **7× less** |
| Throttled seconds (H3) | 225 s | 30 s | **7.5× less** |
| Min clock during recovery (H3) | 1500 MHz | 2400 MHz | naive degrades −37 % |
| Peak temperature | 87.3 °C | 87.8 °C | ≈ equal (see §2) |
| SD written in recovery (H4) | 3.8 MB | 12.7 MB | I/O shifts to SISA |
| Phase-1 checkpoint overhead (H6) | — | 65.5 MB / 7.3 s | SISA's cost of ownership |
| Recovered-model recall (H5) | 0.970 | 1.000 | SISA preserves recall |
| Post-rejoin global recall | 0.927 | 1.000 | SISA more stable (see §3) |

**N = 1 so far** — direction and magnitude are established, statistics require the
full N=10 campaign.

## 2. The thermal metric (important framing point)

Peak temperature does **not** discriminate the arms — a fanless Pi under any sustained
load saturates at ~87 °C. The real thermal signal is **throttle *duration* and *clock
degradation***: naive sits frequency-capped for 225 s and its clock collapses to
1500 MHz; SISA finishes in 29 s before heat-soaking, accumulates ~30 s of throttle,
and never leaves 2400 MHz. Report throttled-seconds + the clock-frequency trace over
the recovery window; put peak temperature in a footnote as "saturates for both".

## 3. The naive utility decline (genuine finding + a caveat to pre-empt)

Naive's post-rejoin global recall drifts 1.0 → 0.927 while SISA holds 1.0. This is
**real and explainable**: the naive recovered model lands *precision-biased*
(recall 0.970 / precision 0.971), missing ~3 % of attacks, and that bias propagates
through the federated rejoin. Since the thesis prioritises **recall > precision** for
an IDS, naive is not only more expensive but produces a **worse model for the use
case** — a stronger story than cost alone.

**Caveat a committee will raise:** the effective training budgets of naive-from-scratch
(10 epochs on retained data) vs. SISA constituents (incremental over 10 rounds,
partially rolled back) are not obviously equal, so the recall gap *could* be
under-training rather than a fundamental property. Address it explicitly:
- (a) confirm the pattern across seeds, and
- (b) a one-seed control — naive recovery with a larger epoch budget; if recall still
  lags → fundamental, if it catches up → frame as *"at equal wall-clock budget, naive
  sacrifices recall."* Either way is a valid claim if worded carefully.

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

## 7. What remains

- **8 more paired seeds (43–51)** → statistics.
- **Supporting runs**: 5 clean references (utility ceiling), 1 scaled-model sensitivity
  study, 1 WAN-condition run.
- **The naive-budget control** (§3) to settle the training-budget question.
- Confirm the naive utility decline holds across seeds.
- Figures once campaign data exists (throttle traces, energy bars with per-seed points,
  recovery-time comparison, F1/recall curves).

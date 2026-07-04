# Experiment Protocol — Naive Retraining vs. SISA Unlearning on a Physical Edge Node

**Status: DRAFT — parameters marked _[pilot]_ are frozen only after the pilot study.
No measured run counts toward the results until this document is frozen.**

## Research question

What are the physical resource costs and model-utility consequences of SISA-based
machine unlearning versus naive retraining on a resource-constrained FL edge node
recovering from data poisoning?

## Pre-registered hypotheses

- **H1** SISA reduces time-to-recovery (TTR) vs. naive local retraining.
- **H2** SISA reduces recovery energy (Wh); naive shows higher sustained power draw.
- **H3** Naive retraining triggers thermal throttling (≥85°C) on the fanless Pi 5; SISA reduces or avoids throttled time.
- **H4** SISA shifts cost toward storage I/O (checkpoint writes, iowait). *Tested, not assumed* — a negligible-I/O result at this model scale is reported honestly; the sensitivity study locates the model size where I/O dominates.
- **H5** Recovered-model utility (F1/recall on the clean test set) is equivalent between arms within ΔF1 ≤ 2 pp; attack success returns to clean-baseline levels in both arms.
- **H6** SISA imposes measurable Phase-1 overhead (per-round wall time, bytes written). Total cost of ownership = Phase-1 overhead + recovery cost.

## Design

| Element | Value |
|---|---|
| Arms | A: naive local retrain-from-scratch · B: SISA rollback + partial replay |
| SISA config | S=5 shards × R=5 slices, seeded permutation assignment |
| Poison | Label flip attack→benign, fraction _[pilot: default 0.5]_ of attack samples in slices 3–4 of shard 1 |
| Data per node | Stratified subsample _[pilot: default 1,000,000]_ rows; equal across all 3 nodes (FedAvg weighting) |
| Clean test set | 100,000 rows, stratified, disjoint from all training subsamples, evaluated host-side |
| Rounds | Phase 1: 10 · Phase 4 rejoin: 5 |
| Recovery budget (naive) | 10 epochs on retained data (= Phase-1 local budget), fresh Adam per epoch |
| Seeds | N = 5 paired (42–46); extend to 10 if time allows |
| Order | A/B counterbalanced across seeds (ABBA BAAB ...) |
| Cooldown gate | SoC ≤ 40°C stable 120 s before every measured phase sequence |
| Statistics | Paired Wilcoxon signed-rank on TTR, energy, throttled time, bytes written; medians + IQR + Cliff's delta; utility mean ± 95% CI |

## Guarantee scope (thesis must state this)

Both arms give **exact local** unlearning (recovered client state provably free of
poison influence) and **approximate global** forgetting: stateless FedAvg cannot
remove already-aggregated poison from the global model client-side. Phase 4 resumes
from the poisoned global model identically in both arms and measures attack-success
decay over the rejoin rounds. Detection is out of scope (oracle poison mask).

Documented deviations from vanilla SISA/FL: constituents never absorb global
weights (required for the rollback guarantee); the FL update is the parameter
average of constituents rather than a prediction ensemble.

## Per-run procedure

Manual, documented in `experiments/RUNBOOK.md`: prepare (seed-specific data +
poison mask, shipped to Pi) → cooldown gate → telemetry + FNB58 start → Phase 1
(10 poisoned FL rounds) → Phase 3 recovery (primary window) → Phase 4 rejoin
(resume from poisoned global checkpoint; Pi on retained data) → FNB58
stop/export → artifact collection. Record ambient temperature for every run.

Reference runs: one clean run per seed (`POISON_MODE=off`, standard client) for the
gold-standard utility ceiling.

Sensitivity study (one seed): repeat the paired comparison with a scaled-up model
to locate where checkpoint I/O becomes the dominant SISA cost.

## Pilot checklist (before freezing)

- [ ] Epoch wall-time on Pi at 0.5M / 1M / 2M rows → fix subsample size
- [ ] Poison fraction achieving ≥10 pp global recall degradation within 10 rounds
- [ ] Naive retraining actually reaches 85°C/throttling (else lengthen budget or data)
- [ ] Checkpoint write footprint per slice (bytes, latency)
- [ ] Determinism: two identical-seed runs → identical model hashes
- [ ] Analysis scripts produce all tables/figures from pilot data

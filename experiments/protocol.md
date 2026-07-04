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

Manual, per the Runbook below. Reference runs: one clean run per seed
(`POISON_MODE=off`, standard client) for the gold-standard utility ceiling.
Sensitivity study (one seed): repeat the paired comparison with a scaled-up model
to locate where checkpoint I/O becomes the dominant SISA cost.

## Pilot checklist (before freezing)

- [ ] Epoch wall-time on Pi at 0.5M / 1M / 2M rows → fix subsample size
- [ ] Poison fraction achieving ≥10 pp global recall degradation within 10 rounds
- [ ] Naive retraining actually reaches 85°C/throttling (else lengthen budget or data)
- [ ] Checkpoint write footprint per slice (bytes, latency)
- [ ] Determinism: two identical-seed runs → identical model hashes
- [ ] Analysis scripts produce all tables/figures from pilot data

---

## Runbook — one experiment run (one arm, one seed)

Manual, copy-paste procedure. You are physically present anyway (power meter,
ambient temp), so no orchestration script — just do the steps in order and
don't skip the cooldown gate.

**Placeholders:** `ARM` = `naive` | `sisa` · `SEED` = 42–46 · `HOST_IP` = this
machine's LAN address. Pi is reachable as `admin@rasp5node.local` with key
`~/.ssh/rasp5node` (ssh-add it first if passphrase-protected), repo cloned at
`~/master-thesis` on the Pi.

### One-time setup

```sh
docker compose run --rm preprocessor          # full partition caches (host)
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml build"
pip install pandas scipy torch                # host venv, for analysis/
```

### 1. Prepare (host)

```sh
python3 experiments/prepare_edge_data.py --seed SEED
scp data/.cache/msc/partition_2_of_3.npz data/.cache/msc/manifest.json \
    admin@rasp5node.local:master-thesis/data/.cache/msc/
ssh admin@rasp5node.local "rm -rf msc-experiment/checkpoints/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json /dev/shm/hardware_telemetry_*.csv"
rm -rf results/msc/global_checkpoints
mkdir -p results/msc/runs/ARM_seedSEED
```

### 2. Cooldown gate + instruments

```sh
experiments/cooldown_gate.sh                  # blocks until SoC <= 40°C for 2 min
ssh admin@rasp5node.local "nohup ./msc-experiment/monitor.sh >/dev/null 2>&1 &"
```

Record ambient temperature. Reset and start FNB58 logging.

### 3. Phase 1 — poisoned federated training (10 rounds)

```sh
ssh admin@rasp5node.local "echo phase1 > /dev/shm/run_marker"
SEED=SEED NUM_ROUNDS=10 RESULTS_SUFFIX=_p1 docker compose -f docker-compose.host.yml up -d
# On the Pi (CLIENT_MODE=standard for naive, sisa for sisa):
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=flip docker compose -f docker-compose.edge.yml up -d"
# Trigger the run (blocks until all 10 rounds finish):
docker compose -f docker-compose.host.yml run --rm runner
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
mv results/msc/global_checkpoints results/msc/runs/ARM_seedSEED/phase1_checkpoints
```

### 4. Phase 3 — recovery on the Pi (primary measurement window)

```sh
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml stop superexec-clientapp-3"
ssh admin@rasp5node.local "echo phase3 > /dev/shm/run_marker"
# naive -> edge_nodes.naive_retrain ; sisa -> edge_nodes.sisa_recover
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml run --rm -v \$PWD:/app -w /app --entrypoint python superexec-clientapp-3 -m edge_nodes.RECOVERY_MODULE"
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
```

### 5. Phase 4 — rejoin (5 rounds, resumed from the poisoned global model)

```sh
ssh admin@rasp5node.local "echo phase4 > /dev/shm/run_marker"
cp results/msc/runs/ARM_seedSEED/phase1_checkpoints/round_10.npz results/msc/resume_from.npz
SEED=SEED NUM_ROUNDS=5 RESULTS_SUFFIX=_p4 INIT_FROM_CHECKPOINT=/results/resume_from.npz \
    docker compose -f docker-compose.host.yml up -d superexec-serverapp
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=drop docker compose -f docker-compose.edge.yml up -d superexec-clientapp-3"
docker compose -f docker-compose.host.yml run --rm runner
ssh admin@rasp5node.local "echo done > /dev/shm/run_marker"
```

Stop FNB58 logging; export its CSV.

### 6. Teardown + collect

```sh
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml down; pkill -f monitor.sh"
docker compose -f docker-compose.host.yml down

RUN=results/msc/runs/ARM_seedSEED
scp "admin@rasp5node.local:/dev/shm/hardware_telemetry_*.csv" \
    admin@rasp5node.local:/dev/shm/recovery_manifest.json \
    admin@rasp5node.local:msc-experiment/checkpoints/recovered_model.pt "$RUN/"
scp admin@rasp5node.local:/dev/shm/sisa_timings.jsonl "$RUN/" 2>/dev/null   # sisa arm only
mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
```

Copy the FNB58 export into `$RUN/power_fnb58.csv` and note the ambient
temperature in a `$RUN/notes.txt`.

### 7. Evaluate + analyze (after runs exist)

```sh
python3 -m analysis.evaluate_model "$RUN/recovered_model.pt"
python3 -m analysis.evaluate_model "$RUN/phase4_checkpoints/round_5.npz"
python3 -m analysis.analyze
```

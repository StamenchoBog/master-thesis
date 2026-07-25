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
| Topology | 4 clients: simulated nodes 1–3 on the host, the Raspberry Pi as the 4th node over the LAN |
| Data per node | Stratified subsample _[pilot: default 1,000,000]_ rows; equal across all 4 nodes (FedAvg weighting) |
| Clean test set | 100,000 rows, stratified, disjoint from all training subsamples, evaluated host-side |
| Rounds | Phase 1: 10 · Phase 4 rejoin: 5 |
| Recovery budget (naive) | 10 epochs on retained data (= Phase-1 local budget), fresh Adam per epoch |
| Seeds | N = 5 paired (42–46); extend to 10 if time allows |
| Order | A/B counterbalanced across seeds (ABBA BAAB ...) |
| Cooldown gate | SoC ≤ 40°C stable 120 s before every measured phase sequence |
| Power | FNB58 inline on the Pi's supply for ALL runs (constant overhead); USB-logged host-side at ~100 sps; energy = trapezoidal ∫W dt per phase window; `PSU_MAX_CURRENT=5000` pinned in EEPROM |
| Statistics | Paired Wilcoxon signed-rank on TTR, energy, throttled time, bytes written; medians + IQR + Cliff's delta; utility mean ± 95% CI |

![MSc hybrid topology](../docs/diagrams/msc-thesis-diagrams-msc-thesis.drawio.png)

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

### Power measurement (FNB58)

Wiring — the meter sits inline on the Pi's power path for **every** run (both
arms and the clean references), so its overhead is a constant, not a confound:

```
Wall PSU (official 27 W) ──USB-C──> FNB58 IN ──USB-C──> Pi 5 power port
FNB58 data port ──USB cable──────> host machine (runs fnirsi_logger.py)
Pi 5 ──Ethernet──> switch ────────> host machine
```

Data is pulled over the FNB58's USB data port, not Bluetooth: the open-source
logger prints ~100 samples/s with **host-clock** UNIX timestamps — the same
clock that issues the phase markers below, so power↔phase alignment needs no
clock-skew correction. The device does not stream Wh; energy is integrated
host-side per phase window. Both machines run NTP (Pi telemetry timestamps are
only cross-referenced coarsely).

The FNB58 sits in the USB-C CC line, which can silently break the Pi 5's 5 V/5 A
PD negotiation (falling back to 3 A). The workload stays well under 15 W, but the
negotiation outcome must not vary across runs — pin it once in the Pi's EEPROM.

### One-time setup

```sh
# Host: 4-way partition caches + analysis deps + FNB58 logger deps.
# The logger lives in the repo (experiments/fnirsi_logger.py) — vendored from
# baryluk/fnirsi-usb-power-data-logger @746e4d3 with macOS-robustness patches.
NUM_PARTITIONS=4 docker compose run --rm preprocessor
pip install pandas scipy torch pyusb crc
brew install libusb
sudo python3 experiments/fnirsi_logger.py | head -3   # sanity check (meter plugged in)

# Pi: build the client image + pin PD behavior (reboot afterwards)
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml build"
ssh admin@rasp5node.local "sudo rpi-eeprom-config --edit"   # add: PSU_MAX_CURRENT=5000
ssh admin@rasp5node.local "sudo reboot"
ssh admin@rasp5node.local "vcgencmd get_config usb_max_current_enable"   # expect =1
```

> **Keep the Mac awake for the whole run.** If macOS sleeps, the FNB58's USB
> endpoints wedge and the logger must be restarted (and the Pi may drop from the
> federation). Run the measured sequence under `caffeinate -dimsu` or disable
> sleep in Settings.

### 1. Prepare (host)

```sh
RUN=results/msc/runs/ARM_seedSEED   # used by every later step — keep the same shell
python3 experiments/prepare_edge_data.py --seed SEED
scp data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json \
    admin@rasp5node.local:master-thesis/data/.cache/msc/
ssh admin@rasp5node.local "rm -rf msc-experiment/checkpoints/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json /dev/shm/hardware_telemetry_*.csv"
rm -rf results/msc/global_checkpoints
mkdir -p "$RUN"
```

### 2. Cooldown gate + instruments

```sh
experiments/cooldown_gate.sh                  # blocks until SoC <= 40°C for 2 min
# Telemetry as a transient systemd unit — survives SSH disconnect (a plain
# nohup/setsid over SSH does not reliably persist).
ssh admin@rasp5node.local "sudo systemctl reset-failed msc-monitor 2>/dev/null; \
  sudo systemd-run --unit=msc-monitor --working-directory=/home/admin \
  /home/admin/msc-experiment/monitor.sh"
sudo python3 -u experiments/fnirsi_logger.py > "$RUN/power_fnb58.csv" &
LOGGER_PID=$!
```

The logger runs silently once streaming (a 5 s heartbeat prints to stderr); do
not Ctrl-C it until teardown. Record the ambient temperature in `$RUN/notes.txt`.
Stop telemetry at teardown with `ssh admin@rasp5node.local "sudo systemctl stop msc-monitor"`.

### 3. Phase 1 — poisoned federated training (10 rounds)

```sh
echo "$(date +%s) phase1" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo phase1 > /dev/shm/run_marker"
SEED=SEED NUM_ROUNDS=10 RESULTS_SUFFIX=_p1 docker compose -f docker-compose.host.yml up -d
# On the Pi (CLIENT_MODE=standard for naive, sisa for sisa):
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=flip docker compose -f docker-compose.edge.yml up -d"
# Trigger the run (blocks until all 10 rounds finish):
docker compose -f docker-compose.host.yml run --rm runner
echo "$(date +%s) idle" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
mv results/msc/global_checkpoints "$RUN/phase1_checkpoints"
```

### 4. Phase 3 — recovery on the Pi (primary measurement window)

```sh
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml stop superexec-clientapp-4"
echo "$(date +%s) phase3" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo phase3 > /dev/shm/run_marker"
# naive -> edge_nodes.naive_retrain ; sisa -> edge_nodes.sisa_recover
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml run --rm -v \$PWD:/app -w /app --entrypoint python superexec-clientapp-4 -m edge_nodes.RECOVERY_MODULE"
echo "$(date +%s) idle" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
```

### 5. Phase 4 — rejoin (5 rounds, resumed from the poisoned global model)

```sh
echo "$(date +%s) phase4" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo phase4 > /dev/shm/run_marker"
cp "$RUN/phase1_checkpoints/round_10.npz" results/msc/resume_from.npz
SEED=SEED NUM_ROUNDS=5 RESULTS_SUFFIX=_p4 INIT_FROM_CHECKPOINT=/results/resume_from.npz \
    docker compose -f docker-compose.host.yml up -d superexec-serverapp
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=HOST_IP SEED=SEED CLIENT_MODE=ARM_CLIENT POISON_MODE=drop docker compose -f docker-compose.edge.yml up -d superexec-clientapp-4"
docker compose -f docker-compose.host.yml run --rm runner
echo "$(date +%s) done" >> "$RUN/phases.log"
ssh admin@rasp5node.local "echo done > /dev/shm/run_marker"
```

### 6. Teardown + collect

```sh
sudo kill $LOGGER_PID                         # stop the FNB58 power log
ssh admin@rasp5node.local "cd master-thesis && docker compose -f docker-compose.edge.yml down; sudo systemctl stop msc-monitor"
docker compose -f docker-compose.host.yml down

scp "admin@rasp5node.local:/dev/shm/hardware_telemetry_*.csv" \
    admin@rasp5node.local:/dev/shm/recovery_manifest.json \
    admin@rasp5node.local:msc-experiment/checkpoints/recovered_model.pt "$RUN/"
scp admin@rasp5node.local:/dev/shm/sisa_timings.jsonl "$RUN/" 2>/dev/null   # sisa arm only
mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
```

The run directory now holds: `power_fnb58.csv`, `phases.log`, telemetry CSV,
recovery manifest, recovered model, both results JSONs, both checkpoint sets,
and `notes.txt` (ambient temperature).

### 7. Evaluate + analyze (after runs exist)

```sh
python3 -m analysis.evaluate_model "$RUN/recovered_model.pt"
python3 -m analysis.evaluate_model "$RUN/phase4_checkpoints/round_5.npz"
python3 -m analysis.analyze
```

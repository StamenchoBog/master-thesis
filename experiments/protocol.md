# Experiment Protocol — Naive Retraining vs. SISA Unlearning on a Physical Edge Node

**Status: FROZEN (pilot complete, 2026-07-25). Parameters below are fixed; this
document is the pre-registration. Measured runs may now proceed.** Pilot results
that set the frozen values are recorded under "Pilot results" below.

## Research question

What are the physical resource costs (time, energy, thermals, storage I/O) of
SISA-based machine unlearning versus naive retraining when a resource-constrained
FL edge node must **remove identified-compromised data**?

## Framing (unlearning cost, not attack potency)

The contribution is the *physical cost of removal on edge hardware*, in the machine-
unlearning tradition where a removal is mandated by detection or compliance (e.g. a
data source is flagged as compromised, or invokes a right-to-be-forgotten), **not**
by the data's measured harm to the model. The label-flip poison defines a clean,
well-specified set of samples that must be unlearned; whether it degrades the global
model is secondary and reported honestly.

Pilot finding: at this scale the confined poison (all attack samples in slices 3–4
of one shard, on 1 of 4 nodes) has **no measurable effect on global utility**
(recall stays 1.0, attack-success 0.0 on the clean test set) — FedAvg dilutes the
single compromised client and the IDS recall is near-saturated. This *reinforces*
the framing: unlearning is driven by the data being compromised, not by a visible
performance drop, and the question is what removal costs.

## Pre-registered hypotheses

- **H1** SISA reduces time-to-recovery (TTR) vs. naive local retraining.
- **H2** SISA reduces recovery energy (Wh); naive shows higher sustained power draw.
- **H3** Naive recovery thermally taxes the fanless Pi far more than SISA — measured by **throttled-seconds and clock-frequency degradation over the recovery window, not peak temperature** (peak saturates near ~87°C for any sustained load, so it does not discriminate). Pilot: naive 156 s throttled / clock 2400→1500 MHz vs SISA 13 s / stays 2400 MHz.
- **H4** SISA shifts cost toward storage I/O (checkpoint writes, iowait). *Tested, not assumed* — a negligible-I/O result at this model scale is reported honestly; the sensitivity study locates the model size where I/O dominates.
- **H5** The unlearned model preserves utility: recovered-model F1/recall on the clean test set is within ΔF1 ≤ 2 pp of a clean-trained model, in both arms. Both arms provide **exact local** unlearning by construction (see Guarantee scope), so this checks that removal did not damage utility — not that an attack was reversed (the confined poison has no measurable global effect; see Framing).
- **H6** SISA imposes measurable Phase-1 overhead (per-round wall time, bytes written). Total cost of ownership = Phase-1 overhead + recovery cost.

## Design

| Element | Value |
|---|---|
| Arms | A: naive local retrain-from-scratch · B: SISA rollback + partial replay |
| SISA config | S=5 shards × R=5 slices, seeded permutation assignment |
| Poison | Label flip attack→benign, **fraction 1.0** of attack samples in slices 3–4 of shard 1 ("source fully compromised at time τ"; defines the set to unlearn) |
| Topology | 4 clients: simulated nodes 1–3 on the host, the Raspberry Pi as the 4th node over the LAN |
| Data per node | Stratified subsample **1,000,000** rows; equal across all 4 nodes (FedAvg weighting) |
| Clean test set | 100,000 rows, stratified, disjoint from all training subsamples, evaluated host-side |
| Rounds | Phase 1: 10 · Phase 4 rejoin: 5 |
| Recovery budget (naive) | 10 epochs on retained data (= Phase-1 local budget), fresh Adam per epoch |
| Seeds | **N = 10 paired** (42–51). Justified below — pilot effect sizes are large, but N=10 tightens the bootstrap CIs and guards any marginal metric |
| Order | A/B counterbalanced across seeds (ABBA BAAB ...) |
| Client heterogeneity | The 3 simulated clients get different CPU/memory limits (3.0/2.0/1.5 cores) to model a heterogeneous federation, fixed across runs |
| Network conditions | Primary: clean LAN. Secondary factor (one seed both arms): "realistic WAN" via toxiproxy latency (`--profile wan`, 40 ms ± 20 ms on the Fleet API) |
| Cooldown gate | Stable thermal plateau (range ≤ 2°C for 120 s) at the passive idle floor before every measured phase sequence — the fanless Pi never reaches a low absolute temperature, so runs are equalised on a *stable* start state, not a fixed number. The plateau temperature is logged and doubles as an ambient proxy |
| Ambient | DS18B20 (TO-92, GPIO4, `dtoverlay=w1-gpio`) logged in the telemetry `Ambient_C` column; if absent, the cooldown plateau + a manual room-temp note in `notes.txt` stand in |
| Power | FNB58 inline on the Pi's supply for ALL runs (constant overhead); USB-logged host-side at ~100 sps via `experiments/fnirsi_logger.py`; energy = trapezoidal ∫W dt per phase window, reported **net of the idle baseline**; `PSU_MAX_CURRENT=5000` pinned; **5 A meter→Pi cable** (thin cables under-volt the Pi — see runbook) |
| Statistics | Paired Wilcoxon signed-rank + Cliff's delta + **bootstrap 95% CI on the median paired difference** for TTR, net energy, throttled-seconds, min clock, bytes written. Throttling counted from *live* flag bits only (occurred bits are sticky). Utility mean ± 95% CI |

![MSc hybrid topology](../docs/diagrams/msc-thesis-diagrams-msc-thesis.drawio.png)

## Guarantee scope (thesis must state this)

Both arms give **exact local** unlearning (recovered client state provably free of
poison influence) and **approximate global** forgetting: stateless FedAvg cannot
remove already-aggregated poison from the global model client-side. Phase 4 resumes
from the (nominally poisoned) global model identically in both arms and measures
global-model utility over the rejoin rounds — at this scale the confined poison
leaves global utility unchanged (see Framing), so Phase 4 confirms the rejoin does
not perturb utility rather than showing attack-success decay. Detection is out of
scope (oracle poison mask).

Documented deviations from vanilla SISA/FL: constituents never absorb global
weights (required for the rollback guarantee); the FL update is the parameter
average of constituents rather than a prediction ensemble.

## Per-run procedure

Manual, per the Runbook below. Additional runs beyond the paired A/B campaign:
- **Reference runs** — one clean run per seed (`POISON_MODE=off`, standard client)
  for the gold-standard utility ceiling (H5).
- **Sensitivity study** (one seed) — repeat the paired comparison with a scaled-up
  model to locate where checkpoint I/O becomes the dominant SISA cost (H4).
- **Network factor** (one seed, both arms) — repeat under the WAN condition
  (`--profile wan`, `FLEET_PORT=19092`) to show the recovery-cost result holds
  under realistic latency.
- **Unlearning efficacy** — after each recovery, `analysis/unlearning_efficacy.py`
  membership-inference probe confirms the recovered model treats the removed data as
  unseen (reported honestly against its positive control; exact unlearning is
  guaranteed by construction and verified bit-identically by `tests/smoke_test.py`).

## Reproducibility & environment disclosure

The thesis must report, and the artifact must pin:
- **Hardware**: Raspberry Pi 5 rev, SD-card model + class (A1), official 27 W PSU,
  the 5 A meter→Pi cable, FNB58 (firmware), DS18B20 ambient sensor, host machine.
- **Software**: Flower 1.29.0, the pinned torch/numpy versions, and Docker **image
  digests** (not just tags) for `flwr/{superlink,supernode,superexec,base}:1.29.0`.
- **Determinism**: all seeds (torch/numpy/loader), the vendored logger commit, and
  the `prepare_edge_data.py` manifest (subsample + poison provenance) per seed.
- **Measurement**: FNB58 accuracy spec as instrument uncertainty; on-die thermal
  sensor resolution; ambient per run; page cache dropped before each measured run
  (runbook) so I/O and timing are comparable.

**Sample size (why N=10).** The pilot effect sizes are large (recovery 8× faster,
throttled-time 12× lower), so even N=5 would clear significance; N=10 is chosen to
tighten the bootstrap CIs and protect any metric with a smaller effect (e.g. utility
equivalence). With paired non-parametric tests the effect size + CI carry the claim,
not the p-value — full per-seed data is disclosed.

## Pilot results (2026-07-25, seed 42, 1M rows — frozen)

Measured on the physical Pi (fanless), clean power (Anker 100 W cable; input rail
`0x0` at idle, no active under-voltage under load):

- **Subsample fixed at 1M** — naive recovery (10 epochs, ~737k retained rows) runs
  3.7 min and drives the Pi to the 85°C hard limit, so no need for 2M.
- **Naive recovery (Phase 3):** 224 s · peak 87.3°C · **156 s throttled** · clock
  2400→1500 MHz.
- **SISA recovery (Phase 3):** 28 s · peak 86.7°C · **13 s throttled** · clock stays
  2400 MHz → 8× faster, 12× less throttled time (H1, H3).
- **SISA Phase-1 overhead (H6):** ~13 min / 10 rounds; checkpoints 43.7 MB total,
  **3.4 s checkpoint I/O** → **H4 confirmed: I/O is negligible at this model scale**
  (the scaled-model sensitivity study locates where it dominates).
- **Poison:** confined + fraction 1.0 → no measurable global utility effect (see
  Framing); fixed as the well-specified set to unlearn.
- **Determinism** is guaranteed by construction and covered by `tests/smoke_test.py`
  (bit-identical recovery across reruns); spot-check on the first paired hardware run.
- **Analysis** (`analysis/analyze.py`, `evaluate_model.py`) validated end-to-end on
  pilot artifacts, incl. per-phase energy integration.

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

```text
Wall PSU (official 27 W) ──USB-C──> FNB58 IN ──USB-C(5A cable)──> Pi 5 power port
FNB58 data port ──USB cable──────> host machine (runs fnirsi_logger.py)
Pi 5 ──Ethernet──> switch ────────> host machine
```

> The meter→Pi output cable must be a short, thick **5 A / 100 W** USB-C cable.
> A thin cable drops enough voltage to under-volt the Pi (chronic clock throttling
> that confounds every physical metric) even while the meter reads 5.1 V — verified
> in the pilot. Confirm `vcgencmd get_throttled` is `0x0` at idle before any run.

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
# Verify power delivery is clean before spending a run (thin cable ⇒ under-volt).
ssh admin@rasp5node.local "vcgencmd get_throttled"   # want 0x0
# Drop the page cache so filesystem-cache state doesn't skew I/O/timing run-to-run.
ssh admin@rasp5node.local "sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"

experiments/cooldown_gate.sh                  # blocks until the SoC temperature plateaus (stable idle floor)
# Telemetry as a transient systemd unit — survives SSH disconnect (a plain
# nohup/setsid over SSH does not reliably persist). ALWAYS (re)start it with the
# reset-failed + systemd-run pair below, once per run. Do NOT use `systemctl start
# msc-monitor` to restart it: a transient systemd-run unit is gone after `stop`, so
# `systemctl start` silently no-ops and the run gets NO telemetry (analyze then reads
# the previous run's stale file — caught on sisa-49, whose row was identical to
# naive-49). After each run, verify a fresh hardware_telemetry_<timestamp>.csv exists.
ssh admin@rasp5node.local "sudo systemctl reset-failed msc-monitor 2>/dev/null; \
  sudo systemd-run --unit=msc-monitor --working-directory=/home/admin \
  /home/admin/msc-experiment/monitor.sh"
sudo python3 -u experiments/fnirsi_logger.py > "$RUN/power_fnb58.csv" &
LOGGER_PID=$!
```

Record the ambient temperature (DS18B20 auto-logs it; otherwise note the room
temperature) and the cooldown-plateau temperature in `$RUN/notes.txt`.

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
python3 -m analysis.evaluate_model "$RUN/recovered_model.pt"        # utility (H5)
python3 -m analysis.evaluate_model "$RUN/phase4_checkpoints/round_5.npz"
python3 -m analysis.unlearning_efficacy "$RUN/recovered_model.pt"   # forgetting probe
python3 -m analysis.analyze                                         # summary.csv + paired_stats.csv
```

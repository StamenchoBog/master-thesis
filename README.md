# Master Thesis — Distributed AI for IoT Network Attack Detection

MSc thesis code on Federated Learning for intrusion detection across IoT edge nodes, built with [Flower](https://flower.ai) 1.29.0 and PyTorch on the [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) dataset.

The MSc part asks what it physically costs an edge device to *unlearn* poisoned data. A Raspberry Pi 5 (fanless, A1 SD card) joins a federation of three simulated nodes, gets poisoned, and then recovers either by naive retraining from scratch or with SISA (sharded, sliced training with checkpoint rollback). Time, energy, thermals and SD-card I/O are measured on the Pi during recovery.

The earlier course-assignment work (simulation only, everything in Docker) is kept under [`results/course/`](results/course/).

![MSc architecture](./docs/diagrams/msc-thesis-diagrams-msc-thesis.drawio.png)

## Results at a glance

10 paired seeds (42–51), medians, recovery window on the Pi:

| | Naive | SISA | Ratio |
|---|---|---|---|
| Time to recover | 242.1 s | 29.5 s | 8.2× |
| Net recovery energy | 0.093 Wh | 0.017 Wh | 5.4× |
| Time thermally throttled | 233 s | 27.5 s | 8.5× |
| Written to SD card | 2.75 MB | 12.05 MB | 0.23× |

Every one of these is fully separated between the arms (Cliff's δ = ±1.00, Wilcoxon p = 0.002, the floor for N=10). The catch is model quality: at the default 0.5 threshold the SISA-recovered global model collapses to predicting "attack" for everything (balanced accuracy 0.50 vs 0.60 for naive). With a tuned threshold it recovers (0.71 vs 0.67), so it's a calibration problem caused by averaging the SISA constituents, not lost information.

A one-seed robustness pair under simulated WAN latency (40 ms ± 20 ms) gave 250.8 s vs 31.6 s, the same ~8× gap.

Full write-up: the thesis (Chapters 6–8).

## Repository layout

| Path | What's there |
|---|---|
| `edge_nodes/` | Client side: Flower ClientApp, model, data loading, poisoning, SISA training and both recovery scripts |
| `server/` | Flower ServerApp: aggregation strategy and round config |
| `experiments/` | MSc data prep, FNB58 power logger, cooldown gate, power-log trimming |
| `analysis/` | Paired statistics, utility re-scoring, unlearning checks |
| `tests/` | Smoke test for the unlearning guarantee (synthetic data) |
| `ansible/` | Raspberry Pi provisioning |
| `results/course/` | Course-phase results (FedAvg, FedProx, Krum, TrimmedMean) |
| `results/msc/runs/` | One folder per MSc run plus the aggregated CSVs |
| `docs/diagrams/` | Architecture diagrams |

Run folders are named `naive_seedN` / `sisa_seedN` for the main campaign. The two `wan_*_seed42` folders are the WAN robustness pair; the `wan_` prefix keeps them out of `paired_stats.csv`. Each run folder holds `phases.log`, `power_fnb58.csv`, `hardware_telemetry_*.csv`, `recovery_manifest.json`, `recovered_model.pt`, `results_phase{1,4}.json` and the per-round global checkpoints (`sisa_timings.jsonl` too, for SISA). Large files are stored with Git LFS, so run `git lfs pull` after cloning.

## Setup

1. Download the **TON_IoT Network dataset** and put all 23 `Network_dataset_*.csv` files in `data/` (about 3.3 GB, not in git).
2. Docker Desktop with at least 12 GB memory (Settings > Resources > Memory).
3. Python 3.11+ on the host:

   ```sh
   pip install -e ".[analysis,power]"
   brew install libusb    # only needed for the FNB58 power logger
   ```

## Course-phase simulation (all local)

Everything runs in Docker on one machine:

![Course architecture](./docs/diagrams/msc-thesis-diagrams-course.png)

```sh
docker compose run --rm preprocessor    # build per-node caches, once
docker compose up
```

Settings are environment variables on `superexec-serverapp` in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `FL_STRATEGY` | `fedavg` | `fedavg`, `fedprox` (proximal term for non-IID drift), `krum` (Byzantine-tolerant), `trimmedmean` (drops outliers) |
| `NUM_ROUNDS` | `10` | Federation rounds |
| `MIN_FIT_CLIENTS` / `MIN_EVAL_CLIENTS` / `MIN_AVAILABLE_CLIENTS` | `3` | Clients required per stage |
| `RESULTS_SUFFIX` | empty | Appended to the results filename so variants don't overwrite each other |

## Reproducing the MSc experiment

This is done by hand on purpose: someone has to be there for the power meter and the cooldown wait anyway. One run = one arm (`naive` or `sisa`) for one seed.

### One-time

```sh
NUM_PARTITIONS=4 docker compose run --rm preprocessor     # 4-way caches (3 sim nodes + Pi)
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=<host LAN IP> docker compose -f docker-compose.edge.yml build"
```

Provision the Pi with Ansible first (see below) and keep its checkout of this repo up to date. An old `docker-compose.edge.yml` on the Pi fails quietly (it just ignores `FLEET_PORT`).

Every `docker-compose.edge.yml` command, including `stop` and `down`, needs `HOST_IP` set.

### Per run

```sh
SEED=42; ARM=naive                        # or sisa
RUN=results/msc/runs/${ARM}_seed$SEED     # keep the same shell for the whole run
HOST_IP=<host LAN IP>

# 1. Data prep, then check the Pi has the matching seed. A mismatched cache
#    scatters the poison and SISA degrades to a full retrain.
python3 experiments/prepare_edge_data.py --seed $SEED
scp data/.cache/msc/partition_3_of_4.npz data/.cache/msc/manifest.json \
    admin@rasp5node.local:master-thesis/data/.cache/msc/
ssh admin@rasp5node.local "python3 -c \"import json; print(json.load(open('master-thesis/data/.cache/msc/manifest.json'))['seed'])\""
ssh admin@rasp5node.local "sudo rm -rf msc-experiment/checkpoints/* /dev/shm/sisa_timings.jsonl /dev/shm/recovery_manifest.json /dev/shm/hardware_telemetry_*.csv"
rm -rf results/msc/global_checkpoints && mkdir -p "$RUN"

# 2. Clean start: power check (want 0x0; reboot the Pi if sticky bits are set),
#    drop caches, wait for a thermal plateau, start telemetry and the power logger.
ssh admin@rasp5node.local "vcgencmd get_throttled"
ssh admin@rasp5node.local "sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
experiments/cooldown_gate.sh
ssh admin@rasp5node.local "sudo systemctl reset-failed msc-monitor 2>/dev/null; \
  sudo systemd-run --unit=msc-monitor --working-directory=/home/admin /home/admin/msc-experiment/monitor.sh"
ssh admin@rasp5node.local "sudo systemctl status msc-monitor --no-pager | head -3"   # must be active
sudo python3 -u experiments/fnirsi_logger.py > "$RUN/power_fnb58.csv" &
LOGGER_PID=$!

# 3. Phase 1: 10 poisoned federated rounds.
CLIENT_MODE=$([ $ARM = naive ] && echo standard || echo sisa)
echo "$(date +%s) phase1" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo phase1 > /dev/shm/run_marker"
SEED=$SEED NUM_ROUNDS=10 RESULTS_SUFFIX=_p1 docker compose -f docker-compose.host.yml up -d
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED CLIENT_MODE=$CLIENT_MODE POISON_MODE=flip \
  docker compose -f docker-compose.edge.yml up -d"
docker compose -f docker-compose.host.yml run --rm runner
echo "$(date +%s) idle" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"
mv results/msc/global_checkpoints "$RUN/phase1_checkpoints"

# 4. Phase 3: recovery on the Pi (the measured window).
MODULE=$([ $ARM = naive ] && echo naive_retrain || echo sisa_recover)
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=$HOST_IP docker compose -f docker-compose.edge.yml stop superexec-clientapp-4"
echo "$(date +%s) phase3" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo phase3 > /dev/shm/run_marker"
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED docker compose -f docker-compose.edge.yml run --rm \
  -v \$PWD:/app -w /app --entrypoint python superexec-clientapp-4 -m edge_nodes.$MODULE"
echo "$(date +%s) idle" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo idle > /dev/shm/run_marker"

# 5. Phase 4: 5 rejoin rounds, resumed from the Phase-1 global model.
cp "$RUN/phase1_checkpoints/round_10.npz" results/msc/resume_from.npz
SEED=$SEED NUM_ROUNDS=5 RESULTS_SUFFIX=_p4 INIT_FROM_CHECKPOINT=/results/resume_from.npz \
  docker compose -f docker-compose.host.yml up -d superexec-serverapp
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=$HOST_IP SEED=$SEED CLIENT_MODE=$CLIENT_MODE POISON_MODE=drop \
  docker compose -f docker-compose.edge.yml up -d superexec-clientapp-4"
echo "$(date +%s) phase4" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo phase4 > /dev/shm/run_marker"
docker compose -f docker-compose.host.yml run --rm runner
echo "$(date +%s) done" >> "$RUN/phases.log"; ssh admin@rasp5node.local "echo done > /dev/shm/run_marker"

# 6. Teardown and collect.
sudo kill $LOGGER_PID
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=$HOST_IP docker compose -f docker-compose.edge.yml down; sudo systemctl stop msc-monitor"
docker compose -f docker-compose.host.yml down
scp "admin@rasp5node.local:/dev/shm/hardware_telemetry_*.csv" admin@rasp5node.local:/dev/shm/recovery_manifest.json \
    admin@rasp5node.local:msc-experiment/checkpoints/recovered_model.pt "$RUN/"
[ $ARM = sisa ] && scp admin@rasp5node.local:/dev/shm/sisa_timings.jsonl "$RUN/"
mv results/msc/fedavg_p1.json "$RUN/results_phase1.json"
mv results/msc/fedavg_p4.json "$RUN/results_phase4.json"
mv results/msc/global_checkpoints "$RUN/phase4_checkpoints"
rm -f results/msc/resume_from.npz
```

**WAN condition.** Name the run folder `wan_${ARM}_seed$SEED`, add `--profile wan` to the host `up`/`down` commands, and add `FLEET_PORT=19092` to the Pi's `up` commands so its traffic goes through the 40 ms ± 20 ms toxiproxy link. Nodes 1–3 always go through a separate 5 ms ± 3 ms proxy, standing in for three separate devices.

## Analysis

Run from the repo root:

```sh
python -m analysis.analyze_runs                  # summary.csv + paired_stats.csv (time, energy, thermals, I/O)
python -m analysis.evaluate_utility --phase1 \
    --test-template data/.cache/msc/test_seed{seed}.npz    # utility_evaluation.csv (balanced metrics)
python -m analysis.evaluate_model results/msc/runs/naive_seed42/recovered_model.pt
python -m analysis.evaluate_unlearning results/msc/runs/sisa_seed42/recovered_model.pt
python -m analysis.evaluate_constituents --checkpoints <Pi checkpoint dir>
```

`evaluate_utility` needs one test set per seed. Build them first with `prepare_edge_data.py --seed N` and copy each `test_global.npz` to `test_seedN.npz` (the script's docstring has the loop). Plain recall/F1 are misleading here, since the test set is 96.5% attacks and a model that flags everything still scores recall 1.0.

`experiments/trim_power_logs.py` cuts each `power_fnb58.csv` down to the phase-marker window. Output doesn't change; the files just get small enough for git.

## Tests

```sh
docker run --rm -v "$PWD":/app -w /app --entrypoint python fl-ids-preprocessor:latest tests/smoke_test.py
```

Uses synthetic data (no dataset needed). Checks that poison placement is deterministic and confined, that label flipping only happens in memory, that SISA writes one checkpoint per round/shard/slice, and that recovery rolls back to a clean checkpoint and is bit-identical across reruns.

## Raspberry Pi provisioning

1. Flash Raspberry Pi OS Lite with your SSH public key.
2. Check it's reachable: `ssh -i ~/.ssh/<key> admin@rasp5node.local`
3. Provision:

   ```sh
   cd ansible
   ansible-playbook -i inventory.ini setup_node.yaml --ask-become-pass
   ```

If the key has a passphrase, `ssh-add` it first, since Ansible can't prompt for it.

The playbook stops APT timers, turns swap off permanently (Debian 13 `rpi-swap` → `Mechanism=none`), locks the CPU governor to `performance`, installs Docker with local log rotation, and deploys `~/msc-experiment/monitor.sh`, which writes telemetry to `/dev/shm` so it never adds to the SD-card I/O being measured. Leave the fan unplugged; the thermal behaviour is part of what's being measured.

## License

MIT, see [LICENSE](LICENSE).

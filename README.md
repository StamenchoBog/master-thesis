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

You need a macOS host with [Git LFS](https://git-lfs.com), Python 3.11+ and [Docker Desktop](https://www.docker.com/products/docker-desktop/). The run scripts use macOS tools (`ipconfig`, Homebrew).

**1. Clone the repo.** The run results (checkpoints, power logs, telemetry) are stored with Git LFS, so pull those too:

```sh
git clone https://github.com/StamenchoBog/master-thesis.git
cd master-thesis
git lfs install
git lfs pull
```

**2. Add the dataset.** Download the TON_IoT **Network dataset** (23 `Network_dataset_*.csv` files, about 3.3 GB) from the [UNSW page](https://research.unsw.edu.au/projects/toniot-datasets) and put the files directly in `data/`. They're not in git. Check:

```sh
ls data/Network_dataset_*.csv | wc -l    # should print 23
```

**3. Install the Python packages** in a virtual environment. `analysis` adds the statistics dependencies, `power` adds the USB power-meter driver:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[analysis,power]"
brew install libusb    # needed by the power logger
```

**4. Give Docker enough memory.** Preprocessing loads the whole dataset, so set Docker Desktop to at least 12 GB (Settings > Resources > Memory). Check:

```sh
docker info --format '{{.MemTotal}}' | awk '{printf "%.1f GB\n", $1/1024/1024/1024}'    # ~11.7 GB for a 12 GB setting
```

You only need the Raspberry Pi for the MSc experiment; the course simulation and the analysis run on this machine alone.

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

One run is one arm (`naive` or `sisa`) for one seed. Someone has to be there for the power meter and the cooldown wait, so it isn't fully automated, but each step is a script in `experiments/run/`.

### One-time

```sh
NUM_PARTITIONS=4 docker compose run --rm preprocessor     # 4-way caches (3 sim nodes + Pi)
ssh admin@rasp5node.local "cd master-thesis && HOST_IP=<host LAN IP> docker compose -f docker-compose.edge.yml build"
```

Provision the Pi with Ansible first (see below) and keep its checkout of this repo up to date. An old `docker-compose.edge.yml` on the Pi quietly ignores `FLEET_PORT`.

### Per run

Run the steps in order from the repo root. Each one takes the arm (`naive` or `sisa`), the seed, and optionally `--wan`:

```sh
experiments/run/01_prepare.sh naive 42
experiments/run/02_instruments.sh naive 42
experiments/run/03_train.sh naive 42
experiments/run/04_recover.sh naive 42
experiments/run/05_rejoin.sh naive 42
experiments/run/06_collect.sh naive 42
```

| Script | What it does |
|---|---|
| `01_prepare.sh` | Builds the seed's data, copies the Pi's partition over, checks the seed matches, clears old Pi state |
| `02_instruments.sh` | Checks power (`throttled=0x0`), drops caches, waits for the cooldown plateau, starts telemetry and the power logger |
| `03_train.sh` | Phase 1: 10 federated rounds with poisoned labels |
| `04_recover.sh` | Phase 3: `naive_retrain` or `sisa_recover` on the Pi (the measured window) |
| `05_rejoin.sh` | Phase 4: 5 rejoin rounds from the Phase-1 global model |
| `06_collect.sh` | Stops everything and moves the results into the run folder |

Results land in `results/msc/runs/<arm>_seed<N>/` (`wan_<arm>_seed<N>/` with `--wan`). `HOST_IP` is detected from `en0`; set it yourself if that's wrong. If `02_instruments.sh` reports sticky throttle bits, reboot the Pi and run it again. If a run stops halfway, stop the power logger and the containers by hand (the first lines of `06_collect.sh`), then delete the run folder before starting over.

With `--wan`, the Pi's traffic goes through a 40 ms ± 20 ms toxiproxy link. Nodes 1–3 always go through a separate 5 ms ± 3 ms proxy, standing in for three separate devices.

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

Linting (config in `.flake8`, `.yamllint` and `.shellcheckrc`):

```sh
pip install -e ".[dev]"
flake8 .
yamllint .
shellcheck experiments/*.sh experiments/run/*.sh
```

The same checks run on GitHub on every push.

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

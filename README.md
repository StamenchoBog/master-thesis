# Master Thesis — Distributed AI for IoT Network Attack Detection

Federated Learning for intrusion detection on IoT edge nodes, built with [Flower](https://flower.ai) 1.29.0 and PyTorch on the [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) dataset.

The MSc experiment measures what it physically costs an edge device to *unlearn* poisoned data. A fanless Raspberry Pi 5 joins three simulated nodes, gets poisoned, and recovers either by retraining from scratch (naive) or with SISA (sharded training with checkpoint rollback). Time, energy, thermals and SD-card I/O are measured on the Pi during recovery.

![MSc architecture](./docs/diagrams/msc-thesis-diagrams-msc-thesis.drawio.png)

## Results

Medians over 10 paired seeds (42–51), measured during recovery on the Pi:

| | Naive | SISA | Ratio |
|---|---|---|---|
| Time to recover | 242.1 s | 29.5 s | 8.2× |
| Net energy | 0.093 Wh | 0.017 Wh | 5.4× |
| Time thermally throttled | 233 s | 27.5 s | 8.5× |
| Written to SD card | 2.75 MB | 12.05 MB | 0.23× |

All four are fully separated between the arms (Cliff's δ = ±1.00, Wilcoxon p = 0.002). The trade-off is model quality: at the default 0.5 threshold the SISA model predicts "attack" for everything (balanced accuracy 0.50 vs 0.60). A tuned threshold recovers it (0.71 vs 0.67). A WAN-latency check on seed 42 kept the same ~8× gap.

## Repository layout

| Path | Contents |
|---|---|
| `edge_nodes/` | Flower ClientApp, model, data loading, poisoning, SISA, both recovery methods |
| `server/` | Flower ServerApp and aggregation strategies |
| `experiments/` | Data prep, power logger, cooldown gate; `run/` holds the per-step run scripts |
| `analysis/` | Statistics, utility scoring, unlearning checks |
| `tests/` | Smoke test for the unlearning guarantee |
| `ansible/` | Raspberry Pi provisioning |
| `results/course/` | Course-phase results (FedAvg, FedProx, Krum, TrimmedMean) |
| `results/msc/runs/` | One folder per run (`naive_seedN`, `sisa_seedN`, `wan_*`) and the aggregated CSVs |

Run data is stored with Git LFS: run `git lfs pull` after cloning.

## Setup

- The TON_IoT **Network dataset**: put the 23 `Network_dataset_*.csv` files in `data/` (3.3 GB, not in git).
- Docker Desktop with at least 12 GB memory.
- Python 3.11+: `pip install -e ".[analysis,power]"`
- For the power logger only: `brew install libusb`

## Course simulation

Everything runs in Docker on one machine:

![Course architecture](./docs/diagrams/msc-thesis-diagrams-course.png)

```sh
docker compose run --rm preprocessor
docker compose up
```

The strategy (`fedavg`, `fedprox`, `krum`, `trimmedmean`), rounds and minimum clients are environment variables on `superexec-serverapp` in `docker-compose.yml`.

## MSc experiment

### Preparing the Pi (once)

1. Flash Raspberry Pi OS Lite with your SSH key, then provision it:

   ```sh
   cd ansible && ansible-playbook -i inventory.ini setup_node.yaml --ask-become-pass
   ```

   This disables swap, locks the CPU governor, installs Docker and deploys the telemetry script. If your SSH key has a passphrase, `ssh-add` it first. Leave the fan unplugged; the thermals are part of what's being measured.

2. Clone this repo on the Pi as `~/master-thesis` and keep it up to date.
3. Build the 4-node data caches and the Pi's client image:

   ```sh
   NUM_PARTITIONS=4 docker compose run --rm preprocessor
   ssh admin@rasp5node.local "cd master-thesis && HOST_IP=<host LAN IP> docker compose -f docker-compose.edge.yml build"
   ```

### Running one arm for one seed

Each step takes the same arguments: `naive|sisa SEED [--wan]`. Run them in order from the repo root:

1. `experiments/run/01_prepare.sh naive 42`: builds the seed's data and copies it to the Pi.
2. `experiments/run/02_instruments.sh naive 42`: waits for the Pi to cool down, then starts telemetry and the power logger. If it reports throttle bits, reboot the Pi and run it again.
3. `experiments/run/03_train.sh naive 42`: Phase 1, 10 federated rounds with poisoned data.
4. `experiments/run/04_recover.sh naive 42`: Phase 3, the recovery on the Pi (the measured part).
5. `experiments/run/05_rejoin.sh naive 42`: Phase 4, 5 more federated rounds.
6. `experiments/run/06_collect.sh naive 42`: stops everything and saves the results to `results/msc/runs/naive_seed42/`.

`HOST_IP` is read from `en0`; set it yourself if that's the wrong interface. `--wan` puts a 40 ms ± 20 ms delay on the Pi's link and saves to `wan_<arm>_seed<N>/`. If a run fails partway, stop the logger and containers (the first lines of `06_collect.sh`) and delete the run folder before retrying.

### Analysis

```sh
python -m analysis.analyze_runs          # summary.csv, paired_stats.csv
python -m analysis.evaluate_utility --phase1 --test-template data/.cache/msc/test_seed{seed}.npz
python -m analysis.evaluate_unlearning results/msc/runs/sisa_seed42/recovered_model.pt
```

`evaluate_utility` needs a test set per seed; its docstring shows how to build them. Use its balanced metrics rather than plain recall/F1, since the test set is 96.5% attacks.

## Tests and linting

```sh
docker run --rm -v "$PWD":/app -w /app --entrypoint python fl-ids-preprocessor:latest tests/smoke_test.py

pip install -e ".[dev]"
flake8 . && yamllint . && shellcheck experiments/*.sh experiments/run/*.sh
```

The smoke test uses synthetic data and checks that SISA recovery rolls back to a clean checkpoint and gives identical results on every rerun. Linting also runs on GitHub on every push.

## License

MIT, see [LICENSE](LICENSE).

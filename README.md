# Master Thesis — Distributed AI for IoT Network Attack Detection

MSc thesis on Federated Learning for intrusion detection across IoT edge nodes. Extended with a physical-edge experiment comparing Naive Retraining against SISA machine unlearning after a data-poisoning attack.

The FL system is built with [Flower](https://flower.ai) 1.29.0 and PyTorch, trained on the [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) dataset. The original course-assignment work (simulation only, everything in Docker) is kept as-is under [`results/course/`](results/course/); the MSc work builds on top of that.

Topology for the MSc part: three simulated edge nodes on the host, plus a real Raspberry Pi joining over the LAN as the 4th node.

![MSc architecture](./docs/diagrams/msc-thesis-diagrams-msc-thesis.drawio.png)

## Repository layout

| Directory | Description |
|---|---|
| `edge_nodes/` | ClientApp: local training, model, data pipeline, poisoning, SISA |
| `server/` | ServerApp: aggregation strategy and round config |
| `ansible/` | Provisioning for the Pi |
| `experiments/` | Data prep, power + thermal logging for the MSc experiment |
| `analysis/` | Stats, utility scoring, unlearning verification |
| `tests/` | Checks the unlearning guarantee (rollback, determinism) |
| `results/course/` | Course-phase results (FedAvg, FedProx, Krum, TrimmedMean) |
| `results/msc/` | MSc experiment outputs |
| `docs/diagrams/` | Architecture diagrams |

Full design and results for the MSc experiment are written up in the thesis itself (`docs/thesis/sections/`). The N=10 paired runs are done; raw output is under [`results/msc/runs/`](results/msc/runs/) (`summary.csv`, `paired_stats.csv`, `utility_evaluation.csv`).

## Dataset

Download the **TON_IoT Network dataset** from [research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets) and place all 23 `Network_dataset_*.csv` files into `data/`. The CSVs are excluded from git (3.3 GB total).

## Quick start (all-local simulation)

Course-phase architecture, everything simulated in Docker on one machine:

![Course architecture](./docs/diagrams/msc-thesis-diagrams-course.png)

![Course architecture (detailed)](./docs/diagrams/msc-thesis-diagrams-course-detailed.drawio.png)

Requires Docker Desktop with at least 12 GB memory allocated (Settings > Resources > Memory).

1. Download the dataset and place the 23 CSV files in `data/` (see above).
2. Build partition caches (run once):

   ```sh
   docker compose run --rm preprocessor
   ```

3. Start the federation:

   ```sh
   docker compose up
   ```

## Configuration

Set via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `FL_STRATEGY` | `fedavg` | `fedavg` (weighted average), `fedprox` (proximal term for non-IID drift), `krum` (Byzantine-tolerant), `trimmedmean` (drops outliers) |
| `NUM_ROUNDS` | `10` | Federation rounds |
| `MIN_FIT_CLIENTS` | `3` | Clients required for training |
| `MIN_EVAL_CLIENTS` | `3` | Clients required for evaluation |
| `MIN_AVAILABLE_CLIENTS` | `3` | Clients required before starting |

## Physical edge node (Raspberry Pi 5)

For the MSc experiment the federation gets a real Raspberry Pi 5 (4 GB, fan unplugged on purpose, A1 SD card) as a 4th node over the LAN, alongside the three simulated nodes. Provisioning is handled by Ansible.

### Bootstrap

1. Generate an SSH key and flash Raspberry Pi OS Lite with the flasher (add the public key).
2. Verify connectivity:

   ```sh
   ssh -i ~/.ssh/<generated-private-ssh-key> admin@rasp5node.local
   ```

3. Provision the node:

   ```sh
   cd ansible
   ansible-playbook -i inventory.ini setup_node.yaml --ask-become-pass
   ```

> If your SSH key has a passphrase, Ansible can't prompt for it (non-interactive). Run `ssh-add ~/.ssh/<generated-private-ssh-key>` first, once per shell session, so the agent already holds the decrypted key. Otherwise the playbook fails with `Permission denied (publickey)` even though the key is fine on the Pi side.

The playbook does the usual experiment prep: stops APT timers, disables swap for good (Debian 13's `rpi-swap` set to `Mechanism=none`), locks the CPU governor to `performance`, sets up Docker with local log rotation, and drops a small telemetry script (`~/msc-experiment/monitor.sh`) that writes to `/dev/shm` so its own I/O doesn't pollute the SD-card metrics we're trying to measure.

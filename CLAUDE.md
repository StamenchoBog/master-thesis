# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

MSc thesis repo on Federated Learning for IoT network intrusion detection: an FL intrusion-detection system built with [Flower](https://flower.ai) 1.29.0 + PyTorch, trained on the TON_IoT dataset, extended with a physical Raspberry Pi 5 edge node for the MSc experiment (Naive Retraining vs. SISA unlearning after data poisoning — see the MSc section below).

The codebase is unified at the repo root. The earlier course-assignment phase (simulation-only) is preserved at the `course-final` git tag; its results live in `results/course/`. Key directories: `edge_nodes/` (ClientApp), `server/` (ServerApp), `ansible/` (Pi provisioning), `experiments/` (MSc protocol + runbook + data prep), `analysis/` (stats), `results/{course,msc}/`. Experiment runs are executed manually per the runbook in `experiments/protocol.md` — deliberately not automated, since the operator must be present for the power meter and cooldown anyway.

## Commands

Requires Docker Desktop with ≥12 GB memory (Settings > Resources > Memory). There is no test suite or linter configured in this repo.

**One-time setup:** download the TON_IoT dataset (23 `Network_dataset_*.csv` files, ~3.3 GB, from research.unsw.edu.au/projects/toniot-datasets) into `data/`. CSVs are gitignored.

```sh
# Build per-node .npz partition caches (idempotent — skips if caches + columns.json already exist)
docker compose run --rm preprocessor

# Force re-preprocessing (e.g. after changing NUM_PARTITIONS or the dataset)
rm -rf data/.cache

# Run the full all-local simulation (server + 3 edge nodes; reproduces course-phase results)
docker compose up

# Provision the Raspberry Pi edge node (from ansible/; ssh-add the key first if passphrase-protected)
ansible-playbook -i inventory.ini setup_node.yaml --ask-become-pass
```

To change the aggregation strategy or round config, edit the `superexec-serverapp` environment block in `docker-compose.yml` (`FL_STRATEGY`, `NUM_ROUNDS`, `MIN_FIT_CLIENTS`, `MIN_EVAL_CLIENTS`, `MIN_AVAILABLE_CLIENTS`) — there is no CLI flag for these, they're read from env vars in `server/server_app.py`. Set `RESULTS_SUFFIX` (e.g. `_chaos`) when running a variant so it doesn't overwrite `results/course/<strategy>.json`.

To run `flwr run .` manually from the host instead of via the `runner` service, use the `[superlink.local]` profile in `.flwr/config.toml` (`127.0.0.1:9093`); the `docker`-profile address (`superlink:9093`) only resolves inside the compose network.

## Architecture

### Flower deployment topology

This uses Flower's **deployment engine** (not simulation), so the moving parts are actual containers wired together in `docker-compose.yml`:

- **`superlink`** — central coordinator/message broker. Everything else registers with it.
- **`supernode-{1,2,3}`** — one per edge node, each started with `--node-config "partition-id=N num-partitions=3"`. This is how each node learns which data slice is "its own" (read in `client_fn` via `context.node_config`).
- **`superexec-clientapp-{1,2,3}`** — execution plugins attached to each supernode that actually run `edge_nodes/client_app.py`'s `ClientApp`.
- **`superexec-serverapp`** — execution plugin that runs `server/server_app.py`'s `ServerApp`. Aggregation config (strategy, round count, min-clients) is env-driven here.
- **`runner`** — a throwaway container that submits the job with `flwr run .` and streams logs; it's the trigger, not a persistent service.
- **`preprocessor`** — one-shot job, must complete before any `clientapp` container starts (`depends_on: condition: service_completed_successfully`).
- **`toxiproxy` / `toxiproxy-setup`** — sits between supernodes and superlink (`superlink-fl` proxy, listen `19092` → upstream `superlink:9092`) to inject 5ms±3ms latency for chaos testing. The proxy itself is declared statically in `toxiproxy.json`, but the toxic (latency) is added via a REST call in `toxiproxy-setup` at container startup, since toxiproxy's static config format can't express toxics. Simulation-only — the MSc hybrid setup uses the real LAN instead.

`pyproject.toml`'s `[tool.flwr.app.components]` wires `serverapp = "server.server_app:app"` / `clientapp = "edge_nodes.client_app:app"` — this is how the Flower CLI/runtime finds the two `ClientApp`/`ServerApp` instances regardless of which container invokes them.

For the MSc hybrid topology, `docker-compose.host.yml` runs superlink + serverapp + simulated nodes 1–3 with the Fleet API (9092) published to the LAN, and `docker-compose.edge.yml` (run on the Pi) starts supernode 4 (partition 3 of 4) + clientapp pointing at the host's LAN IP. The MSc federation therefore has 4 clients and needs its own 4-way caches (`NUM_PARTITIONS=4 docker compose run --rm preprocessor`); the course simulation keeps its 3-way caches.

### Data pipeline

`edge_nodes/preprocess.py` runs once against all 23 CSVs: finds numeric columns common to every file (metadata/label columns dropped first), then re-splits the file *list* (not rows) across `NUM_PARTITIONS` so each node gets a disjoint set of whole CSVs, standard-scales its slice independently, and writes `data/.cache/partition_{i}_of_{N}.npz`. `edge_nodes/data_loader.py` just loads that cache and does an 80/20 train/val split per node — no cross-node shuffling, so partitions are naturally non-IID by file/source.

### Training round flow

1. `server_app.py::fit_config` sends `{local_epochs: 1, lr: 0.001}` to every client each round — hyperparameters live server-side so they can change without rebuilding client images.
2. `client_app.py::FlowerClient.fit` trains locally, returns weights + sample count + `train_loss`.
3. Aggregation happens via whichever strategy `FL_STRATEGY` selects in `build_strategy()`: `fedavg` (custom `ResilientFedAvg`), `fedprox` (proximal term for non-IID drift), `krum` (Byzantine-tolerant, picks most central update), `trimmedmean` (drops outliers before averaging).
4. `ResilientFedAvg.aggregate_fit` is the one subtlety worth knowing: if fewer than `min_fit_clients` respond in a round, it returns the *previous* round's weights instead of aggregating a partial/incomplete update — this is what keeps chaos-engineering runs from diverging on a bad round.
5. `FlowerClient.evaluate` computes accuracy/precision/recall/F1 from raw TP/FP/FN counts (not sklearn) against the global model on each node's local val split; `_weighted_average` in `server_app.py` combines per-client metrics weighted by sample count.
6. After every round, `_save_results()` overwrites `results/course/<FL_STRATEGY><RESULTS_SUFFIX>.json` with the full history so far (not append-only — the whole dict is rewritten each time).

### Model

`edge_nodes/model.py::IDSModel` is a plain 4-layer MLP (128→64→32→1, ReLU/BatchNorm/Dropout(0.3) per hidden layer, sigmoid output) trained with `BCELoss` — binary "attack vs. benign" classification, not multi-class attack typing.

### Interpreting results

Per `results/course/README.md`, metric priority is **Recall > Precision > F1** — for an IDS, a missed attack (false negative) is worse than a false alarm. `results/course/*.json` files are the durable course-phase outputs; MSc experiment artifacts go to `results/msc/`.

## MSc Thesis: Hardware & SISA Experiment (Physical Edge Node)

The MSc experiment transitions the project from a pure software simulation to a physical Systems Engineering experiment. The core objective is to compare **Naive Retraining** vs. **SISA (Sharded, Isolated, Sliced, and Aggregated) Machine Unlearning** after a data poisoning attack, measuring the physical costs on edge hardware. The full pre-registered design (hypotheses, variables, protocol, statistics) lives in `experiments/protocol.md`.

### Hardware Constraints (The "Why")

The physical edge node is intentionally bottlenecked to simulate real-world IoT environments:

- **Compute & Thermals:** Raspberry Pi 5 (4GB). The **cooling fan is intentionally unplugged**. This forces thermal throttling at 85°C. Naive Retraining is expected to hit this limit, causing CPU clock degradation.
- **Storage I/O:** 128GB A1 SD Card. An A1 card is limited to ~500 random write IOPS. This is the critical bottleneck for SISA, which requires saving multiple model state checkpoints. The SD card forces a severe I/O wait state, extending Time-to-Recovery. **Do not suggest SSD upgrades; the I/O bottleneck is the metric.** Note: the I/O claim is *tested, not assumed* — the primary IDSModel checkpoints are small, so a one-seed sensitivity study with a scaled-up model locates where SD I/O becomes dominant.
- **Power:** FNIRSI FNB58 external tester logs physical Watt-hours to contrast the short, intense power spike of Naive Retraining against the sustained, I/O-bound power draw of SISA.

### Topology & Telemetry Rules

- **Hybrid Network:** The central Aggregator (`superlink` and `serverapp`) runs on the primary host machine via `docker-compose.host.yml`. The Raspberry Pi joins over the local network as a remote `supernode` via `docker-compose.edge.yml`.
- **Zero I/O Noise:** The Pi is provisioned via Ansible (`ansible/setup_node.yaml`) with swap disabled (Debian 13 `rpi-swap` set to `Mechanism=none`). The `monitor.sh` telemetry script writes exclusively to `/dev/shm` (RAM). This ensures the SD card's I/O metrics reflect *only* the FL workload.
- **Determinism:** All PyTorch initializations and dataset splits must use hardcoded seeds (e.g., `torch.manual_seed(42)`) to ensure run-to-run comparability.

### Experiment design (decided)

- **Arms:** Naive = local retrain-from-scratch on retained data (fixed epoch budget); SISA = drop poisoned samples, roll affected constituent back to last clean slice checkpoint, retrain only affected slices. Both arms give exact *local* unlearning and approximate *global* forgetting (measured over K=5 post-recovery rounds with identical protocol) — stateless FedAvg makes exact global removal impossible client-side.
- **SISA config:** S=5 shards × R=5 slices, stratified, seeded; persistent constituents across rounds; FL update = parameter average of constituents (documented deviation from vanilla SISA's prediction ensembling).
- **Poisoning:** seeded label flip (attack→benign) confined to the later slices of one shard ("source compromised at time τ") — exercises both sharding and slicing. Detection is out of scope (oracle poison mask).
- **Data:** Pi trains on a stratified ~1M-row subsample of its partition; a global clean test set is carved from the val regions (rows past the 0.8 split are never trained on).
- **Both-sided costs:** SISA's Phase-1 overhead (checkpoint writes each round) is measured alongside its recovery savings — report total cost of ownership, not just time-to-recovery.
- **Protocol discipline:** cooldown gate (stable thermal plateau at the passive idle floor — the fanless Pi can't reach a low absolute temperature, so runs are equalised on a stable start state) before every measured run; A/B order counterbalanced; N=5 paired seeds; Wilcoxon signed-rank + Cliff's delta; parameters frozen in `experiments/protocol.md` after the pilot, before measured runs.

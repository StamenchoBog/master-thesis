# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

MSc thesis repo on Federated Learning for IoT network intrusion detection: an FL intrusion-detection system built with [Flower](https://flower.ai) 1.29.0 + PyTorch, trained on the TON_IoT dataset, extended with a physical Raspberry Pi 5 edge node for the MSc experiment (Naive Retraining vs. SISA unlearning after data poisoning — see the MSc section below).

**Status: the physical experiment is complete.** N=10 paired seeds (42-51) have been run on the Pi and analyzed (`results/msc/runs/`, `results/msc/runs/{summary,paired_stats,utility_evaluation}.csv`). Remaining work is analysis refinement and writing up `docs/thesis/sections/*.tex` (Macedonian thesis text, LaTeX, destined for Overleaf — see `docs/thesis/sections/README.md`). `experiments/protocol.md` is FROZEN (pre-registered) and is the authoritative source for hypotheses, design parameters, and the run procedure — treat it as ground truth over any narrative elsewhere. Note: it's gitignored (kept locally only, superseded by `docs/thesis/sections/` as the public write-up), so it exists in this checkout but won't be in a fresh clone.

The codebase is unified at the repo root. The earlier course-assignment phase (simulation-only) is preserved unmodified in `results/course/`. Key directories: `edge_nodes/` (ClientApp), `server/` (ServerApp), `ansible/` (Pi provisioning), `experiments/` (MSc protocol + data prep + power/thermal tooling), `analysis/` (stats + figures), `results/{course,msc}/`, `docs/thesis/` (thesis LaTeX sections). Experiment runs are executed manually per the runbook in `experiments/protocol.md` — deliberately not automated, since the operator must be present for the power meter and cooldown anyway.

## Commands

Requires Docker Desktop with ≥12 GB memory (Settings > Resources > Memory). There is no linter configured in this repo. The one test suite is `tests/smoke_test.py` (synthetic-data verification of the unlearning guarantee — see Testing below).

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

# Run the unlearning-guarantee smoke test (needs torch + flwr, so run inside the client image)
docker run --rm -v "$PWD":/app -w /app --entrypoint python fl-ids-preprocessor:latest tests/smoke_test.py

# Re-run the MSc analysis pipeline over existing run artifacts
python -m analysis.analyze_runs             # summary.csv + paired_stats.csv (H1-H4, H6)
python -m analysis.evaluate_utility --phase1 --test-template data/.cache/msc/test_seed{seed}.npz  # H5, balanced metrics
python -m analysis.evaluate_unlearning results/msc/runs/sisa_seed42/recovered_model.pt             # membership-inference forgetting probe
python -m analysis.evaluate_constituents --checkpoints ~/msc-experiment/checkpoints                # single-run diagnostic, see docstring
python -m analysis.make_figures             # regenerates docs/thesis figures + tables (not tracked — see note below)
```

To change the aggregation strategy or round config, edit the `superexec-serverapp` environment block in `docker-compose.yml` (`FL_STRATEGY`, `NUM_ROUNDS`, `MIN_FIT_CLIENTS`, `MIN_EVAL_CLIENTS`, `MIN_AVAILABLE_CLIENTS`) — there is no CLI flag for these, they're read from env vars in `server/server_app.py`. Set `RESULTS_SUFFIX` (e.g. `_chaos`) when running a variant so it doesn't overwrite `results/course/<strategy>.json`.

To run `flwr run .` manually from the host instead of via the `runner` service, use the `[superlink.local]` profile in `.flwr/config.toml` (`127.0.0.1:9093`); the `docker`-profile address (`superlink:9093`) only resolves inside the compose network.

Thesis figures/tables (`docs/thesis/figures/`, `docs/thesis/sections/tables/`) are regenerable from `results/msc/` via `analysis/make_figures.py` and are **not tracked in git** — regenerate locally before pasting into Overleaf; the source of truth is the CSVs in `results/msc/runs/`.

## Architecture

### Flower deployment topology

This uses Flower's **deployment engine** (not simulation), so the moving parts are actual containers wired together in `docker-compose.yml`:

- **`superlink`** — central coordinator/message broker. Everything else registers with it.
- **`supernode-{1,2,3}`** — one per edge node, each started with `--node-config "partition-id=N num-partitions=3"`. This is how each node learns which data slice is "its own" (read in `client_fn` via `context.node_config`).
- **`superexec-clientapp-{1,2,3}`** — execution plugins attached to each supernode that actually run `edge_nodes/client_app.py`'s `ClientApp`.
- **`superexec-serverapp`** — execution plugin that runs `server/server_app.py`'s `ServerApp`. Aggregation config (strategy, round count, min-clients) is env-driven here.
- **`runner`** — a throwaway container that submits the job with `flwr run .` and streams logs; it's the trigger, not a persistent service.
- **`preprocessor`** — one-shot job, must complete before any `clientapp` container starts (`depends_on: condition: service_completed_successfully`).
- **`toxiproxy` / `toxiproxy-setup`** — sits between supernodes and superlink (`superlink-fl` proxy, listen `19092` → upstream `superlink:9092`) to inject 5ms±3ms latency. The proxy itself is declared statically in `toxiproxy.json`, but the toxic (latency) is added via a REST call in `toxiproxy-setup` (a `curlimages/curl` container — the toxiproxy image has no shell) at startup, since toxiproxy's static config format can't express toxics.

`pyproject.toml`'s `[tool.flwr.app.components]` wires `serverapp = "server.server_app:app"` / `clientapp = "edge_nodes.client_app:app"` — this is how the Flower CLI/runtime finds the two `ClientApp`/`ServerApp` instances regardless of which container invokes them.

For the MSc hybrid topology, `docker-compose.host.yml` runs superlink + serverapp + simulated nodes 1–3 with the Fleet API (9092) published to the LAN, and `docker-compose.edge.yml` (run on the Pi) starts supernode 4 (partition 3 of 4) + clientapp pointing at the host's LAN IP. The MSc federation therefore has 4 clients and needs its own 4-way caches (`NUM_PARTITIONS=4 docker compose run --rm preprocessor`); the course simulation keeps its 3-way caches. The three simulated host clients get different CPU/memory limits in `docker-compose.host.yml` to model a heterogeneous federation. `toxiproxy.msc.json` defines two proxies: `superlink-sim` (supernode-1/2/3 always route through it, 5ms±3ms, standing in for separate devices — added after the N=10 campaign, which ran with nodes 1–3 connected directly) and `superlink-fl` (the Pi's link, only with `--profile wan` + `FLEET_PORT=19092` on the edge, 40ms±20ms). The one-seed WAN robustness pair is done (`results/msc/runs/wan_{naive,sisa}_seed42/`, excluded from `paired_stats.csv` by the `wan_` prefix). Every `docker compose -f docker-compose.edge.yml ...` command (incl. `stop`/`down`) needs `HOST_IP` set because of the `${HOST_IP:?}` interpolation.

### Data pipeline

`edge_nodes/preprocess.py` runs once against all 23 CSVs: finds numeric columns common to every file (metadata/label columns dropped first), then re-splits the file *list* (not rows) across `NUM_PARTITIONS` so each node gets a disjoint set of whole CSVs, standard-scales its slice independently, and writes `data/.cache/partition_{i}_of_{N}.npz`. `edge_nodes/data_loader.py` just loads that cache and does an 80/20 train/val split per node (`TRAIN_FRACTION` in `sisa_partition.py`) — no cross-node shuffling, so partitions are naturally non-IID by file/source.

For the MSc experiment, `experiments/prepare_edge_data.py` builds a second cache generation under `data/.cache/msc/`: equal-sized stratified subsamples per node (all 4 nodes get the same row count — FedAvg weights by sample count, so unequal sizes would dilute the Pi's contribution) plus a disjoint global test set. It also computes and embeds the poison index set for the Pi's partition (seeded label flips, attack→benign, confined to specific SISA shard/slices) — labels on disk stay clean; `data_loader.py`'s `POISON_MODE` env var (`off`/`flip`/`drop`) controls whether/how poison is applied at load time.

### Training round flow

1. `server_app.py::fit_config` sends `{local_epochs: 1, lr: 0.001}` to every client each round — hyperparameters live server-side so they can change without rebuilding client images.
2. `client_app.py::FlowerClient.fit` trains locally, returns weights + sample count + `train_loss`.
3. Aggregation happens via whichever strategy `FL_STRATEGY` selects in `build_strategy()`: `fedavg` (custom `ResilientFedAvg`), `fedprox` (proximal term for non-IID drift), `krum` (Byzantine-tolerant, picks most central update), `trimmedmean` (drops outliers).
4. `ResilientFedAvg.aggregate_fit` is the one subtlety worth knowing: if fewer than `min_fit_clients` respond in a round, it returns the *previous* round's weights instead of aggregating a partial/incomplete update — this is what keeps chaos-engineering runs from diverging on a bad round.
5. `FlowerClient.evaluate` computes accuracy/precision/recall/F1 from raw TP/FP/FN counts (not sklearn) against the global model on each node's local val split; `_weighted_average` in `server_app.py` combines per-client metrics weighted by sample count.
6. After every round, `_save_results()` overwrites `results/{course,msc}/<FL_STRATEGY><RESULTS_SUFFIX>.json` with the full history so far (not append-only — the whole dict is rewritten each time).

### Model

`edge_nodes/model.py::IDSModel` is a plain 4-layer MLP (128→64→32→1, ReLU/BatchNorm/Dropout(0.3) per hidden layer, sigmoid output) trained with `BCELoss` — binary "attack vs. benign" classification, not multi-class attack typing. BatchNorm running stats matter when comparing/averaging checkpoints (see the SISA collapse note below) — always call `.eval()` before scoring a loaded model.

### Interpreting results

Per `results/course/README.md`, metric priority is **Recall > Precision > F1** for the course-phase FL comparison — for an IDS, a missed attack (false negative) is worse than a false alarm. For the MSc experiment, raw recall/F1 are **misleading** on the ~96.5%-attack test set (a majority-class model scores recall 1.0 / F1 0.98 with zero benign detection) — utility (H5) must be read via `analysis/evaluate_utility.py`'s imbalance-robust metrics (balanced accuracy, MCC, ROC-AUC/PR-AUC), not the plain `analysis/evaluate_model.py` output. `results/course/*.json` are the durable course-phase outputs; MSc experiment artifacts go to `results/msc/`.

## MSc Thesis: Hardware & SISA Experiment (Physical Edge Node)

The MSc experiment transitions the project from a pure software simulation to a physical Systems Engineering experiment. The core objective is to compare **Naive Retraining** vs. **SISA (Sharded, Isolated, Sliced, and Aggregated) Machine Unlearning** after a data poisoning attack, measuring the physical costs on edge hardware. The full pre-registered design (hypotheses, variables, protocol, statistics, runbook) lives in `experiments/protocol.md` — **read it before touching anything under `experiments/` or `edge_nodes/{sisa_*,naive_retrain}.py`**, since parameters (S=5 shards × R=5 slices, poison shard/slice/fraction, N=10 seeds, etc.) are frozen post-pilot and any change invalidates comparability with already-measured runs.

### Hardware constraints (the "why")

The physical edge node is intentionally bottlenecked to simulate real-world IoT environments:

- **Compute & thermals:** Raspberry Pi 5 (4GB). The **cooling fan is intentionally unplugged**. This forces thermal throttling at 85°C. Naive Retraining hits this limit (measured: clock 2400→1500 MHz, ~156s throttled at N=1 pilot); SISA does not.
- **Storage I/O:** 128GB A1 SD Card (~500 random write IOPS). This is SISA's bottleneck — it checkpoints every constituent's state after every slice. At the primary model scale this cost was measured as negligible (H4 tested, not assumed); a scaled-up-model sensitivity study locates where SD I/O becomes dominant. **Do not suggest SSD upgrades; the I/O bottleneck is the metric.**
- **Power:** FNIRSI FNB58 external tester (`experiments/fnirsi_logger.py`, vendored + patched) logs physical Watt-hours to contrast Naive Retraining's short intense spike against SISA's sustained, I/O-bound draw.

### Topology & telemetry rules

- **Hybrid network:** the central Aggregator (`superlink` + `serverapp`) runs on the host via `docker-compose.host.yml`. The Pi joins over the LAN as a remote `supernode` via `docker-compose.edge.yml`.
- **Zero I/O noise:** the Pi is provisioned via Ansible (`ansible/setup_node.yaml`) with swap disabled (Debian 13 `rpi-swap` set to `Mechanism=none`). The `monitor.sh` telemetry script writes exclusively to `/dev/shm` (RAM), so the SD card's I/O metrics reflect *only* the FL workload.
- **Determinism:** all PyTorch/NumPy RNGs are seeded (`SEED` env var, derived per-partition/round/shard/slice in `edge_nodes/sisa_client.py`) so recovery is bit-identical across reruns — verified by `tests/smoke_test.py`.

### Experiment design (frozen — see `experiments/protocol.md` for the authoritative version)

- **Arms:** Naive (`edge_nodes/naive_retrain.py`) = local retrain-from-scratch on retained data, fixed epoch budget. SISA (`edge_nodes/sisa_recover.py`) = roll the affected constituent back to its last pre-poison checkpoint, drop poisoned samples, replay only the affected shard's slices; untouched constituents keep their Phase-1 checkpoints. Both give exact *local* unlearning by construction; global forgetting is approximate (stateless FedAvg can't remove already-aggregated poison client-side).
- **Poisoning:** seeded label flip (attack→benign), confined to slices ≥3 of shard 1, fraction 1.0 ("source fully compromised at time τ"). Placement and selection are pure functions of `(n_samples, seed, S, R)` (`edge_nodes/sisa_partition.py`), so identical across arms.
- **Guarantee scope:** exact local unlearning + approximate global forgetting — the thesis must not overclaim exact global removal.
- **Known finding (H5 / SISA utility):** the SISA client's FL update is the *parameter average* of its 5 constituents (a documented deviation from vanilla SISA's prediction ensembling, required to fit FedAvg). This causes a majority-class collapse (balanced accuracy ≈0.50) not visible in raw recall/F1 — see `analysis/evaluate_utility.py` and `analysis/evaluate_constituents.py`, which isolates whether averaging (vs. shard isolation itself) is the cause.
- **Protocol discipline:** cooldown gate (`experiments/cooldown_gate.sh`, waits for a stable thermal plateau, not an absolute temperature — the fanless Pi never reaches a low absolute temperature) before every measured run; A/B order counterbalanced across seeds; N=10 paired seeds (42–51); Wilcoxon signed-rank + Cliff's delta + bootstrap 95% CI on the paired difference (`analysis/analyze_runs.py`).

### Testing

`tests/smoke_test.py` is the one automated test: synthetic 2000-row data, no dataset download required. It verifies (1) poison placement is deterministic and confined to target slices, (2) `POISON_MODE=flip` semantics (in-memory corruption, disk stays clean), (3) one SISA checkpoint per (round, shard, slice), and (4) recovery rolls back to a pre-poison checkpoint and is bit-identical across reruns. It must run inside the client image (needs torch + flwr) — see the Commands section above. This is the mechanism that backs the "exact unlearning by construction" claim; `analysis/evaluate_unlearning.py` is the complementary *empirical* check (membership-inference probe) run against real recovered models from actual experiment runs.

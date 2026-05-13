# Federated Learning for IoT Network Attack Detection

Proof-of-concept FL system that trains an intrusion detection model across three edge nodes without sharing raw network traffic data. Built with [Flower](https://flower.ai) 1.29.0 and PyTorch on the [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) dataset.

![Architecture](./docs/diagrams/msc-thesis-diagrams-course.png)

![Architecture (detailed)](./docs/diagrams/msc-thesis-diagrams-course-detailed.drawio.png)

## Dataset

Download the **TON_IoT Network dataset** from [research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets) and place all 23 `Network_dataset_*.csv` files into `data/`. The CSVs are excluded from git (3.3 GB total).

## Quick start

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
| `FL_STRATEGY` | `fedavg` | Strategy: `fedavg`, `fedprox`, `krum`, `trimmedmean` |
| `NUM_ROUNDS` | `10` | Federation rounds |
| `MIN_FIT_CLIENTS` | `3` | Clients required for training |
| `MIN_EVAL_CLIENTS` | `3` | Clients required for evaluation |
| `MIN_AVAILABLE_CLIENTS` | `3` | Clients required before starting |

## Components

| Directory | Description |
|---|---|
| `server/` | Aggregation strategy and round configuration |
| `edge_nodes/` | Local training, model, and data pipeline |

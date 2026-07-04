# Server (ServerApp)

Drives federated rounds: selects clients, broadcasts global weights, collects local updates, and aggregates them. Submitted as a job to the SuperLink — not a persistent process.

## Strategies

| `FL_STRATEGY` | Description |
|---|---|
| `fedavg` | Weighted average of all client updates (baseline) |
| `fedprox` | Adds a proximal penalty to limit local drift; handles non-IID data |
| `krum` | Selects the most central update; tolerates one malicious client |
| `trimmedmean` | Discards outlier updates before averaging; Byzantine-robust |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `FL_STRATEGY` | `fedavg` | Aggregation strategy |
| `NUM_ROUNDS` | `10` | Federation rounds |
| `MIN_FIT_CLIENTS` | `3` | Clients required for training |
| `MIN_EVAL_CLIENTS` | `3` | Clients required for evaluation |
| `MIN_AVAILABLE_CLIENTS` | `3` | Clients required before starting |

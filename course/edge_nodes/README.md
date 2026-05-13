# Edge Nodes (ClientApp)

Each node trains the IDS model on its local data partition and returns updated weights to the server. No raw data leaves the node.

## Files

| File | Description |
|---|---|
| `client_app.py` | FlowerClient — local training and evaluation per round |
| `model.py` | IDS_Model — three-layer MLP binary classifier |
| `data_loader.py` | Loads a data partition from `.npz` cache |
| `preprocess.py` | Finds common columns and builds `.npz` partition caches |
| `Dockerfile` | Extends `flwr/superexec` with project dependencies |
| `requirements.txt` | Python dependencies |

## Preprocessing

Run once before starting the federation:

```sh
docker compose run --rm preprocessor
```

Scans all CSV files for numeric columns present in every file, splits them into N partitions, and saves each as a compressed `.npz` cache. Skips automatically on subsequent runs if caches already exist.

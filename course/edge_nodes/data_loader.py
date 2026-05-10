import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache")
COLUMNS_FILE = os.path.join(CACHE_DIR, "columns.json")

DROP_COLS = ["ts", "src_ip", "dst_ip", "src_mac", "dst_mac", "Unnamed: 0", "type"]


def _cache_path(partition_id: int, num_partitions: int) -> str:
    return os.path.join(CACHE_DIR, f"partition_{partition_id}_of_{num_partitions}.npz")


def load_data(partition_id: int, num_partitions: int, batch_size: int = 512):
    """Load the TON_IoT dataset and return data loaders for one partition.

    Uses a two-level cache strategy:
    1. If a .npz cache exists for this partition, load it directly (~1s).
    2. Otherwise, load from CSV, keep only the agreed-upon numeric columns
       (from columns.json), normalise, and save to cache.

    The column list in columns.json is created by preprocess.py by taking the
    intersection of numeric columns across all CSV files. This guarantees every
    node builds a model with the same input_dim, so global weights are always
    compatible during aggregation.

    Args:
        partition_id:    Index of this node (0-based).
        num_partitions:  Total number of nodes.
        batch_size:      Mini-batch size for both loaders.

    Returns:
        trainloader: DataLoader for the training split.
        valloader:   DataLoader for the validation split.
        input_dim:   Number of features — passed to IDS_Model at construction time.
    """
    cache = _cache_path(partition_id, num_partitions)

    if os.path.exists(cache):
        print(f"[Node {partition_id}] Loading from cache: {cache}")
        data = np.load(cache)
        X, y = data["X"], data["y"]
    else:
        csvs = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Place the TON_IoT dataset there and re-run.")

        if not os.path.exists(COLUMNS_FILE):
            raise FileNotFoundError(
                f"Column list not found at {COLUMNS_FILE}. "
                "Run the preprocessor service first: docker compose run preprocessor"
            )
        with open(COLUMNS_FILE) as f:
            use_cols = json.load(f)

        assigned = np.array_split(csvs, num_partitions)[partition_id]
        print(f"[Node {partition_id}] Preprocessing {len(assigned)} file(s): {[os.path.basename(f) for f in assigned]}")

        df = pd.concat(
            [pd.read_csv(f, low_memory=False) for f in assigned],
            ignore_index=True,
        )
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
        y = df.pop("label").values.astype(np.int64)

        available = [c for c in use_cols if c in df.columns]
        X = StandardScaler().fit_transform(df[available].values).astype(np.float32)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
        print(f"[Node {partition_id}] Cache saved to {cache}")

    split = int(0.8 * len(X))

    def make_loader(Xa, ya, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return make_loader(X[:split], y[:split], True), make_loader(X[split:], y[split:], False), X.shape[1]

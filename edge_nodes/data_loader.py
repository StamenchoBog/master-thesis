import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache")


def _cache_path(partition_id: int, num_partitions: int) -> str:
    return os.path.join(CACHE_DIR, f"partition_{partition_id}_of_{num_partitions}.npz")


def load_data(partition_id: int, num_partitions: int, batch_size: int = 512):
    """Load a pre-built partition from the .npz cache and return train/val data loaders.

    Args:
        partition_id:   Index of this node (0-based).
        num_partitions: Total number of nodes.
        batch_size:     Mini-batch size for both loaders.

    Returns:
        Tuple of (trainloader, valloader, input_dim).
    """
    cache = _cache_path(partition_id, num_partitions)
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"Cache not found: {cache}. "
            "Run the preprocessor first: docker compose run --rm preprocessor"
        )

    print(f"[Node {partition_id}] Loading from cache: {cache}")
    data = np.load(cache)
    X, y = data["X"], data["y"]
    split = int(0.8 * len(X))

    def make_loader(Xa, ya, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return make_loader(X[:split], y[:split], True), make_loader(X[split:], y[split:], False), X.shape[1]

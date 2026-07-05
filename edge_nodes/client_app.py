import os
import random

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .data_loader import load_data
from .model import IDSModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.getenv("SEED", "42"))
CLIENT_MODE = os.getenv("CLIENT_MODE", "standard")  # standard (Arm A) | sisa (Arm B)
RECOVERED_MODEL_PATH = os.getenv("RECOVERED_MODEL_PATH", "")


def seed_everything(seed: int):
    """Fix all RNGs for run-to-run comparability (MSc determinism requirement)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FlowerClient(NumPyClient):
    """Flower client that trains and evaluates the IDS model on a local data partition."""

    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = torch.nn.BCELoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def _apply_incoming(self, parameters):
        """Adopt the broadcast global weights, or the recovered local model once.

        After Phase-3 recovery the incoming global weights still carry poison
        influence, so the first post-recovery fit starts from the recovered
        model instead (one-time override, marked with a .used file).
        """
        marker = RECOVERED_MODEL_PATH + ".used" if RECOVERED_MODEL_PATH else ""
        if RECOVERED_MODEL_PATH and os.path.exists(RECOVERED_MODEL_PATH) and not os.path.exists(marker):
            self.model.load_state_dict(torch.load(RECOVERED_MODEL_PATH, map_location=DEVICE))
            open(marker, "w").close()
            print(f"[Client] Rejoin: starting from recovered model {RECOVERED_MODEL_PATH}")
        else:
            self.set_parameters(parameters)

    def fit(self, parameters, config):
        """Train locally for one round; hyperparameters come from the server config."""
        self._apply_incoming(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.get("lr", 0.001)))
        self.model.train()
        total_loss, batches = 0.0, 0
        for _ in range(int(config.get("local_epochs", 1))):
            for X, y in self.trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                optimizer.zero_grad()
                loss = self.criterion(self.model(X).squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
        return self.get_parameters({}), len(self.trainloader.dataset), {"train_loss": total_loss / max(batches, 1)}

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set (F1 over accuracy: imbalanced data)."""
        self.set_parameters(parameters)
        total_loss = 0.0
        tp = fp = fn = correct = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in self.valloader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                pred = self.model(X).squeeze()
                total_loss += self.criterion(pred, y).item()
                predicted = (pred > 0.5).long()
                labels = y.long()
                correct += (predicted == labels).sum().item()
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()
        n = len(self.valloader.dataset)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        return total_loss / max(len(self.valloader), 1), n, {
            "accuracy": correct / n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def client_fn(context: Context):
    """Instantiate a FlowerClient; the partition comes from the SuperNode's --node-config."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    seed_everything(SEED + partition_id)
    trainloader, valloader, input_dim = load_data(partition_id, num_partitions)
    model = IDSModel(input_dim=input_dim).to(DEVICE)
    return FlowerClient(model, trainloader, valloader).to_client()


# Arm B swaps in the SISA client. The import sits below FlowerClient on purpose:
# sisa_client imports FlowerClient back from this module.
if CLIENT_MODE == "sisa":
    from .sisa_client import client_fn as _selected_client_fn
else:
    _selected_client_fn = client_fn

app = ClientApp(client_fn=_selected_client_fn)

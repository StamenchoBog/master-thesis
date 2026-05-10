import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .data_loader import load_data
from .model import IDS_Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
    """Flower client that trains and evaluates the IDS model on a local data partition."""

    def __init__(self, model, trainloader, valloader):
        """Store the model and the train/validation data loaders for this node."""
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        """Extract model weights as a list of NumPy arrays to send to the server."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Replace the model weights with the global weights received from the server."""
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on local data for one federation round.

        The server sends the current global weights and a config dict with
        `local_epochs` and `lr`. After training, the updated weights and
        average training loss are returned to the server for aggregation.
        """
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config.get("lr", 0.001)))
        criterion = torch.nn.BCELoss()
        self.model.train()
        total_loss, batches = 0.0, 0
        for _ in range(int(config.get("local_epochs", 1))):
            for X, y in self.trainloader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                optimizer.zero_grad()
                loss = criterion(self.model(X).squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
        return self.get_parameters({}), len(self.trainloader.dataset), {"train_loss": total_loss / max(batches, 1)}

    def evaluate(self, parameters, config):
        """Evaluate the global model on this node's local validation set.

        Returns BCE loss, number of samples, and four classification metrics:
        accuracy, precision, recall, and F1. Precision and recall matter more
        than accuracy here because the dataset is class-imbalanced — a model
        predicting 'normal' for everything would score high accuracy but zero recall.
        """
        self.set_parameters(parameters)
        criterion = torch.nn.BCELoss()
        total_loss = 0.0
        tp = fp = fn = correct = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in self.valloader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                pred = self.model(X).squeeze()
                total_loss += criterion(pred, y).item()
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
    """Instantiate a FlowerClient for this SuperNode.

    Flower calls this once per run. The partition-id and num-partitions values
    come from the --node-config flag set on each SuperNode in docker-compose,
    so each node automatically trains on a different slice of the dataset.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader, input_dim = load_data(partition_id, num_partitions)
    model = IDS_Model(input_dim=input_dim).to(DEVICE)
    return FlowerClient(model, trainloader, valloader).to_client()


app = ClientApp(client_fn=client_fn)

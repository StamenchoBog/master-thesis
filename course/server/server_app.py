import json
import os

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, FedTrimmedAvg, Krum

FL_STRATEGY = os.getenv("FL_STRATEGY", "fedavg")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
MIN_FIT_CLIENTS = int(os.getenv("MIN_FIT_CLIENTS", "2"))
MIN_EVAL_CLIENTS = int(os.getenv("MIN_EVAL_CLIENTS", "2"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/results")
RESULTS_SUFFIX = os.getenv("RESULTS_SUFFIX", "")

_results = {
    "strategy": FL_STRATEGY,
    "num_rounds": NUM_ROUNDS,
    "train_loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
}


def _aggregate(metrics):
    """Compute a weighted average of per-client metrics."""
    total = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


def _save():
    """Persist the current results dict to a JSON file after every round."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{FL_STRATEGY}{RESULTS_SUFFIX}.json"), "w") as f:
        json.dump(_results, f, indent=2)


def log_fit_metrics(metrics):
    """Aggregate training metrics from all clients and append to results."""
    agg = _aggregate(metrics)
    _results["train_loss"].append(round(agg.get("train_loss", 0), 6))
    _save()
    return agg


def log_evaluate_metrics(metrics):
    """Aggregate evaluation metrics from all clients and append to results."""
    agg = _aggregate(metrics)
    for key in ("accuracy", "precision", "recall", "f1"):
        _results[key].append(round(agg.get(key, 0), 6))
    _save()
    return agg


def fit_config(server_round: int) -> dict:
    """Return hyperparameters sent to every client before each training round."""
    return {"local_epochs": 1, "lr": 0.001}


def get_strategy():
    """Instantiate the aggregation strategy selected via the FL_STRATEGY env var."""
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=log_fit_metrics,
        evaluate_metrics_aggregation_fn=log_evaluate_metrics,
    )
    if FL_STRATEGY == "fedprox":
        return FedProx(**common, proximal_mu=0.1)
    elif FL_STRATEGY == "krum":
        return Krum(**common, num_malicious_clients=1)
    elif FL_STRATEGY == "trimmedmean":
        return FedTrimmedAvg(**common)
    else:
        return FedAvg(**common)


def server_fn(context):
    """Entry point called by Flower to create the server components for this run."""
    return ServerAppComponents(
        strategy=get_strategy(),
        config=ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=3600),
    )


app = ServerApp(server_fn=server_fn)

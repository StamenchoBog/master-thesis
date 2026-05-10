import os

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, FedTrimmedAvg, Krum

FL_STRATEGY = os.getenv("FL_STRATEGY", "fedavg")
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "10"))
MIN_FIT_CLIENTS = int(os.getenv("MIN_FIT_CLIENTS", "2"))
MIN_EVAL_CLIENTS = int(os.getenv("MIN_EVAL_CLIENTS", "2"))
MIN_AVAILABLE_CLIENTS = int(os.getenv("MIN_AVAILABLE_CLIENTS", "2"))


def weighted_metrics(metrics):
    """Aggregate any set of per-node metrics into global values, weighted by dataset size.

    Works for both fit metrics (train_loss) and evaluate metrics (accuracy,
    precision, recall, f1) without needing a separate function per metric.
    """
    total = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


def fit_config(server_round: int) -> dict:
    """Return hyperparameters sent to every client before each training round."""
    return {"local_epochs": 1, "lr": 0.001}


def get_strategy():
    """Instantiate the FL aggregation strategy selected via the FL_STRATEGY env var.

    All strategies share the same base settings (client counts, config fn,
    metrics aggregation). Strategy-specific parameters are added on top:
        - fedprox:      proximal_mu controls the penalty pulling local models
                        toward the global model (useful when data is non-IID).
        - krum:         tolerates up to num_malicious_clients Byzantine nodes by
                        selecting the update closest to its neighbours.
        - trimmedmean:  discards the highest and lowest updates before averaging,
                        making aggregation robust to outliers.
        - fedavg:       plain weighted average — the baseline.
    """
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_metrics,
        evaluate_metrics_aggregation_fn=weighted_metrics,
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
        config=ServerConfig(num_rounds=NUM_ROUNDS),
    )


app = ServerApp(server_fn=server_fn)

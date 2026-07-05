import json
import os

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
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


def _weighted_average(metrics):
    """Weighted by sample count; requires identical metric keys across clients."""
    total = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


def _save_results():
    """Persist the current results dict to a JSON file after every round."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{FL_STRATEGY}{RESULTS_SUFFIX}.json"), "w") as f:
        json.dump(_results, f, indent=2)


def aggregate_fit_metrics(metrics):
    agg = _weighted_average(metrics)
    _results["train_loss"].append(round(agg.get("train_loss", 0), 6))
    _save_results()
    return agg


def aggregate_eval_metrics(metrics):
    agg = _weighted_average(metrics)
    for key in ("accuracy", "precision", "recall", "f1"):
        _results[key].append(round(agg.get(key, 0), 6))
    _save_results()
    return agg


def fit_config(server_round: int) -> dict:
    """Hyperparameters live server-side so clients never need a rebuild to change them."""
    return {"local_epochs": 1, "lr": 0.001}


class ResilientFedAvg(FedAvg):
    """FedAvg that reuses the previous round's weights if too few clients respond."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_good_weights = None

    def aggregate_fit(self, server_round, results, failures):
        if len(results) < self.min_fit_clients:
            print(f"[Round {server_round}] Only {len(results)} clients responded — reusing previous weights.")
            return self._last_good_weights, {}
        aggregated = super().aggregate_fit(server_round, results, failures)
        self._last_good_weights = aggregated
        return aggregated


def _with_global_checkpoints(strategy):
    """Save the aggregated global weights after every round (MSc experiment only).

    Enabled via SAVE_GLOBAL_CHECKPOINTS=1; the course simulation leaves it off.
    Checkpoints allow the round-resume logic after client-side recovery.
    """
    if os.getenv("SAVE_GLOBAL_CHECKPOINTS", "0") != "1":
        return strategy
    original = strategy.aggregate_fit

    def aggregate_fit(server_round, results, failures):
        aggregated, metrics = original(server_round, results, failures)
        if aggregated is not None:
            ckpt_dir = os.path.join(RESULTS_DIR, "global_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            arrays = parameters_to_ndarrays(aggregated)
            np.savez(os.path.join(ckpt_dir, f"round_{server_round}.npz"), *arrays)
        return aggregated, metrics

    strategy.aggregate_fit = aggregate_fit
    return strategy


def build_strategy():
    """Instantiate the aggregation strategy selected via the FL_STRATEGY env var."""
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )
    # Phase-4 rejoin (MSc): resume from the Phase-1 global model instead of a
    # fresh random init — global forgetting-by-dilution is measured from there.
    init_ckpt = os.getenv("INIT_FROM_CHECKPOINT", "")
    if init_ckpt:
        z = np.load(init_ckpt)
        common["initial_parameters"] = ndarrays_to_parameters([z[f] for f in z.files])
        print(f"Resuming global model from {init_ckpt}")
    if FL_STRATEGY == "fedprox":
        return FedProx(**common, proximal_mu=0.1)
    elif FL_STRATEGY == "krum":
        return Krum(**common, num_malicious_clients=1)
    elif FL_STRATEGY == "trimmedmean":
        return FedTrimmedAvg(**common)
    else:
        return ResilientFedAvg(**common)


def server_fn(context):
    return ServerAppComponents(
        strategy=_with_global_checkpoints(build_strategy()),
        config=ServerConfig(num_rounds=NUM_ROUNDS),
    )


app = ServerApp(server_fn=server_fn)

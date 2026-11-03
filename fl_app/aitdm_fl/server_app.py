"""Flower ServerApp: orchestrates the rounds and aggregates the client models."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import (
    Grid,
    NeighboringRelation,
    PrivacyConfig,
    RdpAccountant,
    SamplingMethod,
    ServerApp,
)
from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    QFedAvg,
)

from aitdm_fl.task import (
    evaluate,
    get_device,
    load_test_data,
    model_from_test_set,
    num_classes_of_dataset,
    set_seed,
)

app = ServerApp()

STRATEGIES = ["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi",
              "fedmedian", "fedtrimmedavg", "krum", "qfedavg"]


def build_strategy(context: Context):
    """Create the aggregation strategy selected with run config 'strategy'.
    See docs/04_aggregation_strategies.md for what each one does."""
    cfg = context.run_config
    common = dict(
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
        # Without these the server starts as soon as 2 clients are connected!
        min_train_nodes=int(cfg["min-nodes"]),
        min_evaluate_nodes=int(cfg["min-nodes"]),
        min_available_nodes=int(cfg["min-nodes"]),
    )
    if float(cfg["dp-noise-multiplier"]) > 0:
        # DP-FedAvg averages client updates with EQUAL weights (otherwise one big
        # client would dominate and the privacy analysis would not hold)
        common["weighted_by_key"] = "unit-weight"
    name = str(cfg["strategy"]).lower()

    # --- Plain averaging and its variants for non-IID data -----------------
    if name == "fedavg":
        strategy = FedAvg(**common)
    elif name == "fedprox":
        strategy = FedProx(proximal_mu=float(cfg["proximal-mu"]), **common)
    elif name == "fedavgm":
        strategy = FedAvgM(server_momentum=float(cfg["server-momentum"]), **common)
    # --- Adaptive server optimisers (FedOpt family) --------------------------
    elif name == "fedadam":
        strategy = FedAdam(eta=float(cfg["server-lr"]), **common)
    elif name == "fedyogi":
        strategy = FedYogi(eta=float(cfg["server-lr"]), **common)
    # --- Byzantine-robust aggregation ---------------------------------------
    elif name == "fedmedian":
        strategy = FedMedian(**common)
    elif name == "fedtrimmedavg":
        strategy = FedTrimmedAvg(beta=float(cfg["trim-beta"]), **common)
    elif name == "krum":
        strategy = Krum(num_malicious_nodes=int(cfg["num-malicious"]), **common)
    # --- Fairness across clients ---------------------------------------------
    elif name == "qfedavg":
        # q-FedAvg estimates the Lipschitz constant as 1 / client_learning_rate
        strategy = QFedAvg(client_learning_rate=float(cfg["qfedavg-client-lr"]),
                           q=float(cfg["qfedavg-q"]), **common)
    else:
        raise ValueError(f"Unknown strategy '{name}'. Use one of: {', '.join(STRATEGIES)}")

    # --- Optional: central differential privacy on top of any strategy -----
    noise = float(cfg["dp-noise-multiplier"])
    if noise > 0:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=noise,
            clipping_norm=float(cfg["dp-clipping-norm"]),
            num_sampled_clients=int(cfg["min-nodes"]),
            accountant=build_accountant(strategy, cfg),
        )
    return strategy


def build_accountant(strategy, cfg):
    """Track the privacy budget (epsilon at delta = dp-delta) over the rounds.
    Flower's accountant only supports strategies that aggregate like FedAvg."""
    if not (isinstance(strategy, FedAvg)
            and type(strategy).aggregate_train is FedAvg.aggregate_train):
        print(f"[DP] No epsilon accounting for '{cfg['strategy']}' (FedAvg-style "
              "aggregation only). Noise is still added.")
        return None
    return RdpAccountant(PrivacyConfig(
        target_delta=float(cfg["dp-delta"]),
        population_size=int(cfg["min-nodes"]),
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
        sampling_method=SamplingMethod.NO_AMPLIFICATION,
    ))


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    partitions_dir = str(cfg["partitions-dir"])
    if not partitions_dir:
        raise ValueError(
            "run config 'partitions-dir' is empty. Pass it with "
            "--run-config \"partitions-dir='/abs/path/to/data/partitions/<name>'\""
        )

    # Same seed -> same initial model, same client shuffling, same DP noise
    set_seed(int(cfg["seed"]))

    # Initial global model
    global_model = model_from_test_set(partitions_dir)
    test_loader = load_test_data(partitions_dir)
    device = get_device()
    num_rounds = int(cfg["num-server-rounds"])
    test_every_round = bool(cfg["test-every-round"])

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        """Evaluate the aggregated model on the held-out global test set.

        By default only once, after the last round: the test set is for the final
        numbers, not for watching (and tuning) the training. Follow the training
        with the clients' validation metrics instead (aggregate_evaluate in the log)."""
        if server_round != num_rounds and not (test_every_round and server_round > 0):
            return None
        model = model_from_test_set(partitions_dir)
        model.load_state_dict(arrays.to_torch_state_dict())
        results = evaluate(model, test_loader, device)
        print(f"[round {server_round}] global test: "
              + ", ".join(f"{k}={v:.4f}" for k, v in results.items()))
        return MetricRecord(results)

    # Sent to the clients with every message: they build the model from the
    # shape of their own data and this number of classes
    num_classes = num_classes_of_dataset(partitions_dir)
    strategy = build_strategy(context)
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord({"lr": float(cfg["learning-rate"]),
                                   "num-classes": num_classes}),
        evaluate_config=ConfigRecord({"num-classes": num_classes}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    save_dir = str(cfg["save-dir"])
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(result.arrays.to_torch_state_dict(), out / "final_model.pt")
        history = {
            "run_config": dict(cfg),
            "train_metrics_clientapp": _to_dict(result.train_metrics_clientapp),
            "evaluate_metrics_clientapp": _to_dict(result.evaluate_metrics_clientapp),
            "evaluate_metrics_serverapp": _to_dict(result.evaluate_metrics_serverapp),
        }
        spent = strategy.privacy_spent() if hasattr(strategy, "privacy_spent") else None
        if spent is not None:
            history["privacy"] = {"epsilon": spent.epsilon, "delta": spent.delta}
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        print(f"\nSaved final model and metrics to {out}")


def _to_dict(metrics_per_round) -> dict:
    """{round: MetricRecord} -> {round: {name: value}} for JSON."""
    return {int(rnd): dict(rec) for rnd, rec in metrics_per_round.items()}

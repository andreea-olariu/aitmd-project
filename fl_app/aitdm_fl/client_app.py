"""Flower ClientApp: what every hospital / device / user runs on its own data."""

from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aitdm_fl.task import balanced_class_weights, build_model, derive_seed, get_device
from aitdm_fl.task import evaluate as evaluate_fn
from aitdm_fl.task import load_client_data, set_seed
from aitdm_fl.task import train as train_fn

app = ClientApp()


def get_data_path(context: Context) -> Path:
    """Which partition does this client own?

    * Deployment: each SuperNode can point to its own file with
      --node-config "data-path='/abs/path/client_0.npz'"  (most realistic), or
      just give its partition-id: --node-config "partition-id=0 num-partitions=3"
    * Simulation: Flower sets partition-id automatically (0 ... num-supernodes-1).
    """
    if "data-path" in context.node_config:
        return Path(str(context.node_config["data-path"]))
    partition_id = int(context.node_config["partition-id"])
    return Path(str(context.run_config["partitions-dir"])) / f"client_{partition_id}.npz"


def is_malicious(context: Context) -> bool:
    """Is this client an attacker? (for robustness experiments, see docs/04)

    * Simulation: --run-config "malicious-clients='1,3'"  (partition ids)
    * Deployment: --node-config "data-path='...' malicious=true"
    """
    if context.node_config.get("malicious", False):
        return True
    ids = str(context.run_config["malicious-clients"]).replace(" ", "")
    pid = context.node_config.get("partition-id")
    return pid is not None and ids != "" and str(pid) in ids.split(",")


def poison(global_state: dict, local_state: dict, attack: str) -> dict:
    """Model poisoning: replace the honest update with a harmful one.

    * "sign-flip": push the model 5x in the OPPOSITE direction of learning
    * "noise":     send random Gaussian weights
    * "none":      send the honestly trained weights (e.g. a data poisoner with
                   wrong labels, combined with attack-inflate-examples)
    """
    if attack == "none":
        return local_state
    poisoned = {}
    for name, w_local in local_state.items():
        w_global = global_state[name]
        if not torch.is_floating_point(w_local):
            poisoned[name] = w_local
        elif attack == "sign-flip":
            poisoned[name] = w_global - 5.0 * (w_local - w_global)
        elif attack == "noise":
            poisoned[name] = torch.randn_like(w_local)
        else:
            raise ValueError(f"Unknown attack '{attack}'. Use 'sign-flip', 'noise' or 'none'.")
    return poisoned


def local_model(msg: Message, train_loader) -> torch.nn.Module:
    """The global model, built from what THIS client knows: the shape of its own
    data and the number of classes sent by the server. The client never needs
    the server's test set (in a real federation it does not have it)."""
    num_classes = int(msg.content["config"]["num-classes"])
    model = build_model(sample_shape=tuple(train_loader.dataset.x.shape[1:]),
                        num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


def client_seed(msg: Message, context: Context) -> int:
    """Reproducible seed for this client in this round (run config 'seed')."""
    return derive_seed(int(context.run_config["seed"]), get_data_path(context).name,
                       int(msg.content["config"]["server-round"]))


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Receive the global model, train it on local data, send it back."""
    config = msg.content["config"]
    seed = client_seed(msg, context)
    set_seed(seed)

    batch_size = int(context.run_config["batch-size"])
    train_loader, _ = load_client_data(get_data_path(context), batch_size, seed=seed)
    model = local_model(msg, train_loader)
    global_state = {k: v.clone() for k, v in model.state_dict().items()}

    class_weights = None
    if str(context.run_config["loss-weighting"]) == "balanced":  # rare classes count more
        class_weights = balanced_class_weights(train_loader.dataset.y.numpy(),
                                               int(config["num-classes"]))

    train_loss = train_fn(
        model,
        train_loader,
        epochs=int(context.run_config["local-epochs"]),
        lr=float(config["lr"]),
        device=get_device(),
        proximal_mu=float(config.get("proximal-mu", 0.0)),  # set by FedProx
        class_weights=class_weights,
    )

    # "num-examples" is used by the server to weight the aggregation (FedAvg)
    num_examples = len(train_loader.dataset)
    if is_malicious(context):
        local_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(poison(global_state, local_state, str(context.run_config["attack"])))
        # Nothing stops a client from lying about its size: FedAvg trusts it
        num_examples *= int(context.run_config["attack-inflate-examples"])

    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": num_examples,
        "unit-weight": 1,  # equal weights, used when differential privacy is on
    })
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on this client's local validation data."""
    train_loader, val_loader = load_client_data(get_data_path(context),
                                                int(context.run_config["batch-size"]))
    model = local_model(msg, train_loader)
    results = evaluate_fn(model, val_loader, get_device())

    metrics = MetricRecord({
        "eval_loss": results["loss"],
        "eval_acc": results["accuracy"],
        "eval_bal_acc": results["balanced_accuracy"],
        "num-examples": len(val_loader.dataset),
        "unit-weight": 1,
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

"""Shared evaluation for experiments: one function, used for federated runs and for
the centralised baseline, so that both are measured in exactly the same way.

* validation = all clients' validation sets together (+ one row per client):
  the numbers to look at while you develop and compare models
* test = the global test set: the final number of a model you have chosen
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from aitdm_fl.task import load_npz_arrays, load_pooled_validation, model_from_test_set


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + 512])).argmax(dim=1).numpy()
                           for i in range(0, len(x), 512)])


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def evaluate_model(model_path: Path, partitions_dir: Path) -> dict:
    """{'val': {...}, 'val_per_client': {client: {...}}, 'test': {...}}"""
    model = model_from_test_set(str(partitions_dir))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    x, y, owner = load_pooled_validation(str(partitions_dir))
    pred = predict(model, x)
    result = {"val": scores(y, pred), "val_per_client": {}}
    for cid in np.unique(owner):
        m = owner == cid
        result["val_per_client"][int(cid)] = {"samples": int(m.sum()), **scores(y[m], pred[m])}

    x_test, y_test = load_npz_arrays(partitions_dir / "test.npz", "test")
    result["test"] = scores(y_test, predict(model, x_test))
    return result


def write_results(run_dir: Path, partitions_dir: Path) -> dict:
    """Evaluate run_dir/final_model.pt and store results.json next to it."""
    result = evaluate_model(run_dir / "final_model.pt", partitions_dir)
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        privacy = json.loads(metrics_file.read_text()).get("privacy")
        if privacy:
            result["privacy"] = privacy
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result

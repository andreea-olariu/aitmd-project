"""Centralised baseline: the same model, trained on the union of all clients'
training data (what you could do if the data could be pooled in one place).

Stage 1 of the project asks for this comparison. It works for images and HAR.

    python experiments/centralised.py --partitions data/partitions/dirichlet --epochs 10
    python experiments/centralised.py --partitions data/partitions/dirichlet --epochs 10 \\
        --loss-weighting balanced --seeds 1 2 3 --name central-balanced

Every seed is saved like a federated run (outputs/experiments/<name>/seed<s>/ with
final_model.pt, metrics.json and results.json), so all scripts in explainability/
work on it, and experiments/summarize.py puts it in the same table.
Pick the number of epochs on the validation numbers, never on the test numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import write_results  # noqa: E402
from summarize import summarize_experiment  # noqa: E402

from aitdm_fl.task import (  # noqa: E402
    ArrayDataset,
    balanced_class_weights,
    get_device,
    load_npz_arrays,
    model_from_test_set,
    num_classes_of_dataset,
    set_seed,
    train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralised baseline on the union of the clients' training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loss-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", help="experiment name (default: central-<partitions>)")
    args = parser.parse_args()
    partitions = args.partitions.resolve()
    name = args.name or f"central-{partitions.name}"
    exp_dir = REPO_ROOT / "outputs" / "experiments" / name

    files = sorted(partitions.glob("client_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    parts = [load_npz_arrays(path, "train") for path in files]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    weights = None
    if args.loss_weighting == "balanced":
        weights = balanced_class_weights(y, num_classes_of_dataset(str(partitions)))

    for seed in args.seeds:
        set_seed(seed)
        loader = DataLoader(ArrayDataset(x, y, augment=x.ndim == 4, seed=seed),
                            batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        model = model_from_test_set(str(partitions))
        train(model, loader, epochs=args.epochs, lr=args.lr, device=get_device(),
              class_weights=weights)

        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   run_dir / "final_model.pt")
        run_config = {"partitions-dir": str(partitions), "seed": seed, "mode": "centralised",
                      "epochs": args.epochs, "learning-rate": args.lr,
                      "batch-size": args.batch_size, "loss-weighting": args.loss_weighting}
        (run_dir / "metrics.json").write_text(json.dumps({"run_config": run_config}, indent=2))
        result = write_results(run_dir, partitions)
        print(f"seed {seed}: val balanced acc {result['val']['balanced_accuracy']:.3f}")

    summarize_experiment(exp_dir)


if __name__ == "__main__":
    main()

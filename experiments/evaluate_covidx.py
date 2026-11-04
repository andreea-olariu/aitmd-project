"""Score trained COVIDx models on the test set, without retraining.

    python experiments/evaluate_covidx.py outputs/experiments/covidx-resnet18-none

For every seed<s>/ of the experiment it loads final_model.pt and the settings in
config.json (architecture, image size, data folder, and the threshold chosen on
val_select), scores the test set at that threshold, adds "test" and "test_per_source"
to results.json, writes predictions_test.csv, and updates summary.json.

The test set is for the final numbers of a model you have ALREADY chosen on val_select.
Do not compare candidate models on it, and do not change anything because of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from train_covidx import summarize, write_predictions

from aitdm_fl.covidx import (
    binary_metrics,
    build_model,
    make_loader,
    per_source_metrics,
    predict_log_odds,
    read_manifest,
)
from aitdm_fl.task import get_device


def evaluate_seed(run_dir: Path, device: torch.device, batch_size: int, num_workers: int) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    results_file = run_dir / "results.json"
    results = json.loads(results_file.read_text())
    if config.get("limit") is not None:
        print(f"  NOTE: {run_dir.name} was trained with --limit {config['limit']}; "
              "it is scored on the full test set anyway")

    model = build_model(config["arch"], num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu"))

    data_dir = Path(config["data_dir"])
    records = read_manifest(data_dir, "test")
    loader = make_loader(data_dir, records, train=False, batch_size=batch_size,
                         num_workers=num_workers, image_size=config["image_size"])
    y, score = predict_log_odds(model, loader, device)

    threshold = results["threshold"]  # chosen on val_select during training, never on test
    results["test"] = binary_metrics(y, score, threshold)
    results["test_per_source"] = per_source_metrics(y, score, [r.source for r in records], threshold)
    results_file.write_text(json.dumps(results, indent=2))
    write_predictions(run_dir / "predictions_test.csv", records, y, score)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-set scores of trained COVIDx models.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiments", type=Path, nargs="+",
                        help="experiment folders, e.g. outputs/experiments/covidx-resnet18-none")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = get_device()

    for exp_dir in args.experiments:
        runs = sorted(exp_dir.glob("seed*/final_model.pt"))
        if not runs:
            print(f"{exp_dir}: no seed*/final_model.pt, skipped")
            continue
        all_results = []
        for model_file in runs:
            run_dir = model_file.parent
            result = evaluate_seed(run_dir, device, args.batch_size, args.num_workers)
            t = result["test"]
            print(f"{exp_dir.name}/{run_dir.name}: test AUROC {t['auroc']:.3f}, "
                  f"bal. acc. {t['balanced_accuracy']:.3f}, sens {t['sensitivity']:.3f}, "
                  f"spec {t['specificity']:.3f}")
            for source, m in result["test_per_source"].items():
                parts = [f"{k} {m[k]:.3f}" for k in ("auroc", "sensitivity", "specificity")
                         if m[k] is not None]
                print(f"    {source:<10} n={m['n']:>5}  " + "  ".join(parts))
            all_results.append(result)
        summarize(exp_dir, all_results)


if __name__ == "__main__":
    main()

"""Centralised baseline on COVIDx CXR-4: one CNN trained on all the training data.

    # pipeline check: tiny subset, 1 epoch (minutes, works on CPU too)
    python experiments/train_covidx.py --limit 2000 --epochs 1 --seeds 1 --name smoke

    # the baseline, 3 seeds
    python experiments/train_covidx.py --arch resnet18 --epochs 10 --seeds 1 2 3

    # against the source shortcut: balance the labels inside every source
    python experiments/train_covidx.py --arch resnet18 --sampler within-source

    # once the model is chosen: also score the test set (once!)
    python experiments/train_covidx.py --arch resnet18 --evaluate-test

Model selection. After every epoch the model is scored on the whole validation set and
on "val_select", its bimcv-only 50 / 50 slice built like the test set
(see covidx.selection_indices). The epoch with the best val_select AUROC is kept, and its
decision threshold is the one with the best balanced accuracy on val_select. The full
validation set is reported too, but it rewards the source shortcut, so it is not used
for decisions.

Outputs, in outputs/experiments/<name>/:
    seed<s>/final_model.pt          state_dict of the best epoch (architecture in config.json)
    seed<s>/config.json             all arguments + chosen epoch and threshold
    seed<s>/history.json            training loss and validation metrics per epoch
    seed<s>/results.json            val, val_select (+ test): overall and per source
    seed<s>/predictions_<split>.csv path, source, label, P(positive) of every image
    summary.json                    mean +- std over the seeds
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from aitdm_fl.covidx import (
    ARCHITECTURES,
    SAMPLERS,
    Record,
    best_threshold,
    binary_metrics,
    build_model,
    make_loader,
    make_optimizer,
    per_source_metrics,
    predict_log_odds,
    read_manifest,
    selection_indices,
    sigmoid,
    train_one_epoch,
    warmup_cosine,
)
from aitdm_fl.task import balanced_class_weights, get_device, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_METRICS = ("auroc", "balanced_accuracy", "f1_macro", "sensitivity", "specificity")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralised COVIDx CXR-4 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "256",
                        help="output folder of data_splitting/preprocess_covidx.py")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="start from ImageNet weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sampler", choices=SAMPLERS, default="none",
                        help="how training images are drawn, see covidx.sample_weights")
    parser.add_argument("--loss-weighting", choices=["none", "balanced"], default="none",
                        help="'balanced': class weights inversely proportional to frequency")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="mixed precision on the GPU")
    parser.add_argument("--num-workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int,
                        help="pipeline check: use N random training images (and N/2 val / test)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also score the test set. Do it once, for the final model")
    parser.add_argument("--name", help="experiment name (default: covidx-<arch>-<sampler>)")
    return parser.parse_args()


def subsample(records: list[Record], n: int | None, seed: int = 0) -> list[Record]:
    """n random records (fixed seed), in manifest order; all of them if n is None."""
    if n is None or n >= len(records):
        return records
    keep = np.sort(np.random.default_rng(seed).choice(len(records), size=n, replace=False))
    return [records[i] for i in keep]


def write_predictions(path: Path, records: list[Record], y: np.ndarray, score: np.ndarray) -> None:
    """log_odds is the score metrics are computed on; prob_positive = sigmoid(log_odds)
    is rounded and saturates at 1.0 for confident predictions, so do not rank on it."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "source", "patient_id", "label", "log_odds", "prob_positive"])
        for record, label, s, p in zip(records, y, score, sigmoid(score)):
            writer.writerow([record.path, record.source, record.patient_id, int(label),
                             f"{s:.6f}", f"{p:.6f}"])


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def run_seed(args: argparse.Namespace, seed: int, run_dir: Path, device: torch.device,
             train_records: list[Record], val_records: list[Record],
             test_records: list[Record], select_idx: list[int]) -> dict:
    set_seed(seed)
    train_loader = make_loader(args.data_dir, train_records, train=True,
                               batch_size=args.batch_size, num_workers=args.num_workers,
                               seed=seed, sampler=args.sampler, image_size=args.image_size)
    val_loader = make_loader(args.data_dir, val_records, train=False,
                             batch_size=args.batch_size * 2, num_workers=args.num_workers,
                             image_size=args.image_size)

    model = build_model(args.arch, num_classes=2, pretrained=args.pretrained).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = warmup_cosine(optimizer, total_steps=args.epochs * steps_per_epoch,
                              warmup_steps=max(1, int(args.warmup_epochs * steps_per_epoch)))
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    class_weights = None
    if args.loss_weighting == "balanced":
        class_weights = balanced_class_weights(np.array([r.label for r in train_records]), 2)

    history, best = [], {"auroc": -1.0}
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler=scheduler,
                                     scaler=scaler, class_weights=class_weights, amp=args.amp)
        y_val, p_val = predict_log_odds(model, val_loader, device)
        val = binary_metrics(y_val, p_val)
        sel = binary_metrics(y_val[select_idx], p_val[select_idx])
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val, "val_select": sel,
                        "seconds": time.time() - start})
        print(f"  epoch {epoch:>2}/{args.epochs}  loss {train_loss:.4f}  "
              f"val AUROC {fmt(val['auroc'])}  val_select AUROC {fmt(sel['auroc'])}  "
              f"({time.time() - start:.0f}s)")
        sel_auroc = sel["auroc"] if sel["auroc"] is not None else 0.0
        if sel_auroc > best["auroc"]:
            best = {"auroc": sel_auroc, "epoch": epoch, "y_val": y_val, "p_val": p_val,
                    "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}

    # The chosen model: best epoch on val_select, threshold tuned on val_select.
    model.load_state_dict(best["state"])
    y_val, p_val = best["y_val"], best["p_val"]
    threshold = best_threshold(y_val[select_idx], p_val[select_idx])
    val_sources = [r.source for r in val_records]
    results = {
        "best_epoch": best["epoch"],
        "threshold": threshold,                              # on the log-odds
        "threshold_prob": float(sigmoid(np.array(threshold))),  # the same as P(positive)
        "val_select": binary_metrics(y_val[select_idx], p_val[select_idx], threshold),
        "val": binary_metrics(y_val, p_val, threshold),
        "val_per_source": per_source_metrics(y_val, p_val, val_sources, threshold),
    }
    write_predictions(run_dir / "predictions_val.csv", val_records, y_val, p_val)

    if args.evaluate_test:
        test_loader = make_loader(args.data_dir, test_records, train=False,
                                  batch_size=args.batch_size * 2, num_workers=args.num_workers,
                                  image_size=args.image_size)
        y_test, p_test = predict_log_odds(model, test_loader, device)
        results["test"] = binary_metrics(y_test, p_test, threshold)
        results["test_per_source"] = per_source_metrics(
            y_test, p_test, [r.source for r in test_records], threshold)
        write_predictions(run_dir / "predictions_test.csv", test_records, y_test, p_test)

    torch.save(best["state"], run_dir / "final_model.pt")
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "seed": seed, "mode": "centralised", "best_epoch": best["epoch"],
              "threshold": threshold, "device": str(device),
              "train_size": len(train_records), "val_size": len(val_records),
              "val_select_size": len(select_idx)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def mean_std(values: list[float | None]) -> dict | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": arr.tolist()}


def summarize(exp_dir: Path, all_results: list[dict]) -> None:
    summary = {"experiment": exp_dir.name, "seeds": len(all_results)}
    splits = ["val_select", "val"] + (["test"] if "test" in all_results[0] else [])
    for split in splits:
        summary[split] = {m: mean_std([r[split][m] for r in all_results]) for m in SUMMARY_METRICS}
    if "test" in all_results[0]:
        sources = all_results[0]["test_per_source"]
        summary["test_per_source"] = {
            s: {m: mean_std([r["test_per_source"][s][m] for r in all_results])
                for m in SUMMARY_METRICS} for s in sources}
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{exp_dir.name}: mean +- std over {len(all_results)} seed(s), "
          "at the threshold chosen on val_select")
    for split in splits:
        row = "  ".join(
            f"{m} {summary[split][m]['mean']:.3f}±{summary[split][m]['std']:.3f}"
            for m in SUMMARY_METRICS if summary[split][m] is not None)
        print(f"  {split:<10} {row}")


def main() -> None:
    args = get_arguments()
    args.data_dir = args.data_dir.resolve()
    name = args.name or f"covidx-{args.arch}-{args.sampler}"
    exp_dir = REPO_ROOT / "outputs" / "experiments" / name
    device = get_device()
    if device.type != "cuda":
        print("WARNING: no CUDA GPU visible to PyTorch, training on the CPU will be very slow.")

    train_records = subsample(read_manifest(args.data_dir, "train"), args.limit)
    val_records = subsample(read_manifest(args.data_dir, "val"),
                            None if args.limit is None else args.limit // 2)
    test_records = subsample(read_manifest(args.data_dir, "test"),
                             None if args.limit is None else args.limit // 2)
    select_idx = selection_indices(val_records)
    print(f"{name}: {args.arch} on {device}; train {len(train_records)}, val {len(val_records)} "
          f"(val_select {len(select_idx)}), test {len(test_records) if args.evaluate_test else '-'}")

    all_results = []
    for seed in args.seeds:
        print(f"seed {seed}")
        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_seed(args, seed, run_dir, device, train_records, val_records,
                          test_records, select_idx)
        sel = result["val_select"]
        print(f"  -> best epoch {result['best_epoch']}, threshold {result['threshold']:.2f} log-odds "
              f"(P {result['threshold_prob']:.4f}): "
              f"val_select AUROC {fmt(sel['auroc'])}, bal. acc. {fmt(sel['balanced_accuracy'])}")
        all_results.append(result)
    summarize(exp_dir, all_results)


if __name__ == "__main__":
    main()

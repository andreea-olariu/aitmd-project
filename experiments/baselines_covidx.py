"""Sanity floors for COVIDx CXR-4: models that never look at the image.

    python experiments/baselines_covidx.py
    python experiments/baselines_covidx.py --evaluate-test     # for the final table only

* majority     always P(positive) = the training positive rate (84 %). The floor:
               AUROC 0.5 everywhere.
* source-lr    logistic regression on the one-hot source (bimcv, rsna, ...), nothing
               else. It shows how far the hospital alone gets: high on the full
               validation set, where the source predicts the label, and exactly 0.5 on
               val_select (bimcv only), where it carries no information.

A CNN whose full-val AUROC is close to source-lr's may have learnt little more than the
source. Compare CNNs with each other on val_select, not on the full validation set.

Both use a fixed threshold of 0.5: there is nothing to tune (on val_select their scores
are constant). Results are written in the same format as train_covidx.py, to
outputs/experiments/floor-<model>/seed0/results.json (deterministic, so one "seed").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from aitdm_fl.covidx import (
    Record,
    binary_metrics,
    per_source_metrics,
    read_manifest,
    selection_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5


def majority(train: list[Record]):
    rate = float(np.mean([r.label for r in train]))
    return lambda records: np.full(len(records), rate)


def source_lr(train: list[Record]):
    encoder = OneHotEncoder(handle_unknown="ignore")  # unseen source -> all zeros
    x = encoder.fit_transform([[r.source] for r in train])
    model = LogisticRegression(max_iter=1000).fit(x, [r.label for r in train])
    return lambda records: model.predict_proba(encoder.transform([[r.source] for r in records]))[:, 1]


def evaluate(predict, records: list[Record], select_idx: list[int] | None = None) -> dict:
    y = np.array([r.label for r in records])
    prob = predict(records)
    out = {"metrics": binary_metrics(y, prob, THRESHOLD),
           "per_source": per_source_metrics(y, prob, [r.source for r in records], THRESHOLD)}
    if select_idx is not None:
        out["select"] = binary_metrics(y[select_idx], prob[select_idx], THRESHOLD)
    return out


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Image-free sanity floors for COVIDx.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "256")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also score the test set (for the final report table)")
    args = parser.parse_args()

    train = read_manifest(args.data_dir, "train")
    val = read_manifest(args.data_dir, "val")
    test = read_manifest(args.data_dir, "test")
    select_idx = selection_indices(val)

    rows = []
    for name, fit in (("majority", majority), ("source-lr", source_lr)):
        predict = fit(train)
        v = evaluate(predict, val, select_idx)
        results = {"best_epoch": None, "threshold": THRESHOLD,
                   "val_select": v["select"], "val": v["metrics"], "val_per_source": v["per_source"]}
        if args.evaluate_test:
            t = evaluate(predict, test)
            results["test"], results["test_per_source"] = t["metrics"], t["per_source"]
        run_dir = REPO_ROOT / "outputs" / "experiments" / f"floor-{name}" / "seed0"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        rows.append((name, results))

    splits = ["val_select", "val"] + (["test"] if args.evaluate_test else [])
    print(f"Sanity floors (threshold {THRESHOLD}); val_select = {len(select_idx)} bimcv images, 50/50\n")
    print(f"{'model':<11}" + "".join(f"{s + ' AUROC':>17}{s + ' bal.acc':>19}" for s in splits))
    for name, results in rows:
        print(f"{name:<11}" + "".join(
            f"{fmt(results[s]['auroc']):>17}{fmt(results[s]['balanced_accuracy']):>19}"
            for s in splits))
    sources = sorted({r.source for r in train})
    probs = source_lr(train)([Record("", 0, s, "") for s in sources])
    print("\nsource-lr P(positive) per source: "
          + ", ".join(f"{s} {p:.3f}" for s, p in zip(sources, probs)))


if __name__ == "__main__":
    main()

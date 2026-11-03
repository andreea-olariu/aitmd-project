"""Mean +- std over seeds, and comparison tables of several experiments.

    # one table with all experiments (validation numbers only)
    python experiments/summarize.py outputs/experiments/*

    # the same with the test numbers: for the FINAL table of your report
    python experiments/summarize.py outputs/experiments/* --with-test

Each experiment folder holds one sub-folder per seed (seed1/, seed2/, ...) with a
results.json, written by experiments/run_seeds.py or experiments/centralised.py.
std is the sample standard deviation over the seeds (ddof = 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [("accuracy", "acc."), ("balanced_accuracy", "bal. acc."), ("f1_macro", "macro-F1")]


def _mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "values": arr.tolist()}


def summarize_experiment(exp_dir: Path, quiet: bool = False) -> dict:
    """Aggregate exp_dir/seed*/results.json into exp_dir/summary.json."""
    runs = sorted(exp_dir.glob("seed*/results.json"))
    if not runs:
        raise SystemExit(f"No seed*/results.json in {exp_dir}")
    results = [json.loads(p.read_text()) for p in runs]
    summary = {"experiment": exp_dir.name, "seeds": [p.parent.name for p in runs]}
    for split in ("val", "test"):
        summary[split] = {m: _mean_std([r[split][m] for r in results]) for m, _ in METRICS}
    clients = sorted(results[0]["val_per_client"], key=int)
    summary["val_per_client"] = {
        c: {m: _mean_std([r["val_per_client"][c][m] for r in results]) for m, _ in METRICS}
        for c in clients}
    # the worst client in each run, averaged: a simple fairness number
    summary["val_worst_client_accuracy"] = _mean_std(
        [min(r["val_per_client"][c]["accuracy"] for c in clients) for r in results])
    eps = [r["privacy"]["epsilon"] for r in results if "privacy" in r]
    if eps:
        summary["epsilon"] = _mean_std(eps)
    config_file = exp_dir / "experiment.json"
    if config_file.exists():
        summary["config"] = json.loads(config_file.read_text())
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(f"\n{exp_dir.name} ({len(runs)} seeds, mean +- std)")
        for split in ("val", "test"):
            row = ", ".join(f"{label} {fmt(summary[split][m])}" for m, label in METRICS)
            print(f"  {split:<5} {row}")
        print("  val per client: " + ", ".join(
            f"{c}: {fmt(summary['val_per_client'][c]['accuracy'])}" for c in clients))
    return summary


def fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def table(exp_dirs: list[Path], with_test: bool) -> str:
    header = ["Experiment", "Val acc.", "Val bal. acc.", "Val macro-F1", "Worst client val acc."]
    if with_test:
        header += ["Test acc.", "Test bal. acc.", "Test macro-F1"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for d in exp_dirs:
        s = summarize_experiment(d, quiet=True)
        row = [d.name] + [fmt(s["val"][m]) for m, _ in METRICS]
        row.append(fmt(s["val_worst_client_accuracy"]))
        if with_test:
            row += [fmt(s["test"][m]) for m, _ in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--with-test", action="store_true",
                        help="add the test columns (for the final report only)")
    args = parser.parse_args()
    dirs = [d for d in args.experiments if list(d.glob("seed*/results.json"))]
    print(table(dirs, args.with_test))


if __name__ == "__main__":
    main()

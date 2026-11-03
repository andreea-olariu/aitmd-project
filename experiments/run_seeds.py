"""Run one federated experiment with several seeds (default 1, 2, 3) and report
mean +- std. This is how every number in the tutorials was produced.

    python experiments/run_seeds.py --partitions data/partitions/dirichlet \\
        --config "num-server-rounds=10" --name fedavg

    python experiments/run_seeds.py --partitions data/partitions/dirichlet \\
        --config "num-server-rounds=10 loss-weighting='balanced'" --name fedavg-balanced

    python experiments/summarize.py outputs/experiments/fedavg outputs/experiments/fedavg-balanced

Each seed is a normal simulation run (scripts/run_simulation.sh with seed=<s>),
saved in outputs/experiments/<name>/seed<s>/ and evaluated on the clients'
validation data and on the test set (results.json). Seeds that already have a
results.json are skipped, so an interrupted batch can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import write_results  # noqa: E402
from summarize import summarize_experiment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a federated experiment with several seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--config", default="", help="run-config overrides, as for flwr run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", required=True, help="experiment name (output folder)")
    parser.add_argument("--force", action="store_true", help="re-run seeds that are done")
    args = parser.parse_args()
    if "seed=" in args.config.replace("-seed=", ""):
        parser.error("do not put seed=... in --config, use --seeds")

    partitions = args.partitions.resolve()
    exp_dir = REPO_ROOT / "outputs" / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.json").write_text(json.dumps(
        {"partitions": str(partitions), "config": args.config, "seeds": args.seeds}, indent=2))

    for seed in args.seeds:
        run_dir = exp_dir / f"seed{seed}"
        if (run_dir / "results.json").exists() and not args.force:
            print(f">>> {args.name} seed {seed}: done already, skipped")
            continue
        print(f">>> {args.name} seed {seed}", flush=True)
        log = exp_dir / f"seed{seed}.log"
        with open(log, "w") as out:
            subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "run_simulation.sh"), str(partitions),
                 f"{args.config} seed={seed}".strip()],
                stdout=out, stderr=subprocess.STDOUT, check=False,
                env={**os.environ, "SAVE_DIR": str(run_dir)})
        if not (run_dir / "metrics.json").exists():
            raise SystemExit(f"Run failed, see {log}")
        write_results(run_dir, partitions)

    summarize_experiment(exp_dir)


if __name__ == "__main__":
    main()

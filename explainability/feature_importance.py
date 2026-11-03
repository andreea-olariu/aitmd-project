"""Permutation feature importance for a model trained on feature vectors (e.g. HAR).

Idea: shuffle one feature (or a group of features, e.g. all statistics of one
joint) between the samples, which breaks its link with the label, and measure how
much the accuracy drops. Big drop = the model relies on it. Works for any model.

Examples (run from the repository root, with the venv activated):

    # Which joints matter? (HAR skeletons, default grouping)
    python explainability/feature_importance.py --run outputs/sim-har_camera-20261021-160935

    # Which statistics (mean / std / min / max)? On client 0's validation data
    python explainability/feature_importance.py --run outputs/<run> --group-by stat --data client_0

    # Sanity check: an UNTRAINED model should give importances close to 0
    python explainability/feature_importance.py --run outputs/<run> --untrained

Output: a bar chart (+ a map of the mean skeleton for --group-by joint), a JSON
file with all values, and the top features printed to the terminal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from aitdm_fl.task import load_analysis_data, model_from_test_set

STAT_PREFIX = re.compile(r"^(mean|std|min|max)_")
JOINT = re.compile(r"^(J\d+)[XYZ]$")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation feature importance for a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("model (choose one)")
    src.add_argument("--run", type=Path,
                     help="run folder from outputs/ (contains final_model.pt + metrics.json)")
    src.add_argument("--model", type=Path, help="path to a state_dict (.pt)")
    parser.add_argument("--partitions", type=Path,
                        help="partition folder (default: the one the run was trained on)")
    parser.add_argument("--data", default="val",
                        help="'val' = all clients' validation sets (default: diagnose here), "
                             "'client_<i>' = one client, 'test' = global test set (final "
                             "report only)")
    parser.add_argument("--group-by", choices=["joint", "column", "stat", "feature"],
                        help="shuffle features together per joint (J5 = all its X/Y/Z "
                             "statistics), per CSV column (J5X), per statistic (mean, std, "
                             "...) or one feature at a time. Default: joint if the columns "
                             "look like J1X ... J25Z, else column")
    parser.add_argument("--repeats", type=int, default=5,
                        help="shuffles per group (the result is their average)")
    parser.add_argument("--top", type=int, default=25, help="groups shown in the bar chart")
    parser.add_argument("--untrained", action="store_true",
                        help="sanity check: explain a randomly initialised model instead")
    parser.add_argument("--out", type=Path, help="output PNG (default: inside the run folder)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if (args.run is None) == (args.model is None):
        parser.error("use exactly one of --run or --model")
    if args.run is not None:
        args.model = args.run / "final_model.pt"
        if args.partitions is None:
            run_config = json.loads((args.run / "metrics.json").read_text())["run_config"]
            args.partitions = Path(run_config["partitions-dir"])
    if args.partitions is None:
        parser.error("--model needs --partitions")
    return args


def group_of(name: str, group_by: str) -> str:
    """mean_J5X -> J5 (joint) / J5X (column) / mean (stat) / mean_J5X (feature)."""
    column = STAT_PREFIX.sub("", name)
    if group_by == "column":
        return column
    if group_by == "stat":
        match = STAT_PREFIX.match(name)
        return match.group(1) if match else "value"
    if group_by == "joint":
        match = JOINT.match(column)
        return match.group(1) if match else column
    return name


@torch.no_grad()
def accuracy(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    return float((model(torch.from_numpy(x)).argmax(dim=1).numpy() == y).mean())


def plot_skeleton(ax, feature_names: list[str], x: np.ndarray, importance: dict[str, float]):
    """Mean pose of the data (front view), each joint coloured by its importance."""
    index = {name: i for i, name in enumerate(feature_names)}
    joints = [j for j in importance if f"mean_{j}X" in index and f"mean_{j}Y" in index]
    xs = [x[:, index[f"mean_{j}X"]].mean() for j in joints]
    ys = [x[:, index[f"mean_{j}Y"]].mean() for j in joints]
    values = np.array([importance[j] for j in joints])
    points = ax.scatter(xs, ys, c=values, s=60 + 600 * np.clip(values, 0, None) / max(
        values.max(), 1e-9), cmap="Reds", edgecolors="black", vmin=min(0.0, values.min()))
    for j, px, py in zip(joints, xs, ys, strict=True):
        ax.annotate(j, (px, py), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_aspect("equal")
    ax.set_title("Mean pose, coloured by importance")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(points, ax=ax, label="accuracy drop")


def main() -> None:
    args = get_arguments()
    meta = json.loads((args.partitions / "meta.json").read_text())
    if meta.get("data_type") != "features":
        raise SystemExit("This script is for feature data (e.g. HAR). "
                         "For images use explainability/gradcam.py.")
    feature_names = meta["feature_names"]
    if args.group_by is None:
        joint_like = any(JOINT.match(STAT_PREFIX.sub("", n)) for n in feature_names)
        args.group_by = "joint" if joint_like else "column"
    if args.out is None:
        folder = args.run if args.run is not None else Path("outputs")
        suffix = "_untrained" if args.untrained else ""
        args.out = folder / f"importance_{args.data}_{args.group_by}{suffix}.png"

    # 1. Model
    model = model_from_test_set(str(args.partitions))
    if not args.untrained:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # 2. Data
    x, y, _, tag = load_analysis_data(str(args.partitions), args.data)
    baseline = accuracy(model, x, y)
    print(f"{tag}: {len(y)} samples, accuracy {baseline:.3f}")

    # 3. Shuffle every group of features and measure the accuracy drop
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        groups.setdefault(group_of(name, args.group_by), []).append(i)
    rng = np.random.default_rng(args.seed)
    importance, spread = {}, {}
    for group, columns in groups.items():
        drops = []
        for _ in range(args.repeats):
            shuffled = x.copy()
            shuffled[:, columns] = x[rng.permutation(len(x))][:, columns]
            drops.append(baseline - accuracy(model, shuffled, y))
        importance[group] = float(np.mean(drops))
        spread[group] = float(np.std(drops))  # noise of the method itself
    ranking = sorted(importance, key=importance.get, reverse=True)

    print(f"\nMost important ({args.group_by}): accuracy drop when shuffled "
          f"(± std over {args.repeats} shuffles)")
    for group in ranking[:10]:
        print(f"  {group:<16} {importance[group]:+.3f} ± {spread[group]:.3f}")

    # 4. Plot
    shown = ranking[:args.top][::-1]
    with_skeleton = args.group_by == "joint"
    fig, axes = plt.subplots(1, 2 if with_skeleton else 1, squeeze=False,
                             figsize=(12 if with_skeleton else 7, max(3.0, 0.25 * len(shown) + 1)))
    ax = axes[0, 0]
    ax.barh(shown, [importance[g] for g in shown], color="tab:red")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("accuracy drop when shuffled (higher = more important)")
    ax.tick_params(axis="y", labelsize=8)
    if with_skeleton:
        plot_skeleton(axes[0, 1], feature_names, x, importance)
    title = f"Permutation importance per {args.group_by} | {args.data} | " \
            f"baseline accuracy {baseline:.3f}"
    if args.untrained:
        title = "UNTRAINED model (sanity check) | " + title
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"baseline_accuracy": baseline, "group_by": args.group_by, "data": args.data,
         "repeats": args.repeats,
         "importance": {g: importance[g] for g in ranking},
         "importance_std": {g: spread[g] for g in ranking}}, indent=2))
    print(f"\nSaved {args.out} (+ .json)")


if __name__ == "__main__":
    main()

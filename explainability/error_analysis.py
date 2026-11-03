"""Where does a trained model fail? Per-class metrics, confusion matrix and the
hardest examples (highest loss): step 2 of the iterative process in docs/05_gradcam.md.

Examples (run from the repository root, with the venv activated):

    # All clients' validation data (default), with a per-client breakdown
    python explainability/error_analysis.py --run outputs/sim-dirichlet-20261021-133800

    # The hardest TRAINING examples of client 1 (candidates for wrong labels / artefacts)
    python explainability/error_analysis.py --run outputs/<run> --data client_1 --split train

    # Any model / partition folder (e.g. your centralised baseline)
    python explainability/error_analysis.py --model my_model.pt --partitions data/partitions/iid

Output (in the run folder): errors_<data>.png (confusion matrix + per-class recall),
hardest_<data>.png (images only) and errors_<data>.json (all numbers, to compare runs).
Works for images and for feature data (e.g. HAR).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from aitdm_fl.task import load_analysis_data, model_from_test_set


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-class errors and hardest examples of a trained model.",
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
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="for client_<i>: its training or validation data")
    parser.add_argument("--num-images", type=int, default=18,
                        help="hardest examples to show / list")
    parser.add_argument("--out-dir", type=Path, help="default: the run folder")
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
    if args.out_dir is None:
        args.out_dir = args.run if args.run is not None else Path("outputs")
    return args


def main() -> None:
    args = get_arguments()
    meta = json.loads((args.partitions / "meta.json").read_text())
    class_names = meta["class_names"]
    num_classes = len(class_names)

    # 1. Data and predictions
    x, y, owner, tag = load_analysis_data(str(args.partitions), args.data, args.split)

    model = model_from_test_set(str(args.partitions))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(x[i:i + 512]))
                            for i in range(0, len(x), 512)])
    losses = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y),
                                               reduction="none").numpy()
    probs = logits.softmax(dim=1).numpy()
    preds = probs.argmax(axis=1)

    # 2. Per-class metrics
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, preds, labels=labels, zero_division=0)
    cm = confusion_matrix(y, preds, labels=labels)

    width = max(12, *(len(n[:28]) for n in class_names))
    print(f"\n{tag}: {len(y)} samples, accuracy {np.mean(preds == y):.3f}, "
          f"balanced accuracy {recall[support > 0].mean():.3f}\n")
    print(f"{'class':<{width}}  support  recall  precision     f1  most confused with")
    for c in labels:
        row = cm[c].copy()
        row[c] = 0
        confused = f"{class_names[row.argmax()][:28]} ({row.max()})" if row.max() else "-"
        flag = "  <-- never predicted correctly" if support[c] and recall[c] == 0 else ""
        print(f"{class_names[c][:28]:<{width}}  {support[c]:>7}  {recall[c]:>6.2f}  "
              f"{precision[c]:>9.2f}  {f1[c]:>5.2f}  {confused}{flag}")

    # 2b. Per client (pooled validation data): who does the global model fail?
    per_client = {}
    if owner is not None:
        print(f"\n{'client':<8} samples  accuracy  balanced acc.")
        for cid in np.unique(owner):
            m = owner == cid
            present = np.unique(y[m])
            rec = [np.mean(preds[m & (y == c)] == c) for c in present]
            per_client[int(cid)] = {"samples": int(m.sum()),
                                    "accuracy": float(np.mean(preds[m] == y[m])),
                                    "balanced_accuracy": float(np.mean(rec))}
            r = per_client[int(cid)]
            print(f"{cid:<8} {r['samples']:>7}  {r['accuracy']:>8.3f}  "
                  f"{r['balanced_accuracy']:>13.3f}")

    # 3. Hardest examples = highest loss (confidently wrong first)
    hardest = np.argsort(-losses)[:args.num_images]
    print(f"\nHardest {len(hardest)} examples (index: true -> predicted, loss):")
    for i in hardest[:10]:
        print(f"  #{i:<6} {class_names[y[i]][:24]:>24} -> {class_names[preds[i]][:24]:<24} "
              f"loss {losses[i]:.2f}")

    # 4. Plots: confusion matrix (row-normalised) + recall per class
    args.out_dir.mkdir(parents=True, exist_ok=True)
    short = [f"{c}: {n[:22]}" for c, n in enumerate(class_names)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * num_classes + 3.5),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axes[0].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j]:
                axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                             color="white" if norm[i, j] > 0.5 else "black")
    axes[0].set_xticks(labels, [str(c) for c in labels])
    axes[0].set_yticks(labels, short, fontsize=8)
    axes[0].set_xlabel("predicted class")
    axes[0].set_title("Confusion matrix (colour = fraction of the true class)")
    bars = axes[1].barh(short, recall,
                        color=["tab:red" if r < 0.3 else "tab:blue" for r in recall])
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].set_xlabel("recall (fraction of the class found)")
    axes[1].set_title("Recall per class (red < 0.3)")
    fig.suptitle(f"{args.model} | {tag}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"errors_{tag}.png", dpi=120)
    plt.close(fig)

    if x.ndim == 4:  # images: grid of the hardest examples
        cols = min(6, len(hardest))
        rows = math.ceil(len(hardest) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, hardest, strict=False):
            image = x[i].transpose(1, 2, 0)
            ax.imshow(image.squeeze(), cmap="gray" if image.shape[2] == 1 else None)
            ax.set_title(f"#{i} true: {class_names[y[i]][:14]}\npred: "
                         f"{class_names[preds[i]][:14]} ({probs[i, preds[i]]:.0%})",
                         fontsize=7, color="red" if preds[i] != y[i] else "green")
        fig.suptitle(f"Hardest examples (highest loss) | {tag}", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"hardest_{tag}.png", dpi=120)
        plt.close(fig)

    report = {
        "data": tag,
        "accuracy": float(np.mean(preds == y)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "per_class": {class_names[c]: {"support": int(support[c]), "recall": float(recall[c]),
                                       "precision": float(precision[c]), "f1": float(f1[c])}
                      for c in labels},
        "per_client": per_client,
        "confusion_matrix": cm.tolist(),
        "hardest": [{"index": int(i), "true": int(y[i]), "pred": int(preds[i]),
                     "loss": float(losses[i])} for i in hardest],
    }
    (args.out_dir / f"errors_{tag}.json").write_text(json.dumps(report, indent=2))
    saved = [f"errors_{tag}.png", f"errors_{tag}.json"]
    if x.ndim == 4:
        saved.insert(1, f"hardest_{tag}.png")
    print(f"\nSaved {', '.join(saved)} in {args.out_dir}")


if __name__ == "__main__":
    main()

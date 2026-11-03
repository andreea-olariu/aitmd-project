"""Grad-CAM explanations for a model trained with the Flower app.

Examples (run from the repository root, with the venv activated):

    # Explain 12 random validation images (all clients) of a finished run
    python explainability/gradcam.py --run outputs/sim-dirichlet-20261021-133800

    # Only the mistakes, on client 2's local validation data, with Grad-CAM++
    python explainability/gradcam.py --run outputs/<run> --data client_2 \
        --select mistakes --method gradcam++

    # Sanity check: the same images explained by an UNTRAINED model
    python explainability/gradcam.py --run outputs/<run> --untrained

    # Any model / partition folder (e.g. your centralised baseline)
    python explainability/gradcam.py --model my_model.pt --partitions data/partitions/iid

Output: one PNG grid (input image on top, heat map below, true / predicted class
in the title) + a short summary printed to the terminal.
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
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from aitdm_fl.task import load_analysis_data, model_from_test_set

METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "hirescam": HiResCAM,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM heat maps for a trained model.",
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
    parser.add_argument("--select", choices=["all", "mistakes", "correct"], default="all")
    parser.add_argument("--true-class",
                        help="only images of this class: its number or (part of) its name, "
                             "e.g. 4 or melanoma")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--method", choices=list(METHODS), default="gradcam")
    parser.add_argument("--layer", default="net.layer_1",
                        help="module to explain, e.g. net.layer_0 / net.layer_1 / net.layer_2 "
                             "(print(model) to see all names)")
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted",
                        help="explain the predicted class or the true class")
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
    if args.out is None:
        folder = args.run if args.run is not None else Path("outputs")
        suffix = "_untrained" if args.untrained else ""
        if args.true_class is not None:
            suffix += "_class-" + args.true_class.replace(" ", "-")
        args.out = folder / f"gradcam_{args.data}_{args.select}_{args.method}{suffix}.png"
    return args


def find_class(text: str, class_names: list[str] | None) -> int:
    """'4' -> 4; 'melanoma' -> index of the first class name containing it."""
    if text.isdigit():
        return int(text)
    matches = [i for i, n in enumerate(class_names or []) if text.lower() in n.lower()]
    if not matches:
        raise SystemExit(f"--true-class '{text}' matches no class: {class_names}")
    return matches[0]


def get_module(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def main() -> None:
    args = get_arguments()
    meta_path = args.partitions / "meta.json"
    class_names = json.loads(meta_path.read_text())["class_names"] if meta_path.exists() else None

    if meta_path.exists() and json.loads(meta_path.read_text()).get("data_type") == "features":
        raise SystemExit("Grad-CAM needs images. For feature data (e.g. HAR) use "
                         "explainability/feature_importance.py instead.")

    # 1. Model
    model = model_from_test_set(str(args.partitions))
    if not args.untrained:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # 2. Predictions on the chosen data
    x, y, _, _ = load_analysis_data(str(args.partitions), args.data)
    with torch.no_grad():
        preds = model(torch.from_numpy(x)).argmax(dim=1).numpy()
    correct = preds == y
    print(f"{args.data}: {len(y)} images, accuracy {correct.mean():.3f}")

    # 3. Pick the images to explain
    pool = {"all": np.arange(len(y)),
            "mistakes": np.where(~correct)[0],
            "correct": np.where(correct)[0]}[args.select]
    if args.true_class is not None:
        pool = pool[y[pool] == find_class(args.true_class, class_names)]
    if len(pool) == 0:
        raise SystemExit(f"No images match --select {args.select} / --true-class")
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(pool, size=min(args.num_images, len(pool)), replace=False))

    # 4. Grad-CAM
    target_layer = get_module(model, args.layer)
    explained = y[chosen] if args.target == "true" else preds[chosen]
    targets = [ClassifierOutputTarget(int(c)) for c in explained]
    with METHODS[args.method](model=model, target_layers=[target_layer]) as cam:
        heatmaps = cam(input_tensor=torch.from_numpy(x[chosen]), targets=targets)

    # 5. Plot: image (top) and heat map overlay (bottom) for every sample
    def name(c: int) -> str:
        return class_names[c][:18] if class_names else str(c)

    cols = min(6, len(chosen))
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(2 * rows, cols, figsize=(2.2 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (i, heatmap) in enumerate(zip(chosen, heatmaps, strict=True)):
        r, c = 2 * (k // cols), k % cols
        image = x[i].transpose(1, 2, 0)  # CHW -> HWC, values in [0, 1]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        overlay = show_cam_on_image(image.astype(np.float32), heatmap, use_rgb=True)
        axes[r, c].imshow(image, interpolation="nearest")
        colour = "green" if correct[i] else "red"
        axes[r, c].set_title(f"true: {name(y[i])}\npred: {name(preds[i])}",
                             fontsize=8, color=colour)
        axes[r + 1, c].imshow(overlay, interpolation="bilinear")

    title = f"{args.method} on {args.layer} | {args.data} | {args.select} | " \
            f"explaining the {args.target} class"
    if args.untrained:
        title = "UNTRAINED model (sanity check) | " + title
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {len(chosen)} explanations to {args.out}")


if __name__ == "__main__":
    main()

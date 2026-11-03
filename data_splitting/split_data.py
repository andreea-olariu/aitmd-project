"""Split a dataset into (biased) client partitions for federated learning.

Examples (run from the repository root, with the venv activated):

    # MedMNIST dataset, downloaded automatically, label skew with Dirichlet(0.3)
    python data_splitting/split_data.py --dataset dermamnist --num-clients 3 \
        --strategy dirichlet --alpha 0.3

    # Same, but every client also has a "different scanner" and client 2 a
    # sloppy annotator (20 % wrong labels)
    python data_splitting/split_data.py --dataset dermamnist --num-clients 3 \
        --strategy dirichlet --alpha 0.3 --feature-skew --label-noise 0,0,0.2

    # Your own CSV(s) (e.g. HAR skeletons, see docs/06_har.md): one client per camera
    python data_splitting/split_data.py --input data/raw/har/train_client*.csv \
        --label-col Action --sample-col IDSample --feature-cols '^J' \
        --strategy group --group-col Camera --num-clients 3

Outputs (in --out, default data/partitions/<strategy>):
    client_<i>.npz              one file per client (images or feature vectors),
                                read by the Flower app
    client_<i>_{train,val}.csv  [csv input] the same split as raw rows, for your own models
    test.npz / test.csv         held-out global test set
    meta.json                   settings + statistics
    label_distribution.png      plot of classes per client
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from partitioning import (
    apply_feature_skew,
    apply_label_noise,
    apply_sensor_skew,
    dirichlet_split,
    group_split,
    iid_split,
    pathological_split,
    quantity_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDMNIST_URL = "https://zenodo.org/records/10519652/files/{name}.npz?download=1"
# Official class names (from medmnist/info.py). ChestMNIST is left out on
# purpose: it is multi-label (14 findings per image), the app is single-label.
MEDMNIST_LABELS = {
    "pathmnist": ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
                  "normal colon mucosa", "cancer-associated stroma",
                  "colorectal adenocarcinoma epithelium"],
    "dermamnist": ["actinic keratoses and intraepithelial carcinoma", "basal cell carcinoma",
                   "benign keratosis-like lesions", "dermatofibroma", "melanoma",
                   "melanocytic nevi", "vascular lesions"],
    "octmnist": ["choroidal neovascularization", "diabetic macular edema", "drusen", "normal"],
    "pneumoniamnist": ["normal", "pneumonia"],
    "retinamnist": ["grade 0", "grade 1", "grade 2", "grade 3", "grade 4"],
    "breastmnist": ["malignant", "normal, benign"],
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "immature granulocytes",
                   "lymphocyte", "monocyte", "neutrophil", "platelet"],
    "tissuemnist": ["collecting duct, connecting tubule", "distal convoluted tubule",
                    "glomerular endothelial cells", "interstitial endothelial cells",
                    "leukocytes", "podocytes", "proximal tubule segments",
                    "thick ascending limb"],
    "organamnist": ["bladder", "femur-left", "femur-right", "heart", "kidney-left",
                    "kidney-right", "liver", "lung-left", "lung-right", "pancreas", "spleen"],
}
MEDMNIST_LABELS["organcmnist"] = MEDMNIST_LABELS["organamnist"]
MEDMNIST_LABELS["organsmnist"] = MEDMNIST_LABELS["organamnist"]
MEDMNIST_2D = list(MEDMNIST_LABELS)
STRATEGIES = ["iid", "dirichlet", "pathological", "quantity", "group"]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a dataset into biased federated client partitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("data source (choose one)")
    src.add_argument("--dataset", choices=MEDMNIST_2D,
                     help="MedMNIST 2D dataset to download into data/raw/")
    src.add_argument("--size", type=int, choices=[28, 64, 128, 224], default=28,
                     help="image size for --dataset (MedMNIST+). 64 / 128 / 224 give "
                          "sharper Grad-CAMs but are much bigger and slower")
    src.add_argument("--input", type=Path, nargs="+",
                     help="your own .npz file, or one or more .csv files (several CSVs "
                          "are concatenated; their name is stored in the column "
                          "'source_file', usable with --group-col)")

    own = parser.add_argument_group("options for --input")
    own.add_argument("--x-key", default="images", help="[npz] key holding the images")
    own.add_argument("--y-key", default="labels", help="[npz] key holding the labels")
    own.add_argument("--label-col", help="[csv] column holding the label")
    own.add_argument("--sample-col",
                     help="[csv] column identifying a sample, if a sample spans "
                          "several rows (e.g. frames of a video)")
    own.add_argument("--feature-cols",
                     help="[csv] regular expression selecting the feature columns, e.g. "
                          "'^J' for the joints J1X ... J25Z (default: every numeric column "
                          "except the label / sample / group columns)")
    own.add_argument("--class-names",
                     help="comma-separated class names, in the order of the sorted labels "
                          "(only used for plots and reports)")
    own.add_argument("--test-groups",
                     help="comma-separated values of --group-col held out COMPLETELY as the "
                          "global test set, e.g. --group-col Camera --test-groups 3: the "
                          "test set then measures generalisation to an unseen camera / "
                          "site / device (default: a random --test-fraction of the samples)")
    own.add_argument("--test-fraction", type=float, default=0.2,
                     help="fraction held out as global test set (when the data "
                          "has no official test split)")

    split = parser.add_argument_group("partitioning")
    split.add_argument("--num-clients", type=int, default=3)
    split.add_argument("--strategy", choices=STRATEGIES, default="dirichlet")
    split.add_argument("--alpha", type=float, default=0.5,
                       help="Dirichlet concentration (dirichlet / quantity). "
                            "Smaller = more biased")
    split.add_argument("--classes-per-client", type=int, default=2,
                       help="classes owned by each client (pathological)")
    split.add_argument("--group-col",
                       help="column (csv) or npz key whose value defines the "
                            "client (group strategy)")
    split.add_argument("--min-samples", type=int, default=20,
                       help="minimum samples per client (dirichlet / quantity)")
    split.add_argument("--val-fraction", type=float, default=0.15,
                       help="fraction of each client's data kept for local validation")

    bias = parser.add_argument_group("extra real-world bias")
    bias.add_argument("--feature-skew", action="store_true",
                      help="different device per client: brightness / noise (images) "
                           "or gain / offset / noise per column (csv)")
    bias.add_argument("--label-noise", default="",
                      help="comma-separated label noise rate per client, "
                           "e.g. '0,0,0.2'")

    parser.add_argument("--out", type=Path,
                        help="output folder (default: data/partitions/<strategy>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.dataset is None) == (args.input is None):
        parser.error("use exactly one of --dataset or --input")
    if args.input is not None:
        suffixes = {path.suffix.lower() for path in args.input}
        if suffixes - {".csv", ".npz"} or len(suffixes) > 1:
            parser.error("--input takes either one .npz file or one or more .csv files")
        if suffixes == {".npz"} and len(args.input) > 1:
            parser.error("--input takes only one .npz file")
    if args.strategy == "group" and not args.group_col:
        parser.error("--strategy group needs --group-col")
    if args.test_groups and args.dataset:
        parser.error("--test-groups is for --input data; MedMNIST has an official test split")
    if args.out is None:
        args.out = REPO_ROOT / "data" / "partitions" / args.strategy
    return args


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def download_medmnist(name: str, size: int = 28) -> Path:
    if size != 28:  # MedMNIST+ larger versions are stored as <name>_<size>.npz
        name = f"{name}_{size}"
    path = REPO_ROOT / "data" / "raw" / f"{name}.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = MEDMNIST_URL.format(name=name)
    print(f"Downloading {name} from {url} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def load_npz(path: Path, args) -> dict:
    """Return dict with pool (to be split) and test arrays + optional groups."""
    data = np.load(path)
    if "train_images" in data:  # MedMNIST layout: train / val / test
        x_pool = np.concatenate([data["train_images"], data["val_images"]])
        y_pool = np.concatenate([data["train_labels"], data["val_labels"]]).reshape(-1)
        x_test, y_test = data["test_images"], data["test_labels"].reshape(-1)
        groups = None
    else:
        x_all, y_all = data[args.x_key], data[args.y_key].reshape(-1)
        groups_all = data[args.group_col].reshape(-1) if args.group_col else None
        test_mask = make_test_mask(len(y_all), groups_all, args)
        x_pool, y_pool = x_all[~test_mask], y_all[~test_mask]
        x_test, y_test = x_all[test_mask], y_all[test_mask]
        groups = groups_all[~test_mask] if groups_all is not None else None
    return {"x": x_pool, "y": y_pool.astype(np.int64), "groups": groups,
            "x_test": x_test, "y_test": y_test.astype(np.int64)}


def load_csv(paths: list[Path], args):
    """Return (dataframe, sample column, per-sample ids, labels, groups, test mask)."""
    if not args.label_col:
        sys.exit("ERROR: --label-col is required for CSV input")
    frames = []
    for path in paths:
        print(f"Reading {path} ...")
        frame = pd.read_csv(path)
        for col in (args.label_col, args.sample_col):
            if col and col not in frame.columns:
                sys.exit(f"ERROR: column '{col}' not in {path}. Columns: {list(frame.columns)}")
        if len(paths) > 1:
            frame["source_file"] = path.stem
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if args.group_col and args.group_col not in df.columns:
        sys.exit(f"ERROR: column '{args.group_col}' not found. Columns: {list(df.columns)}")

    sample_col = args.sample_col or "__row__"
    if sample_col == "__row__":
        df[sample_col] = np.arange(len(df))
    elif len(paths) > 1 and df.groupby(sample_col)["source_file"].nunique().max() > 1:
        # The same sample id appears in several files: make the ids unique
        df[sample_col] = df["source_file"] + ":" + df[sample_col].astype(str)

    # One entry per sample (a sample may span several rows)
    per_sample = df.groupby(sample_col, sort=True).first()
    sample_ids = per_sample.index.to_numpy()
    labels = per_sample[args.label_col].to_numpy()
    groups = per_sample[args.group_col].to_numpy() if args.group_col else None

    test_mask = make_test_mask(len(sample_ids), groups, args)
    return df, sample_col, sample_ids, labels, groups, test_mask


def make_test_mask(n: int, groups, args) -> np.ndarray:
    """Which samples form the global test set: a random fraction, or (with
    --test-groups) every sample of the given groups, which no client then sees."""
    if not args.test_groups:
        return np.random.default_rng(args.seed).random(n) < args.test_fraction
    if groups is None:
        sys.exit("ERROR: --test-groups needs --group-col")
    wanted = {v.strip() for v in args.test_groups.split(",")}
    as_text = np.array([str(g) for g in groups])
    mask = np.isin(as_text, list(wanted))
    if not mask.any() or mask.all():
        sys.exit(f"ERROR: --test-groups {args.test_groups} must select some but not all "
                 f"groups. Values of '{args.group_col}': {sorted(set(as_text.tolist()))}")
    print(f"Test set = every sample of {args.group_col} in {sorted(wanted)} "
          f"({mask.sum()} samples); the clients never see these groups")
    return mask


def select_feature_cols(df, args, sample_col: str) -> list[str]:
    special = {args.label_col, sample_col, args.group_col, "source_file"}
    if args.feature_cols:
        cols = [c for c in df.columns if re.search(args.feature_cols, c) and c not in special]
    else:
        cols = [c for c in df.select_dtypes("number").columns if c not in special]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if not cols or non_numeric:
        sys.exit(f"ERROR: --feature-cols must select numeric columns (selected: {cols})")
    preview = ", ".join(cols[:6]) + (", ..." if len(cols) > 6 else "")
    print(f"Feature columns ({len(cols)}): {preview}")
    return cols


def sample_features(rows, sample_col: str, feature_cols: list[str], ids: np.ndarray,
                    multi_row: bool) -> tuple[np.ndarray, list[str]]:
    """One fixed-length feature vector per sample, in the order of `ids`.

    If a sample spans several rows (e.g. the frames of a skeleton sequence), its
    rows are summarised by mean, standard deviation, min and max of every column,
    so sequences of any length give vectors of the same size (4 x #columns).
    Otherwise the row itself is the feature vector."""
    grouped = rows.groupby(sample_col, sort=True)[feature_cols]
    if multi_row:
        stats = {"mean": grouped.mean(), "std": grouped.std(ddof=0),
                 "min": grouped.min(), "max": grouped.max()}
        table = pd.concat(stats, axis=1)
        names = [f"{stat}_{col}" for stat, col in table.columns]
    else:
        table = grouped.first()
        names = list(feature_cols)
    return table.loc[ids].to_numpy(np.float32), names


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def partition(labels: np.ndarray, groups, args) -> list[np.ndarray]:
    n = args.num_clients
    if args.strategy == "iid":
        return iid_split(labels, n, seed=args.seed)
    if args.strategy == "dirichlet":
        return dirichlet_split(labels, n, alpha=args.alpha,
                               min_samples=args.min_samples, seed=args.seed)
    if args.strategy == "pathological":
        return pathological_split(labels, n, classes_per_client=args.classes_per_client,
                                  seed=args.seed)
    if args.strategy == "quantity":
        return quantity_split(labels, n, alpha=args.alpha,
                              min_samples=args.min_samples, seed=args.seed)
    if groups is None:
        sys.exit(f"ERROR: group column '{args.group_col}' not found in the data")
    return group_split(groups, n, seed=args.seed)


def warn_missing_classes(parts, labels, num_classes: int, names: list[str], args) -> None:
    """A class that no client trains on can never be predicted, but it is still in
    the test set. Say so loudly instead of silently reporting a bad score."""
    seen = set(np.unique(np.concatenate([labels[idx] for idx in parts]))) if parts else set()
    missing = [c for c in range(num_classes) if c not in seen]
    if not missing:
        return
    print("\nWARNING: no client has training data for class(es) "
          + ", ".join(f"{c} ({names[c]})" for c in missing)
          + ". The model cannot learn them, but they are still in the test set.")
    if args.strategy == "pathological" and args.num_clients * args.classes_per_client < num_classes:
        print(f"         With --strategy pathological, num-clients x classes-per-client "
              f"({args.num_clients} x {args.classes_per_client}) must be >= the number of "
              f"classes ({num_classes}) to cover them all.")
    print()


def train_val_split(indices: np.ndarray, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    n_val = int(round(len(shuffled) * val_fraction))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def parse_label_noise(text: str, num_clients: int) -> list[float]:
    if not text:
        return [0.0] * num_clients
    rates = [float(v) for v in text.split(",")]
    if len(rates) != num_clients:
        sys.exit(f"ERROR: --label-noise needs {num_clients} comma-separated values")
    return rates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def class_counts(labels: np.ndarray, classes: np.ndarray) -> list[int]:
    return [int((labels == c).sum()) for c in classes]


def print_report(counts: np.ndarray, class_names: list[str]) -> None:
    width = max(7, *(len(str(c)) for c in class_names))
    header = "client".ljust(8) + "".join(str(c).rjust(width + 1) for c in class_names)
    print("\nTraining samples per class and client")
    print(header + "    total  majority%")
    print("-" * (len(header) + 19))
    for i, row in enumerate(counts):
        total = row.sum()
        majority = 100 * row.max() / total if total else 0
        print(f"{i:<8}" + "".join(f"{v:>{width + 1}}" for v in row)
              + f"{total:>9}{majority:>10.1f}")
    print()


def plot_distribution(counts: np.ndarray, class_names: list[str], title: str, path: Path):
    num_clients, num_classes = counts.shape
    legend_rows = math.ceil(num_classes / 3)
    height = 0.6 * num_clients + 2.5 + 0.25 * legend_rows
    fig, axes = plt.subplots(1, 2, figsize=(12, height))

    # Left: absolute counts (shows quantity skew)
    left = np.zeros(num_clients)
    cmap = plt.get_cmap("tab20" if num_classes > 10 else "tab10")
    for c in range(num_classes):
        axes[0].barh(range(num_clients), counts[:, c], left=left,
                     color=cmap(c % cmap.N), label=f"{c}: {class_names[c][:28]}")
        left += counts[:, c]
    axes[0].set_title("Samples per client")
    axes[0].set_xlabel("# training samples")

    # Right: proportions (shows label skew)
    props = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    left = np.zeros(num_clients)
    for c in range(num_classes):
        axes[1].barh(range(num_clients), props[:, c], left=left, color=cmap(c % cmap.N))
        left += props[:, c]
    axes[1].set_title("Label distribution per client")
    axes[1].set_xlabel("fraction of client's data")
    axes[1].set_xlim(0, 1)

    for ax in axes:
        ax.set_yticks(range(num_clients), [f"client {i}" for i in range(num_clients)])
        ax.invert_yaxis()
    fig.legend(loc="lower center", ncol=3, fontsize="small", frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, (0.25 * legend_rows + 0.2) / height, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return str(value) if isinstance(value, Path) else value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = get_arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    noise_rates = parse_label_noise(args.label_noise, args.num_clients)
    is_csv = args.input is not None and args.input[0].suffix.lower() == ".csv"

    if is_csv:
        df, sample_col, sample_ids, raw_labels, groups, test_mask = load_csv(args.input, args)
        feature_cols = select_feature_cols(df, args, sample_col)
        multi_row = bool(df.groupby(sample_col).size().max() > 1)
        class_names, labels = np.unique(raw_labels, return_inverse=True)
        pool_ids, pool_labels = sample_ids[~test_mask], labels[~test_mask]
        pool_groups = groups[~test_mask] if groups is not None else None
        test_ids = sample_ids[test_mask]
    else:
        path = download_medmnist(args.dataset, args.size) if args.dataset else args.input[0]
        data = load_npz(path, args)
        class_names = np.unique(np.concatenate([data["y"], data["y_test"]]))
        pool_labels, pool_groups = data["y"], data["groups"]

    num_classes = len(class_names)
    if args.dataset:
        label_names = MEDMNIST_LABELS[args.dataset]
    elif args.class_names:
        label_names = [name.strip() for name in args.class_names.split(",")]
        if len(label_names) != num_classes:
            sys.exit(f"ERROR: --class-names has {len(label_names)} names, "
                     f"the data has {num_classes} classes: {list(class_names)}")
    else:
        label_names = [str(c) for c in class_names]
    parts = partition(pool_labels, pool_groups, args)
    warn_missing_classes(parts, pool_labels, num_classes, label_names, args)

    counts, client_info = [], []
    for cid, indices in enumerate(parts):
        train_idx, val_idx = train_val_split(indices, args.val_fraction, args.seed + cid)
        noisy_train = apply_label_noise(pool_labels[train_idx], noise_rates[cid],
                                        num_classes, seed=args.seed + cid)

        if is_csv:
            # Map the (possibly noisy) encoded labels back to the original names
            train_rows = df[df[sample_col].isin(pool_ids[train_idx])].copy()
            relabel = dict(zip(pool_ids[train_idx], class_names[noisy_train]))
            train_rows[args.label_col] = train_rows[sample_col].map(relabel)
            val_rows = df[df[sample_col].isin(pool_ids[val_idx])].copy()
            if args.feature_skew:  # the same "device" records train and val data
                client_rows = pd.concat([train_rows, val_rows])
                skewed = apply_sensor_skew(client_rows[feature_cols].to_numpy(np.float64),
                                           cid, args.num_clients, args.seed)
                train_rows[feature_cols] = skewed[:len(train_rows)]
                val_rows[feature_cols] = skewed[len(train_rows):]
            for name, rows in (("train", train_rows), ("val", val_rows)):
                rows = rows.drop(columns="__row__", errors="ignore")
                rows.to_csv(args.out / f"client_{cid}_{name}.csv", index=False)
            # The same split as feature vectors, for the Flower app
            x_train, feature_names = sample_features(train_rows, sample_col, feature_cols,
                                                     pool_ids[train_idx], multi_row)
            x_val, _ = sample_features(val_rows, sample_col, feature_cols,
                                       pool_ids[val_idx], multi_row)
            np.savez_compressed(
                args.out / f"client_{cid}.npz",
                x_train=x_train, y_train=noisy_train,
                x_val=x_val, y_val=pool_labels[val_idx],
            )
        else:
            x_train, x_val = data["x"][train_idx], data["x"][val_idx]
            if args.feature_skew:
                x_train = apply_feature_skew(x_train, cid, args.num_clients, args.seed)
                x_val = apply_feature_skew(x_val, cid, args.num_clients, args.seed)
            np.savez_compressed(
                args.out / f"client_{cid}.npz",
                x_train=x_train, y_train=noisy_train,
                x_val=x_val, y_val=pool_labels[val_idx],
            )

        counts.append(class_counts(noisy_train, np.arange(num_classes)))
        client_info.append({
            "client": cid, "train": int(len(train_idx)), "val": int(len(val_idx)),
            "label_noise": noise_rates[cid],
            "train_class_counts": counts[-1],
        })

    # Global, untouched test set (used by the server for centralised evaluation)
    if is_csv:
        test_rows = df[df[sample_col].isin(test_ids)]
        test_rows.drop(columns="__row__", errors="ignore").to_csv(args.out / "test.csv",
                                                                  index=False)
        x_test, feature_names = sample_features(test_rows, sample_col, feature_cols,
                                                test_ids, multi_row)
        np.savez_compressed(args.out / "test.npz", x_test=x_test, y_test=labels[test_mask])
        n_test = len(test_ids)
        sample_shape = [x_test.shape[1]]
    else:
        np.savez_compressed(args.out / "test.npz", x_test=data["x_test"], y_test=data["y_test"])
        n_test = len(data["y_test"])
        sample_shape = list(data["x"].shape[1:])

    if args.dataset:
        source = f"{args.dataset} ({args.size}px)"
    else:
        source = ", ".join(path.name for path in args.input)
    meta = {
        "source": source,
        "data_type": "features" if is_csv else "images",
        "strategy": args.strategy,
        "num_clients": args.num_clients,
        "num_classes": int(num_classes),
        "class_names": label_names,
        "sample_shape": sample_shape,
        "feature_names": feature_names if is_csv else None,
        "test_samples": int(n_test),
        "settings": {k: json_safe(v) for k, v in vars(args).items()},
        "clients": client_info,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    counts = np.array(counts)
    print_report(counts, [str(c) for c in class_names])
    title = f"{meta['source']} | strategy={args.strategy}"
    if args.strategy in ("dirichlet", "quantity"):
        title += f" (alpha={args.alpha})"
    plot_distribution(counts, label_names, title,
                      args.out / "label_distribution.png")
    print(f"Saved {args.num_clients} client partitions + global test set to {args.out}")
    print(f"Plot: {args.out / 'label_distribution.png'}")


if __name__ == "__main__":
    main()

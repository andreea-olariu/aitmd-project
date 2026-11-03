"""Model, data loading, training and evaluation – everything that is *not*
Flower-specific lives here, so you can reuse it for your centralised baseline.
"""

from __future__ import annotations

import json
import random
import warnings
import zlib
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import Classifier
from monai.transforms import Compose, RandFlip, RandRotate
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(sample_shape: tuple[int, ...], num_classes: int) -> torch.nn.Module:
    """Pick a model that fits the data:

    * images (C, H, W)      -> a small CNN from MONAI. Swap it for any MONAI /
      torchvision network (e.g. monai.networks.nets.DenseNet121) once the pipeline works.
    * feature vectors (F,)  -> an MLP (HAR skeletons, sensors, tabular data).
    """
    if len(sample_shape) == 1:
        return build_mlp(sample_shape[0], num_classes)
    in_channels, image_size = sample_shape[0], sample_shape[-1]
    return Classifier(
        in_shape=(in_channels, image_size, image_size),
        classes=num_classes,
        channels=(16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
    )


def build_mlp(num_features: int, num_classes: int) -> torch.nn.Module:
    """Feed-forward network with 2 hidden layers and dropout.
    Inputs are used as they are: fine for skeleton coordinates in metres. If your
    features have very different scales, standardise them first."""
    return torch.nn.Sequential(
        torch.nn.Linear(num_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, num_classes),
    )


def model_from_test_set(partitions_dir: str) -> torch.nn.Module:
    """Build the model with the shape / number of classes of the dataset.

    For the SERVER and for analysis scripts only: it reads the global test set.
    Clients never do; they use build_model() with the shape of their own data
    and the number of classes sent by the server (see client_app.py)."""
    x, y = load_npz_arrays(Path(partitions_dir) / "test.npz", "test")
    return build_model(sample_shape=x.shape[1:], num_classes=num_classes_of_dataset(partitions_dir))


def num_classes_of_dataset(partitions_dir: str) -> int:
    """Number of classes of the whole dataset (meta.json, written by split_data.py)."""
    meta = Path(partitions_dir) / "meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["num_classes"])
    _, y = load_npz_arrays(Path(partitions_dir) / "test.npz", "test")
    return int(y.max()) + 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (model initialisation, dropout, shuffling)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def derive_seed(seed: int, *parts) -> int:
    """A different but reproducible seed for every (client, round, ...) combination,
    so that clients do not all shuffle their data in the same order."""
    text = ":".join(str(p) for p in (seed, *parts))
    return zlib.crc32(text.encode())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_npz_arrays(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load x_<split>, y_<split>.

    Images are returned as (N, C, H, W) float32 in [0, 1], feature vectors
    (written by split_data.py for CSV data) as (N, F) float32, unchanged."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data_splitting/split_data.py first and pass the "
            "ABSOLUTE output folder with --run-config \"partitions-dir='...'\""
        )
    data = np.load(path)
    x, y = data[f"x_{split}"], data[f"y_{split}"].reshape(-1)
    if x.ndim == 2:  # (N, F) feature vectors
        return x.astype(np.float32), y.astype(np.int64)
    x = x.astype(np.float32) / 255.0
    if x.ndim == 3:  # (N, H, W) grayscale -> (N, 1, H, W)
        x = x[:, None]
    elif x.ndim == 4 and x.shape[-1] in (1, 3):  # (N, H, W, C) -> (N, C, H, W)
        x = x.transpose(0, 3, 1, 2)
    return x, y.astype(np.int64)


class ArrayDataset(Dataset):
    """Images or feature vectors. Augmentation (flip / rotate) is for images only."""

    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False,
                 seed: int | None = None):
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)
        self.transform = (
            Compose([RandFlip(prob=0.5, spatial_axis=1), RandRotate(range_x=0.26, prob=0.5)])
            if augment else None
        )
        if self.transform is not None and seed is not None:
            self.transform.set_random_state(seed=seed % 2**32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        image = self.x[i]
        if self.transform is not None:
            image = torch.as_tensor(self.transform(image))
        return image, self.y[i]


def load_client_data(data_path: Path, batch_size: int,
                     seed: int | None = None) -> tuple[DataLoader, DataLoader]:
    """Return train / validation loaders of ONE client.
    With a seed, shuffling and augmentation are reproducible."""
    x_tr, y_tr = load_npz_arrays(data_path, "train")
    x_va, y_va = load_npz_arrays(data_path, "val")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(ArrayDataset(x_tr, y_tr, augment=x_tr.ndim == 4, seed=seed),
                              batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(ArrayDataset(x_va, y_va), batch_size=batch_size)
    return train_loader, val_loader


def load_test_data(partitions_dir: str, batch_size: int = 256) -> DataLoader:
    """Global held-out test set, used by the server (centralised evaluation).
    Use it ONCE, for the final numbers; take decisions on validation data."""
    x, y = load_npz_arrays(Path(partitions_dir) / "test.npz", "test")
    return DataLoader(ArrayDataset(x, y), batch_size=batch_size)


def load_pooled_validation(partitions_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All clients' validation sets together: (x, y, client id of each sample).
    The data to analyse and compare models on while you develop them."""
    xs, ys, owners = [], [], []
    files = sorted(Path(partitions_dir).glob("client_*.npz"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for path in files:
        x, y = load_npz_arrays(path, "val")
        xs.append(x)
        ys.append(y)
        owners.append(np.full(len(y), int(path.stem.split("_")[1])))
    if not xs:
        raise FileNotFoundError(f"No client_<i>.npz files in {partitions_dir}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(owners)


TEST_SET_REMINDER = (
    "NOTE: you are looking at the TEST set. Use it once, for the final numbers of a "
    "model you have already chosen. Diagnose and compare on --data val instead.")


def load_analysis_data(partitions_dir: str, which: str = "val", split: str = "val"):
    """Data for the analysis scripts in explainability/. Returns (x, y, owner, tag).

    * "val"        all clients' validation sets together (default; owner = client id)
    * "client_<i>" one client's data (its "val" or "train" split, see `split`)
    * "test"       the global test set: only for the final report
    """
    partitions = Path(partitions_dir)
    if which == "val":
        x, y, owner = load_pooled_validation(partitions_dir)
        return x, y, owner, "val"
    if which.startswith("client_"):
        x, y = load_npz_arrays(partitions / f"{which}.npz", split)
        return x, y, None, f"{which}_{split}"
    if which == "test":
        print(TEST_SET_REMINDER)
        x, y = load_npz_arrays(partitions / "test.npz", "test")
        return x, y, None, "test"
    raise ValueError("--data must be 'val', 'client_<i>' or 'test'")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Loss weights inversely proportional to class frequency ("balanced").

    A class with 10x fewer samples gets a 10x larger weight, so mistakes on rare
    classes cost as much as mistakes on common ones. Computed by every client from
    its OWN labels: no data or statistics leave the client. Classes the client
    does not have get weight 1 (they never occur in its loss anyway)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(num_classes)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def train(model, loader, epochs: int, lr: float, device, proximal_mu: float = 0.0,
          class_weights: torch.Tensor | None = None) -> float:
    """Train locally and return the average training loss.

    With proximal_mu > 0 this is FedProx: a penalty keeps the local model close
    to the global model received from the server (helps with non-IID data).
    class_weights (see balanced_class_weights) re-weights the loss per class."""
    model.to(device)
    model.train()
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_params = [p.detach().clone() for p in model.parameters()]

    total_loss, total_batches = 0.0, 0
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Return loss, accuracy, balanced accuracy and macro-F1.
    Balanced accuracy / macro-F1 matter with imbalanced (medical) data."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss, y_true, y_pred = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item()
        y_true.append(labels.cpu().numpy())
        y_pred.append(outputs.argmax(dim=1).cpu().numpy())

    n = sum(len(t) for t in y_true)
    if n == 0:
        return {"loss": 0.0, "accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0}
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    with warnings.catch_warnings():
        # A biased client may not have every class in its validation set
        warnings.simplefilter("ignore", UserWarning)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "loss": total_loss / n,
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_acc),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

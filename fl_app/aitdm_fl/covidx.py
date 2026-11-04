"""COVIDx CXR-4: dataset, model, training loop and metrics. No Flower code here.

Used by experiments/train_covidx.py (centralised baseline), and meant to be used by the
Flower client as well, so that both sides train and measure in exactly the same way.

Data: <data_dir>/manifest.csv and the PNGs written by data_splitting/preprocess_covidx.py.
The dataset yields 1-channel images in [0, 1]. The model adapts them to its ImageNet
weights itself (1 -> 3 channels + ImageNet normalisation), so Grad-CAM, FL clients and
analysis scripts can feed it plain images.

Labels: 0 = negative, 1 = positive. The model has two outputs (softmax). Evaluation
scores images by the log-odds logit[1] - logit[0] (see predict_log_odds), so
P(positive) = sigmoid(log-odds) and the log-odds threshold 0.0 means P = 0.5.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2

LABEL_NAMES = ("negative", "positive")
ARCHITECTURES = ("resnet18", "resnet50", "densenet121")
SAMPLERS = ("none", "class", "within-source")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    path: str        # relative to the data dir, e.g. "train/abc.png"
    label: int       # 0 negative, 1 positive
    source: str      # bimcv, rsna, stonybrook, ...
    patient_id: str


def read_manifest(data_dir: Path, split: str) -> list[Record]:
    """All images of one split ("train", "val" or "test"), in manifest order."""
    manifest = Path(data_dir) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found: run data_splitting/preprocess_covidx.py")
    with manifest.open(newline="") as f:
        return [Record(r["path"], int(r["label"]), r["source"], r["patient_id"])
                for r in csv.DictReader(f) if r["split"] == split]


def selection_indices(records: list[Record], seed: int = 0) -> list[int]:
    """Indices of the model-selection slice of the validation set: every bimcv negative
    and as many randomly drawn bimcv positives.

    It is built like the test set (bimcv, 50 / 50), so recognising the source does not
    help on it, unlike the full validation set where the source predicts the label.
    The seed is fixed (not the training seed) so every run is selected on the same images.
    """
    neg = [i for i, r in enumerate(records) if r.source == "bimcv" and r.label == 0]
    pos = [i for i, r in enumerate(records) if r.source == "bimcv" and r.label == 1]
    rng = np.random.default_rng(seed)
    pos = rng.choice(pos, size=min(len(neg), len(pos)), replace=False).tolist()
    return sorted(neg + pos)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transforms(train: bool, image_size: int = 224) -> v2.Compose:
    """PIL grayscale image -> float tensor (1, image_size, image_size) in [0, 1].

    Training augmentation is mild and anatomically plausible. No horizontal flip: the
    heart is on the left, and a flipped X-ray is not a realistic image."""
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if not train:
        return v2.Compose([*to_tensor, v2.Resize(image_size, antialias=True)])
    return v2.Compose([
        *to_tensor,
        v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])


class CovidxDataset(Dataset):
    """Images of a list of manifest records. Returns (image, label) like any classifier
    dataset; the source of sample i is records[i].source."""

    def __init__(self, data_dir: Path, records: list[Record], transform: v2.Compose):
        self.data_dir = Path(data_dir)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        record = self.records[i]
        with Image.open(self.data_dir / record.path) as img:
            image = self.transform(img.convert("L"))
        return image, record.label


def sample_weights(records: list[Record], scheme: str) -> np.ndarray | None:
    """Per-sample weights for a WeightedRandomSampler.

    * "none":          plain shuffling, the data as it is
    * "class":         both labels equally often (fixes the 84 % positive prior only)
    * "within-source": inside every source both labels equally often, while each source
                       keeps its share of the data. Sources with a single label are
                       unchanged; bimcv's few negatives are drawn as often as its
                       positives. Targets the source -> label shortcut.
    """
    if scheme == "none":
        return None
    if scheme == "class":
        counts = Counter(r.label for r in records)
        return np.array([1.0 / counts[r.label] for r in records])
    if scheme == "within-source":
        cell = Counter((r.source, r.label) for r in records)
        per_source = Counter(r.source for r in records)
        labels_in_source = Counter(source for source, _ in cell)
        return np.array([per_source[r.source] / (labels_in_source[r.source] * cell[(r.source, r.label)])
                         for r in records])
    raise ValueError(f"unknown sampler {scheme!r}, choose from {SAMPLERS}")


def make_loader(data_dir: Path, records: list[Record], train: bool, batch_size: int,
                num_workers: int = 4, seed: int | None = None, sampler: str = "none",
                image_size: int = 224) -> DataLoader:
    """Training loader (augmented, shuffled or re-weighted) or evaluation loader
    (deterministic, manifest order, so predictions line up with `records`)."""
    dataset = CovidxDataset(data_dir, records, build_transforms(train, image_size))
    common = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available(), "persistent_workers": num_workers > 0}
    if not train:
        return DataLoader(dataset, shuffle=False, **common)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    drop_last = len(records) > batch_size  # no size-1 batches for BatchNorm
    weights = sample_weights(records, sampler)
    if weights is None:
        return DataLoader(dataset, shuffle=True, generator=generator, drop_last=drop_last, **common)
    weighted = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                     num_samples=len(records), replacement=True,
                                     generator=generator)
    return DataLoader(dataset, sampler=weighted, drop_last=drop_last, **common)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GrayscaleToImageNet(nn.Module):
    """(N, 1, H, W) in [0, 1] -> (N, 3, H, W) normalised like ImageNet images.
    The constants are non-persistent buffers: not in the state_dict, not federated."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std


def build_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """torchvision CNN with ImageNet weights and a new `num_classes` head.

    The first call with pretrained=True downloads the weights to the torch hub cache
    (~/.cache/torch/hub/checkpoints). Grad-CAM target layers: "backbone.layer4"
    (ResNets), "backbone.features.denseblock4" (DenseNet)."""
    if arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "densenet121":
        net = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture {arch!r}, choose from {ARCHITECTURES}")
    return nn.Sequential(OrderedDict(adapter=GrayscaleToImageNet(), backbone=net))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; no weight decay on biases and normalisation parameters (the usual recipe)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def warmup_cosine(optimizer: torch.optim.Optimizer, total_steps: int,
                  warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the base lr, then cosine decay to 0. Stepped once per batch."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, scheduler=None, scaler: torch.amp.GradScaler | None = None,
                    class_weights: torch.Tensor | None = None, amp: bool = True) -> float:
    """One pass over `loader`; returns the mean training loss.
    amp: mixed precision (float16) on CUDA, ignored on CPU."""
    model.to(device)
    model.train()
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    use_amp = amp and device.type == "cuda"
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(images), labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
        total += len(labels)
    return total_loss / max(total, 1)


@torch.no_grad()
def predict_log_odds(model: nn.Module, loader: DataLoader,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(labels, log-odds of positive) for every sample of an evaluation loader, in order.

    The score is logit[positive] - logit[negative] = log(P(pos) / P(neg)), and
    P(positive) = sigmoid(score). Rank, threshold and compare on this score, not on the
    probability: an overconfident model pushes many probabilities to exactly 1.0 in
    float32 (any score above ~17), which creates ties that blur AUROC and make the
    chosen threshold unstable. Evaluation runs in float32 (no autocast) for the same
    reason; it is a small share of the run time."""
    model.to(device)
    model.eval()
    labels, scores = [], []
    for images, y in loader:
        logits = model(images.to(device, non_blocking=True)).float()
        scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores).astype(np.float64)


def sigmoid(score: np.ndarray) -> np.ndarray:
    """Log-odds -> P(positive), without overflow warnings."""
    return expit(score)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def best_threshold(y: np.ndarray, score: np.ndarray, default: float = 0.0) -> float:
    """Decision threshold with the highest balanced accuracy (Youden's J) on (y, score).
    Works on any score (log-odds, probability). Pick it on validation data, then apply
    it unchanged to the test set. `default` (0.0 = log-odds of P 0.5) if y has one class."""
    if len(np.unique(y)) < 2:
        return default
    fpr, tpr, thresholds = roc_curve(y, score)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):  # roc_curve's first threshold is +inf: "all negative"
        threshold = float(np.max(score)) + 1.0
    return threshold


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> dict:
    """AUROC (threshold-free) and accuracy, balanced accuracy, macro-F1, sensitivity,
    specificity at `threshold` (predicted positive if score >= threshold).
    `score` is the log-odds by default (threshold 0.0 = P 0.5); probabilities work too
    with threshold 0.5. Metrics that need both classes are None if one is missing
    (e.g. rsna has negatives only: only its specificity is defined)."""
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    both = bool(pos.any() and neg.any())
    return {
        "n": int(len(y)),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, score)) if both else None,
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)) if both else None,
        "sensitivity": float(pred[pos].mean()) if pos.any() else None,
        "specificity": float(1.0 - pred[neg].mean()) if neg.any() else None,
    }


def per_source_metrics(y: np.ndarray, score: np.ndarray, sources: list[str],
                       threshold: float) -> dict[str, dict]:
    """binary_metrics for every source separately."""
    sources = np.asarray(sources)
    return {s: binary_metrics(y[sources == s], score[sources == s], threshold)
            for s in sorted(set(sources.tolist()))}

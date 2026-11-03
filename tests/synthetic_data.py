"""Small synthetic datasets for the tests, so they need no download and no course data.

    python tests/synthetic_data.py OUT_DIR   ->  OUT_DIR/images.npz, OUT_DIR/skeletons.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_images(path: Path, n: int = 600, num_classes: int = 3, seed: int = 0) -> None:
    """28x28 RGB images whose brightness depends on the class (learnable)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    images = rng.normal(60 + 60 * labels[:, None, None, None], 25, size=(n, 28, 28, 3))
    sites = rng.integers(0, 4, size=n)
    np.savez_compressed(path, images=np.clip(images, 0, 255).astype(np.uint8),
                        labels=labels, site=sites)


def make_skeletons(path: Path, sequences: int = 240, seed: int = 0) -> None:
    """HAR-like CSV: several frames per sequence, joints J1X ... J25Z, 3 cameras."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(sequences):
        action, camera = sid % 4, rng.integers(1, 4)
        for frame in range(rng.integers(5, 12)):
            joints = rng.normal(action * 0.3, 0.1 + 0.05 * action, size=75)
            rows.append([sid, frame, *joints, action, camera])
    columns = (["IDSample", "FrameNumber"]
               + [f"J{j}{a}" for j in range(1, 26) for a in "XYZ"] + ["Action", "Camera"])
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    make_images(out / "images.npz")
    make_skeletons(out / "skeletons.csv")

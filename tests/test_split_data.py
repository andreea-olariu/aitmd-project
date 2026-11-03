"""End-to-end tests of data_splitting/split_data.py on synthetic data."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import make_images, make_skeletons  # noqa: E402


def split(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "data_splitting" / "split_data.py"),
                           *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    make_images(d / "images.npz")
    make_skeletons(d / "skeletons.csv")
    return d


def test_images_split(data, tmp_path):
    out = tmp_path / "p"
    r = split("--input", data / "images.npz", "--num-clients", 3, "--strategy", "dirichlet",
              "--alpha", 0.5, "--min-samples", 5, "--out", out)
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "meta.json").read_text())
    assert meta["num_clients"] == 3 and meta["data_type"] == "images"
    total = sum(len(np.load(out / f"client_{i}.npz")["y_train"]) for i in range(3))
    total += sum(len(np.load(out / f"client_{i}.npz")["y_val"]) for i in range(3))
    total += len(np.load(out / "test.npz")["y_test"])
    assert total == 600


def test_csv_split_keeps_sequences_together(data, tmp_path):
    out = tmp_path / "p"
    r = split("--input", data / "skeletons.csv", "--label-col", "Action",
              "--sample-col", "IDSample", "--feature-cols", "^J", "--strategy", "group",
              "--group-col", "Camera", "--num-clients", 3, "--out", out)
    assert r.returncode == 0, r.stderr
    ids = [set(pd.read_csv(out / f"client_{i}_{s}.csv").IDSample)
           for i in range(3) for s in ("train", "val")] + [set(pd.read_csv(out / "test.csv").IDSample)]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j], "a sequence is split between two parts"
    assert np.load(out / "client_0.npz")["x_train"].shape[1] == 4 * 75


def test_test_groups_holds_out_whole_groups(data, tmp_path):
    out = tmp_path / "p"
    r = split("--input", data / "skeletons.csv", "--label-col", "Action",
              "--sample-col", "IDSample", "--feature-cols", "^J", "--strategy", "group",
              "--group-col", "Camera", "--test-groups", 3, "--num-clients", 2, "--out", out)
    assert r.returncode == 0, r.stderr
    assert set(pd.read_csv(out / "test.csv").Camera) == {3}
    for i in range(2):
        assert 3 not in set(pd.read_csv(out / f"client_{i}_train.csv").Camera)


def test_warns_about_classes_nobody_trains_on(data, tmp_path):
    r = split("--input", data / "images.npz", "--num-clients", 1, "--strategy", "pathological",
              "--classes-per-client", 1, "--out", tmp_path / "p")
    assert r.returncode == 0, r.stderr
    assert "WARNING: no client has training data" in r.stdout

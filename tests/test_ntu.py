"""Tests of the NTU RGB+D converter (data_splitting/ntu.py) on a synthetic archive
in the official .skeleton format."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_splitting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ntu  # noqa: E402
from synthetic_ntu import make_archive  # noqa: E402


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("ntu") / "skeletons.zip"
    lengths = make_archive(path, actions=(43, 27, 8), subjects=(1, 3), cameras=(1, 2, 3))
    return path, lengths


@pytest.fixture
def no_missing(tmp_path):
    cache = tmp_path / "missing.txt"
    cache.write_text("")  # no download in the tests
    return cache


def read(out: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(out / f"train_client{i}.csv") for i in (1, 2, 3)]
                     + [pd.read_csv(out / "test.csv")])


def test_build_cross_subject_and_labels(archive, no_missing, tmp_path):
    path, lengths = archive
    out = tmp_path / "har"
    ntu.build(path, [27, 43], [1, 2], out, no_missing)
    df = read(out)
    assert set(df.NTUAction) == {27, 43}                    # action 8 not selected
    assert set(df.Camera) == {1, 2}                         # camera 3 not selected
    assert dict(zip(df.NTUAction, df.Action)) == {27: 0, 43: 1}   # label = position
    test = pd.read_csv(out / "test.csv")
    assert set(test.Subject) == {3}                         # performer 3: cross-subject test
    train = pd.concat([pd.read_csv(out / f"train_client{i}.csv") for i in (1, 2, 3)])
    assert set(train.Subject) == {1}
    assert df.IDSample.nunique() == 2 * 2 * 2               # actions x subjects x cameras


def test_main_body_only_and_frames_without_body(archive, no_missing, tmp_path):
    path, lengths = archive
    out = tmp_path / "har"
    ntu.build(path, [43], [1], out, no_missing)
    df = read(out)
    for _, g in df.groupby("IDSample"):
        assert not g.FrameNumber.duplicated().any()         # the ghost body is dropped
        assert 2 not in set(g.FrameNumber)                  # frame 2 had no body
        assert g.FrameNumber.iloc[0] == 1


def test_all_bodies_keeps_the_ghost(archive, no_missing, tmp_path):
    path, _ = archive
    out = tmp_path / "har"
    ntu.build(path, [43], [1], out, no_missing, bodies="all")
    df = read(out)
    assert all(g.FrameNumber.duplicated().sum() == 1 for _, g in df.groupby("IDSample"))


def test_refuses_to_overwrite(archive, no_missing, tmp_path):
    path, _ = archive
    ntu.build(path, [43], [1], tmp_path, no_missing)
    with pytest.raises(SystemExit):
        ntu.build(path, [43], [1], tmp_path, no_missing)


def test_presets_and_action_ids():
    assert ntu.parse_actions(["course"]) == ntu.PRESETS["course"]
    assert ntu.parse_actions(["A008", "a043", "27"]) == [8, 43, 27]
    with pytest.raises(SystemExit):
        ntu.parse_actions(["A099"])

"""Build the course HAR data from the official NTU RGB+D skeleton archive.

NTU RGB+D (Shahroudy et al., CVPR 2016) may only be used after accepting its Release
Agreement: register at https://rose1.ntu.edu.sg/dataset/actionRecognition/ and
download the 3D skeletons (nturgbd_skeletons_s001_to_s017.zip, NTU RGB+D 60) yourself.
This module never downloads the dataset; it reads the archive you have.

Used through get_data.py:

    python data_splitting/get_data.py --ntu ~/Downloads/nturgbd_skeletons_s001_to_s017.zip
    python data_splitting/get_data.py --ntu <zip> --actions A008 A009 A043 --cameras 1 2
    python data_splitting/get_data.py --ntu <zip> --match data/raw/har    # which NTU actions?

Output (Moodle-compatible, in --out, default data/raw/har/): one row per frame,

    IDSample, FrameNumber, J1X ... J25Z, Action, Camera, Subject, Setup, NTUAction

* the standard NTU **cross-subject** split: performers 1, 2, 4, 5, ... 38 train,
  the others test. Train samples are shuffled into train_client1/2/3.csv (one third
  each, as distributed on Moodle), test samples go to test.csv (with labels).
* Action = 0 ... k-1 in the order of --actions; NTUAction = the official number.
* samples listed by the NTU authors as having missing / incomplete skeletons are skipped.
* only the actor's skeleton is kept (--bodies all keeps every detected body, as in the
  Moodle files, where 138 of 3354 training sequences mix two skeletons).
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# The five actions of the course subset (labels 0 ... 4, in this order). Inferred
# from the Moodle files: per-action sample counts match the NTU missing-skeleton
# list exactly (train and test), and the motion fits (see docs/06_har.md).
# Check with --match once you have the archive.
PRESETS = {
    "course": [44, 27, 15, 26, 43],
}

NTU_ACTIONS = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair",
    5: "drop", 6: "pickup", 7: "throw", 8: "sitting down",
    9: "standing up (from sitting position)", 10: "clapping", 11: "reading", 12: "writing",
    13: "tear up paper", 14: "wear jacket", 15: "take off jacket", 16: "wear a shoe",
    17: "take off a shoe", 18: "wear on glasses", 19: "take off glasses",
    20: "put on a hat/cap", 21: "take off a hat/cap", 22: "cheer up", 23: "hand waving",
    24: "kicking something", 25: "reach into pocket", 26: "hopping (one foot jumping)",
    27: "jump up", 28: "make a phone call/answer phone", 29: "playing with phone/tablet",
    30: "typing on a keyboard", 31: "pointing to something with finger",
    32: "taking a selfie", 33: "check time (from watch)", 34: "rub two hands together",
    35: "nod head/bow", 36: "shake head", 37: "wipe face", 38: "salute",
    39: "put the palms together", 40: "cross hands in front (say stop)",
    41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)", 46: "touch back (backache)",
    47: "touch neck (neckache)", 48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm", 50: "punching/slapping other person",
    51: "kicking other person", 52: "pushing other person",
    53: "pat on back of other person", 54: "point finger at the other person",
    55: "hugging other person", 56: "giving something to other person",
    57: "touch other person's pocket", 58: "handshaking",
    59: "walking towards each other", 60: "walking apart from each other",
}

# Standard cross-subject protocol of NTU RGB+D 60 (Shahroudy et al., 2016)
XSUB_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
MISSING_LIST_URL = ("https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/"
                    "NTU_RGBD_samples_with_missing_skeletons.txt")
NAME = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
JOINT_COLUMNS = [f"J{j}{axis}" for j in range(1, 26) for axis in "XYZ"]


# ---------------------------------------------------------------------------
# Reading the archive
# ---------------------------------------------------------------------------
def list_samples(archive: Path) -> dict[str, tuple]:
    """{sample name: (setup, camera, subject, replication, action)} in the archive."""
    if archive.is_dir():
        names = [p.name for p in archive.rglob("*.skeleton")]
    else:
        with zipfile.ZipFile(archive) as z:
            names = [Path(n).name for n in z.namelist() if n.endswith(".skeleton")]
    samples = {}
    for name in names:
        m = NAME.match(name)
        if m:
            samples[name[:-len(".skeleton")]] = tuple(int(v) for v in m.groups())
    if not samples:
        raise SystemExit(f"ERROR: no .skeleton files (SsssCcccPpppRrrrAaaa.skeleton) in {archive}")
    return samples


class SkeletonReader:
    """Read .skeleton files from the zip or from an extracted folder."""

    def __init__(self, archive: Path):
        self.archive = archive
        self.zip = None if archive.is_dir() else zipfile.ZipFile(archive)
        self.paths = {}
        if self.zip:
            self.paths = {Path(n).stem: n for n in self.zip.namelist() if n.endswith(".skeleton")}
        else:
            self.paths = {p.stem: p for p in archive.rglob("*.skeleton")}

    def lines(self, sample: str) -> list[str]:
        if self.zip:
            return self.zip.read(self.paths[sample]).decode().splitlines()
        return Path(self.paths[sample]).read_text().splitlines()


def parse_skeleton(lines: list[str]) -> list[dict[int, np.ndarray]]:
    """Official .skeleton format -> one {body_id: (25, 3) array} per frame.

    frame count; per frame: body count; per body: one info line (body id first),
    joint count (25), then one line per joint whose first 3 values are x y z
    (camera space, metres)."""
    pos = 0

    def take() -> str:
        nonlocal pos
        pos += 1
        return lines[pos - 1]

    frames = []
    for _ in range(int(take())):
        bodies = {}
        for _ in range(int(take())):
            body_id = int(take().split()[0])
            joints = [take().split()[:3] for _ in range(int(take()))]
            bodies[body_id] = np.array(joints, dtype=np.float64)
        frames.append(bodies)
    return frames


def main_body(frames: list[dict]) -> int | None:
    """The tracked body that moves most (the actor, not a briefly detected ghost)."""
    tracks = defaultdict(list)
    for bodies in frames:
        for body_id, joints in bodies.items():
            tracks[body_id].append(joints)
    if not tracks:
        return None
    return max(tracks, key=lambda b: (np.var(np.stack(tracks[b]), axis=0).sum(), len(tracks[b])))


def load_missing(cache: Path) -> set[str]:
    """Samples the NTU authors list as having missing / incomplete skeletons."""
    if not cache.exists():
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MISSING_LIST_URL, cache)
        except OSError as error:
            print(f"WARNING: could not download the missing-skeleton list ({error}); only "
                  "samples without any tracked body are skipped")
            return set()
    return {m.group(0) for m in NAME.finditer(cache.read_text())}


# ---------------------------------------------------------------------------
# Building the CSV files
# ---------------------------------------------------------------------------
def parse_actions(values: list[str]) -> list[int]:
    if len(values) == 1 and values[0] in PRESETS:
        return PRESETS[values[0]]
    actions = [int(v.upper().lstrip("A")) for v in values]
    unknown = [a for a in actions if a not in NTU_ACTIONS]
    if unknown:
        raise SystemExit(f"ERROR: unknown NTU RGB+D 60 actions {unknown} (A001 ... A060)")
    return actions


def build(archive: Path, actions: list[int], cameras: list[int], out: Path,
          missing_cache: Path, force: bool = False, seed: int = 42,
          bodies: str = "main") -> None:
    """bodies="main": only the actor (the body that moves most). "all": every
    detected body as its own row (two rows per frame number when Kinect saw two
    people or a 'ghost'), which is how the Moodle files were made."""
    targets = [out / f"train_client{i}.csv" for i in (1, 2, 3)] + [out / "test.csv"]
    existing = [t for t in targets if t.exists()]
    if existing and not force:
        raise SystemExit(f"ERROR: {existing[0]} exists. Use another --out, or --force to replace.")

    missing = load_missing(missing_cache)
    samples = list_samples(archive)
    chosen = sorted(name for name, (_, cam, _, _, act) in samples.items()
                    if act in actions and cam in cameras and name not in missing)
    print(f"{len(chosen)} samples: actions {', '.join(f'A{a:03d}' for a in actions)}, "
          f"cameras {cameras} (skipped {sum(n in missing for n in samples)} listed as missing)")

    reader = SkeletonReader(archive)
    rows, info = [], []
    for sample_id, name in enumerate(chosen, start=1):
        setup, cam, subject, _, act = samples[name]
        frames = parse_skeleton(reader.lines(name))
        body = main_body(frames)
        if body is None:
            continue
        label = actions.index(act)
        for frame_no, frame_bodies in enumerate(frames, start=1):
            keep = list(frame_bodies) if bodies == "all" else [body] if body in frame_bodies else []
            for b in keep:
                rows.append([sample_id, frame_no, *frame_bodies[b].reshape(-1), label, cam,
                             subject, setup, act])
        info.append((sample_id, subject, label))
        if sample_id % 500 == 0:
            print(f"  {sample_id} / {len(chosen)}")

    columns = (["IDSample", "FrameNumber", *JOINT_COLUMNS, "Action", "Camera",
                "Subject", "Setup", "NTUAction"])
    df = pd.DataFrame(rows, columns=columns)
    train_ids = [sid for sid, subject, _ in info if subject in XSUB_TRAIN_SUBJECTS]
    test_ids = [sid for sid, subject, _ in info if subject not in XSUB_TRAIN_SUBJECTS]
    thirds = np.array_split(np.random.default_rng(seed).permutation(train_ids), 3)

    out.mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(thirds, start=1):
        df[df.IDSample.isin(set(ids))].to_csv(out / f"train_client{i}.csv", index=False)
    df[df.IDSample.isin(set(test_ids))].to_csv(out / "test.csv", index=False)
    meta = {
        "source": "NTU RGB+D 60, 3D skeletons (Shahroudy et al., CVPR 2016)",
        "actions": {i: {"ntu": f"A{a:03d}", "name": NTU_ACTIONS[a]} for i, a in enumerate(actions)},
        "cameras": cameras,
        "bodies": bodies,
        "split": "cross-subject (train performers: "
                 f"{sorted(XSUB_TRAIN_SUBJECTS)})",
        "samples": {"train": len(train_ids), "test": len(test_ids)},
        "per_action_train": dict(Counter(label for _, subject, label in info
                                         if subject in XSUB_TRAIN_SUBJECTS)),
    }
    (out / "ntu_subset.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote train_client1/2/3.csv ({len(train_ids)} sequences) and test.csv "
          f"({len(test_ids)} sequences, labelled) to {out}")
    for i, a in enumerate(actions):
        print(f"  label {i} = A{a:03d} {NTU_ACTIONS[a]}")


# ---------------------------------------------------------------------------
# Which NTU actions does a set of CSV files contain?
# ---------------------------------------------------------------------------
def match(archive: Path, csv_dir: Path) -> None:
    """Find every CSV sequence in the official archive by its exact first-frame
    coordinates, and report which NTU action each label is."""
    files = sorted(csv_dir.glob("*.csv"))
    firsts = {}
    for path in files:
        df = pd.read_csv(path)
        if "Action" not in df.columns:
            continue
        first = df.sort_values(["IDSample", "FrameNumber"]).groupby("IDSample").first()
        for _, row in first.iterrows():
            key = tuple(np.round(row[["J1X", "J1Y", "J1Z", "J4X", "J4Y", "J4Z"]].to_numpy(float), 5))
            firsts[key] = int(row["Action"])
    if not firsts:
        raise SystemExit(f"ERROR: no labelled CSV files in {csv_dir}")
    print(f"Looking up {len(firsts)} sequences in the archive (this reads every file) ...")

    samples = list_samples(archive)
    reader = SkeletonReader(archive)
    votes = defaultdict(Counter)
    for k, name in enumerate(sorted(samples)):
        frames = parse_skeleton(reader.lines(name))
        for bodies in frames:
            if bodies:
                for joints in bodies.values():
                    key = tuple(np.round(np.concatenate([joints[0], joints[3]]), 5))
                    if key in firsts:
                        votes[firsts[key]][samples[name][4]] += 1
                break
        if k % 5000 == 0:
            print(f"  {k} / {len(samples)}")
    print("\nlabel  found  NTU action")
    for label in sorted(set(firsts.values())):
        counts = votes[label]
        total = sum(counts.values())
        if not total:
            print(f"{label:>5}  {0:>5}  not found")
            continue
        action, n = counts.most_common(1)[0]
        print(f"{label:>5}  {total:>5}  A{action:03d} {NTU_ACTIONS[action]} "
              f"({100 * n / total:.0f} % of the matches)")

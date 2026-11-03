"""Get the datasets you choose into data/raw/ (never committed to git).

Examples (run from the repository root, with the venv activated):

    python data_splitting/get_data.py --list
    python data_splitting/get_data.py --medmnist dermamnist pneumoniamnist
    python data_splitting/get_data.py --medmnist dermamnist --size 64
    python data_splitting/get_data.py --har ~/Downloads/Demo_HAR.zip     # file from Moodle
    python data_splitting/get_data.py --ntu ~/Downloads/nturgbd_skeletons_s001_to_s017.zip

* MedMNIST (https://medmnist.com/) is public and downloaded from Zenodo.
  split_data.py --dataset ... also downloads it by itself; use this script to get
  it in advance (e.g. before working offline).
* The HAR skeleton data is a subsample of NTU RGB+D. Its Release Agreement does not
  allow redistribution, so it is never part of the repository. Two ways to get it:
  - --har: install the course subsample distributed on Moodle;
  - --ntu: build it yourself from the official skeleton archive, after registering
    at https://rose1.ntu.edu.sg/dataset/actionRecognition/ (see ntu.py): choose the
    actions (default: the course preset) and the cameras (default: all 3).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ntu  # noqa: E402
from split_data import MEDMNIST_2D, REPO_ROOT, download_medmnist  # noqa: E402

HAR_DIR = REPO_ROOT / "data" / "raw" / "har"
HAR_COLUMNS = ["IDSample", "FrameNumber", "Action", "Camera"] + [
    f"J{j}{axis}" for j in range(1, 26) for axis in "XYZ"]


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download / install datasets into data/raw/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="show the available datasets")
    parser.add_argument("--medmnist", nargs="+", choices=MEDMNIST_2D, metavar="NAME",
                        help="MedMNIST 2D datasets to download: " + ", ".join(MEDMNIST_2D))
    parser.add_argument("--size", type=int, choices=[28, 64, 128, 224], default=28,
                        help="MedMNIST image size (64 / 128 / 224 are much bigger)")
    parser.add_argument("--har", type=Path,
                        help="the HAR zip downloaded from Moodle (or the unpacked folder)")
    ntu_group = parser.add_argument_group("build the HAR data from the official NTU RGB+D archive")
    ntu_group.add_argument("--ntu", type=Path, metavar="ARCHIVE",
                           help="nturgbd_skeletons_s001_to_s017.zip (or its unpacked folder), "
                                "downloaded by you after accepting the NTU Release Agreement")
    ntu_group.add_argument("--actions", nargs="+", default=["course"],
                           help="'course' (the 5 course actions: "
                                + ", ".join(f"A{a:03d}" for a in ntu.PRESETS["course"])
                                + ") or NTU action ids, e.g. A008 A009 A043 (label = position)")
    ntu_group.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3],
                           choices=[1, 2, 3], help="NTU cameras to keep")
    ntu_group.add_argument("--bodies", choices=["main", "all"], default="main",
                           help="'main' = only the actor; 'all' = every detected body as its own "
                                "rows (like the Moodle files; mixes people in some sequences)")
    ntu_group.add_argument("--out", type=Path, default=HAR_DIR, help="output folder")
    ntu_group.add_argument("--force", action="store_true", help="replace existing CSV files")
    ntu_group.add_argument("--match", type=Path, metavar="CSV_DIR",
                           help="instead of building: find which NTU actions the labelled "
                                "CSV files in CSV_DIR contain (e.g. the Moodle files)")
    args = parser.parse_args()
    if not (args.list or args.medmnist or args.har or args.ntu):
        parser.print_help()
        sys.exit(1)
    return args


def show_list() -> None:
    print("MedMNIST 2D (public, downloaded from Zenodo, https://medmnist.com/):")
    for name in MEDMNIST_2D:
        state = "downloaded" if (REPO_ROOT / "data" / "raw" / f"{name}.npz").exists() else ""
        print(f"  {name:<16} {state}")
    print("  --size 28 (default, a few MB per dataset) or 64 / 128 / 224 px (much bigger: "
          "dermamnist is 20 MB at 28 px and 100 MB at 64 px)")
    har = sorted(HAR_DIR.glob("train_client*.csv"))
    print("\nHAR skeletons (course data, from Moodle; install with --har <zip>):")
    print(f"  {'installed: ' + ', '.join(p.name for p in har) if har else 'not installed'}")


def install_har(source: Path) -> None:
    source = source.expanduser()
    if not source.exists():
        sys.exit(f"ERROR: {source} does not exist")
    with tempfile.TemporaryDirectory() as tmp:
        if source.is_file():
            if not zipfile.is_zipfile(source):
                sys.exit(f"ERROR: {source} is not a zip file or a folder")
            with zipfile.ZipFile(source) as archive:
                archive.extractall(tmp)
            root = Path(tmp)
        else:
            root = source
        files = sorted(p for p in root.rglob("*.csv") if "__MACOSX" not in p.parts)
        if not files:
            sys.exit(f"ERROR: no .csv files found in {source}")

        HAR_DIR.mkdir(parents=True, exist_ok=True)
        installed = []
        for path in files:
            columns = list(pd.read_csv(path, nrows=0).columns)
            missing = [c for c in HAR_COLUMNS if c not in columns]
            if "Action" in missing:
                print(f"  skip {path.name}: no 'Action' column (no labels to train or "
                      "evaluate on)")
                continue
            if missing:
                print(f"  skip {path.name}: missing columns {missing[:5]}")
                continue
            shutil.copy2(path, HAR_DIR / path.name)
            installed.append(path.name)
    if not installed:
        sys.exit("ERROR: no valid HAR file found (expected columns IDSample, FrameNumber, "
                 "J1X ... J25Z, Action, Camera)")
    print(f"Installed {', '.join(installed)} in {HAR_DIR}")
    print("Next: python data_splitting/split_data.py --input data/raw/har/train_client*.csv "
          "--label-col Action --sample-col IDSample --feature-cols '^J' ...  (docs/06_har.md)")


def main() -> None:
    args = get_arguments()
    if args.list:
        show_list()
    for name in args.medmnist or []:
        path = download_medmnist(name, args.size)
        print(f"{name} ({args.size} px): {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.har:
        install_har(args.har)
    if args.ntu:
        if not args.ntu.expanduser().exists():
            sys.exit(f"ERROR: {args.ntu} does not exist")
        archive = args.ntu.expanduser()
        if args.match:
            ntu.match(archive, args.match)
        else:
            ntu.build(archive, ntu.parse_actions(args.actions), sorted(args.cameras), args.out,
                      REPO_ROOT / "data" / "raw" / "ntu_missing_skeletons.txt", args.force,
                      bodies=args.bodies)


if __name__ == "__main__":
    main()

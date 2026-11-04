"""Preprocess COVIDx CXR-4 once: grayscale, square, downsampled PNGs + one manifest.

Input (as distributed):
    data/raw/{train,val,test}/<image files>      1024x1024-ish PNG / JPEG
    data/raw/{train,val,test}.txt                "<patient_id> <filename> <label> <source>"

Output:
    data/processed/<size>/{train,val,test}/<stem>.png   uint8 grayscale, size x size
    data/processed/<size>/manifest.csv                  one row per image (see COLUMNS)
    data/processed/<size>/summary.json                  counts + sanity checks

Every image is converted to grayscale, centre-cropped to a square (the aspect ratio
is never distorted; for the common 1024x~840 images this trims ~90 px of shoulders /
arms per side), then resized with a high-quality filter. Cropping, not padding, is the
default because only some sources are non-square: black bars would tell the model the
source, and the source almost determines the label in the training set.
Already-converted images are skipped, so an interrupted run can simply be restarted
(use --overwrite to redo everything).

    python data_splitting/preprocess_covidx.py
    python data_splitting/preprocess_covidx.py --size 224 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

SPLITS = ("train", "val", "test")
LABELS = {"negative": 0, "positive": 1}
COLUMNS = ["split", "path", "label", "label_name", "source", "patient_id",
           "orig_filename", "orig_width", "orig_height", "orig_mode"]


def read_split_file(raw_dir: Path, split: str) -> list[dict]:
    """Parse <split>.txt. Lines are space-separated: patient_id filename label source."""
    rows = []
    path = raw_dir / f"{split}.txt"
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_no}: expected 4 fields, got {len(parts)}: {line!r}")
        patient_id, filename, label_name, source = parts
        if label_name not in LABELS:
            raise ValueError(f"{path}:{line_no}: unknown label {label_name!r}")
        rows.append({"split": split, "patient_id": patient_id, "orig_filename": filename,
                     "label_name": label_name, "label": LABELS[label_name], "source": source})
    return rows


def convert_one(task: tuple[str, str, int, str, bool]) -> tuple[int, int, str]:
    """Convert one image; returns its original (width, height, mode).
    Runs in a worker process, so it only takes / returns plain values."""
    src, dst, size, resize_mode, overwrite = task
    with Image.open(src) as img:
        width, height, mode = img.width, img.height, img.mode
        if overwrite or not os.path.exists(dst):
            gray = img.convert("L")  # RGBA / RGB / I;16 -> 8-bit grayscale
            if width != height:
                if resize_mode == "crop":  # centre square of side min(w, h)
                    side = min(width, height)
                    left, top = (width - side) // 2, (height - side) // 2
                    gray = gray.crop((left, top, left + side, top + side))
                else:  # "pad": letterbox with black bars
                    side = max(width, height)
                    gray = ImageOps.pad(gray, (side, side), color=0, centering=(0.5, 0.5))
            gray = gray.resize((size, size), Image.Resampling.LANCZOS)
            tmp = dst + ".tmp"
            gray.save(tmp, format="PNG", optimize=False)
            os.replace(tmp, dst)  # atomic: no half-written files if interrupted
    return width, height, mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"),
                        help="images go to <out>/<size>/")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=["crop", "pad"], default="crop",
                        help="non-square images: centre-crop to a square, or pad with "
                             "black bars. Padding marks the non-square sources (mostly "
                             "bimcv) and gives the model a shortcut, so crop is default.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out / str(args.size)
    rows: list[dict] = []
    for split in SPLITS:
        rows.extend(read_split_file(args.raw, split))

    # Check inputs and plan outputs before doing any work.
    missing, seen = [], set()
    for row in rows:
        src = args.raw / row["split"] / row["orig_filename"]
        if not src.exists():
            missing.append(str(src))
        # Some sources ship foo.jpg AND foo.png as different images (cohen), so keep
        # the original extension in the name of anything that was not a PNG.
        orig = Path(row["orig_filename"])
        ext = orig.suffix.lower()
        stem = orig.stem if ext == ".png" else f"{orig.stem}_{ext.lstrip('.')}"
        rel = f"{row['split']}/{stem}.png"
        if rel in seen:
            sys.exit(f"Output name collision: {rel}")
        seen.add(rel)
        row["path"] = rel
        row["_src"] = str(src)
    if missing:
        sys.exit(f"{len(missing)} listed images are missing, e.g. {missing[:3]}")

    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    # Images converted with a different --resize-mode must not be silently reused.
    mode_file = out_dir / "resize_mode.txt"
    previous = mode_file.read_text().strip() if mode_file.exists() else None
    if previous is not None and previous != args.resize_mode and not args.overwrite:
        sys.exit(f"{out_dir} holds '{previous}' images; pass --overwrite to redo them "
                 f"with '{args.resize_mode}'.")
    mode_file.write_text(args.resize_mode)

    tasks = [(r["_src"], str(out_dir / r["path"]), args.size, args.resize_mode,
              args.overwrite) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(tqdm(pool.map(convert_one, tasks, chunksize=64), total=len(tasks),
                            desc=f"Converting to {args.size}px grayscale"))
    for row, (width, height, mode) in zip(rows, results):
        row["orig_width"], row["orig_height"], row["orig_mode"] = width, height, mode

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Sanity checks worth knowing before any training.
    patients = defaultdict(set)
    for row in rows:
        patients[row["split"]].add(row["patient_id"])
    overlap = {f"{a}&{b}": len(patients[a] & patients[b])
               for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    summary = {
        "size": args.size,
        "resize_mode": args.resize_mode,
        "num_images": len(rows),
        "per_split": {s: dict(Counter(r["label_name"] for r in rows if r["split"] == s))
                      for s in SPLITS},
        "per_split_source_label": {
            s: {f"{src}/{lab}": n for (src, lab), n in sorted(Counter(
                (r["source"], r["label_name"]) for r in rows if r["split"] == s).items())}
            for s in SPLITS},
        "unique_patients": {s: len(patients[s]) for s in SPLITS},
        "patient_overlap_between_splits": overlap,
        "original_sizes": {f"{w}x{h}": n for (w, h), n in Counter(
            (r["orig_width"], r["orig_height"]) for r in rows).most_common(10)},
        "original_modes": dict(Counter(r["orig_mode"] for r in rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {len(rows)} images and {manifest}")
    print(f"Patient overlap between splits: {overlap}")
    print(f"Most common original sizes: {summary['original_sizes']}")


if __name__ == "__main__":
    main()

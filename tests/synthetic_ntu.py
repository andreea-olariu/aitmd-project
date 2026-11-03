"""Write a small archive in the official NTU RGB+D .skeleton format (for the tests;
the real dataset is never needed or downloaded)."""

import zipfile
from pathlib import Path

import numpy as np


def skeleton_text(frames: list[list[tuple[int, np.ndarray]]]) -> str:
    """frames: per frame a list of (body_id, (25, 3) joints). Official layout: frame
    count; per frame body count; per body an info line (10 values, body id first),
    the joint count and 25 joint lines of 12 values (x y z first)."""
    out = [str(len(frames))]
    for bodies in frames:
        out.append(str(len(bodies)))
        for body_id, joints in bodies:
            out.append(f"{body_id} 0 1 1 1 1 0 0.01 -0.2 2")
            out.append("25")
            for x, y, z in joints:
                out.append(f"{float(x)!r} {float(y)!r} {float(z)!r} 260.1 190.2 1052.3 512.4 0.3 0.01 0.9 0.02 2")
    return "\n".join(out) + "\n"


def make_archive(path: Path, actions=(43, 27), subjects=(1, 3), cameras=(1, 2, 3),
                 seed: int = 0) -> dict[str, int]:
    """Zip with one sample per (action, subject, camera). Each sample has a frame
    without any body and a short-lived second 'ghost' body. Returns {name: frames}."""
    rng = np.random.default_rng(seed)
    lengths = {}
    with zipfile.ZipFile(path, "w") as z:
        for action in actions:
            for subject in subjects:
                for camera in cameras:
                    name = f"S001C{camera:03d}P{subject:03d}R001A{action:03d}"
                    n = int(rng.integers(6, 12))
                    base = rng.normal(0, 0.3, size=(25, 3)) + [0, 0, 3]
                    frames = []
                    for t in range(n):
                        actor = base + 0.05 * t * rng.normal(size=(25, 3))
                        bodies = [(72057594037931101, actor)]
                        if t == 1:
                            bodies = []                      # Kinect lost the person
                        if t == 2:
                            bodies.append((72057594037931999, base + 1.0))  # ghost
                        frames.append(bodies)
                    z.writestr(f"nturgb+d_skeletons/{name}.skeleton", skeleton_text(frames))
                    lengths[name] = n
    return lengths

"""
scripts/make_preview_gif.py
----------------------------
Generate a rotating 3D preview GIF of the Paris scan predictions.

Usage:
    python scripts/make_preview_gif.py
    python scripts/make_preview_gif.py --n_points 150000 --n_frames 36 --fps 12
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRED_DIR = Path("outputs/predictions")
PROC_DIR = Path("data/processed")
OUT_PATH = Path("outputs/figures/preview.gif")
BG       = "#0d1117"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan",     type=str,   default="Paris")
    p.add_argument("--n_points", type=int,   default=150_000)
    p.add_argument("--n_frames", type=int,   default=36)
    p.add_argument("--elev",     type=float, default=35.0)
    p.add_argument("--fps",      type=int,   default=12)
    p.add_argument("--out",      type=str,   default=str(OUT_PATH))
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import Patch
    from PIL import Image

    from src.data.loader import CLASS_COLORS, CLASS_NAMES

    args = parse_args()

    print("Loading data ...")
    xyz  = np.load(PROC_DIR / args.scan / "points.npy")[:, :3].astype(np.float32)
    pred = np.load(PRED_DIR / f"{args.scan}_pred.npy").astype(np.int32)

    # Subsample: keep mostly labeled points (pred > 0)
    rng = np.random.default_rng(42)
    labeled_idx   = np.where(pred > 0)[0]
    unlabeled_idx = np.where(pred == 0)[0]
    n_labeled   = min(int(args.n_points * 0.92), len(labeled_idx))
    n_unlabeled = min(args.n_points - n_labeled, len(unlabeled_idx))
    idx = np.concatenate([
        rng.choice(labeled_idx,   n_labeled,   replace=False),
        rng.choice(unlabeled_idx, n_unlabeled, replace=False),
    ])
    xyz_s  = xyz[idx]
    pred_s = pred[idx]

    # Center scene
    xyz_s -= xyz_s.mean(axis=0)

    colors = CLASS_COLORS[np.clip(pred_s, 0, len(CLASS_COLORS) - 1)]

    present = sorted(np.unique(pred_s))
    legend_handles = [
        Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in present if 0 < c < len(CLASS_NAMES)
    ]

    print(f"Rendering {args.n_frames} frames ...")
    frames   = []
    azimuths = np.linspace(0, 360, args.n_frames, endpoint=False)

    for i, az in enumerate(azimuths):
        fig = plt.figure(figsize=(9, 5), facecolor=BG)
        ax  = fig.add_subplot(111, projection="3d", facecolor=BG)

        ax.scatter(
            xyz_s[:, 0], xyz_s[:, 1], xyz_s[:, 2],
            c=colors, s=0.4, linewidths=0, depthshade=False,
        )

        ax.view_init(elev=args.elev, azim=float(az))
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 0.10])  # flatten Z — urban street scene

        ax.legend(
            handles=legend_handles, loc="lower left",
            facecolor="#161b22", edgecolor="#30363d",
            labelcolor="white", fontsize=7, markerscale=3,
            framealpha=0.85, ncol=2,
        )
        fig.suptitle(
            "Paris — Semantic predictions (PointTransformer  ·  73.4% mIoU)",
            color="white", fontsize=9, y=0.97,
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=90, bbox_inches="tight",
                    facecolor=BG, pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

        if (i + 1) % 6 == 0 or i == 0:
            print(f"  {i + 1}/{args.n_frames}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / args.fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"Saved -> {out_path}  ({size_mb:.1f} MB, {len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()

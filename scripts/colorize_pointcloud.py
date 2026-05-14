"""
scripts/colorize_pointcloud.py
───────────────────────────────
Project a LiDAR point cloud onto a camera image and export a colored PLY.

Usage examples
──────────────
# Real data
python scripts/colorize_pointcloud.py \
    --points  data/demo/scan.npy \
    --image   data/demo/frame.png \
    --calib   data/demo/calibration.json \
    --out     outputs/fusion/scan_colored.ply

# Synthetic demo (no real data needed)
python scripts/colorize_pointcloud.py --demo

Outputs
───────
  <out>               — colored PLY (binary little-endian)
  <out stem>_overlay.png — image with projected points overlaid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── project root on path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.projection import (
    colorize_pointcloud,
    export_colored_ply,
    load_camera_calibration,
)


# ─────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_points(path: Path) -> np.ndarray:
    """Load (N, 3+) point cloud from .npy or binary PLY."""
    if path.suffix == ".npy":
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected (N, 3+) array, got {pts.shape}")
        return pts[:, :3].astype(np.float64)

    if path.suffix == ".ply":
        return _load_ply_xyz(path)

    raise ValueError(f"Unsupported point cloud format: {path.suffix}  (use .npy or .ply)")


def _load_ply_xyz(path: Path) -> np.ndarray:
    """Minimal binary/ASCII PLY reader that extracts x, y, z."""
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").rstrip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        props: list[str] = []
        is_binary_le = False
        for ln in header_lines:
            if ln.startswith("element vertex"):
                n_vertices = int(ln.split()[-1])
            elif ln.startswith("property float") or ln.startswith("property double"):
                props.append(ln.split()[-1])
            elif ln == "format binary_little_endian 1.0":
                is_binary_le = True

        if not is_binary_le:
            raise NotImplementedError("Only binary little-endian PLY is supported here.")

        dtype = np.dtype([(p, "<f4") for p in props])
        raw = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)

    return np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)


def _load_image(path: Path) -> np.ndarray:
    """Load RGB image as (H, W, 3) uint8. Requires Pillow or matplotlib."""
    try:
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return img
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[:, :, :3]
    except ImportError:
        pass
    raise ImportError("Install Pillow (pip install Pillow) to load images.")


def _save_image(path: Path, img: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), img)
        return
    except ImportError:
        pass
    raise ImportError("Install Pillow (pip install Pillow) to save images.")


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay rendering (pure NumPy — matplotlib used only if available)
# ─────────────────────────────────────────────────────────────────────────────

def _depth_colormap(depth: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Map scalar depth values to a jet-like RGB colormap (uint8, shape (N,3))."""
    t = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    # Jet: blue→cyan→green→yellow→red
    r = np.clip(1.5 - abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _draw_overlay(
    image: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    """Draw visible projected points on a copy of the image (depth-colored)."""
    overlay = image.copy()
    H, W = overlay.shape[:2]

    vis_uv  = uv[mask]
    vis_dep = depth[mask]
    colors  = _depth_colormap(vis_dep, vis_dep.min(), vis_dep.max())

    ui = np.round(vis_uv[:, 0]).astype(np.int32)
    vi = np.round(vis_uv[:, 1]).astype(np.int32)

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            vr = np.clip(vi + dr, 0, H - 1)
            uc = np.clip(ui + dc, 0, W - 1)
            overlay[vr, uc] = colors

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic demo
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data(calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic checkerboard scene visible from the demo calibration."""
    rng = np.random.default_rng(0)
    W, H = calib["image_size"]

    # Random points in front of the camera (camera frame: Z=5-20m, X/Y=±5m)
    N = 5_000
    pts_cam = np.column_stack([
        rng.uniform(-5, 5, N),
        rng.uniform(-3, 3, N),
        rng.uniform(5, 20, N),
    ])

    # Colorful synthetic image
    img = np.zeros((H, W, 3), dtype=np.uint8)
    sq = 80
    for r in range(H // sq + 1):
        for c in range(W // sq + 1):
            col = (200, 200, 200) if (r + c) % 2 == 0 else (60, 60, 60)
            r0, c0 = r * sq, c * sq
            img[r0:r0 + sq, c0:c0 + sq] = col
    # Add a color gradient strip
    img[H // 2 - 20:H // 2 + 20, :] = np.linspace([255, 0, 0], [0, 0, 255], W, dtype=np.uint8)

    # Transform camera-frame points to LiDAR frame
    T_cam_to_lidar = np.linalg.inv(calib["T_lidar_to_cam"])
    R, t = T_cam_to_lidar[:3, :3], T_cam_to_lidar[:3, 3]
    pts_lidar = pts_cam @ R.T + t

    return pts_lidar, img


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project LiDAR points onto a camera image and export colored PLY."
    )
    parser.add_argument("--points",  type=Path, help="Input point cloud (.npy or .ply)")
    parser.add_argument("--image",   type=Path, help="Input RGB image (.png / .jpg)")
    parser.add_argument("--calib",   type=Path, help="Calibration JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/fusion/colored.ply"),
        help="Output PLY path (default: outputs/fusion/colored.ply)",
    )
    parser.add_argument(
        "--mode",
        choices=["nearest", "bilinear"],
        default="bilinear",
        help="Color sampling mode (default: bilinear)",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.1,
        help="Minimum valid depth in metres (default: 0.1)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic data and run a full demo (no real files needed)",
    )
    args = parser.parse_args()

    # ── load / generate data ──────────────────────────────────────────────
    calib_path = args.calib or Path(__file__).resolve().parents[1] / "data/demo/calibration.json"
    calib = load_camera_calibration(calib_path)

    if args.demo:
        print(f"[demo]  Generating synthetic scene from {calib_path}")
        points_xyz, image = _make_demo_data(calib)
        out_path = args.out
    else:
        if args.points is None or args.image is None:
            parser.error("--points and --image are required unless --demo is set")
        print(f"[load]  Points : {args.points}")
        print(f"[load]  Image  : {args.image}")
        points_xyz = _load_points(args.points)
        image      = _load_image(args.image)
        out_path   = args.out

    W, H = calib["image_size"]
    print(f"[info]  Points     : {len(points_xyz):,}")
    print(f"[info]  Image size : {image.shape[1]}×{image.shape[0]}  (calib expects {W}×{H})")
    print(f"[info]  Calib      : fx={calib['K'][0,0]:.1f}  fy={calib['K'][1,1]:.1f}  "
          f"cx={calib['K'][0,2]:.1f}  cy={calib['K'][1,2]:.1f}")

    # ── run colorization pipeline ─────────────────────────────────────────
    colors, visible, uv_all, depth_all = colorize_pointcloud(
        points_xyz, image, calib,
        sample_mode=args.mode,
        min_depth=args.min_depth,
    )

    n_vis = visible.sum()
    pct   = 100.0 * n_vis / len(points_xyz)
    d_min = depth_all[visible].min() if n_vis > 0 else float("nan")
    d_max = depth_all[visible].max() if n_vis > 0 else float("nan")
    print(f"\n[proj]  Visible : {n_vis:,} / {len(points_xyz):,}  ({pct:.1f}%)")
    print(f"[proj]  Depth range : {d_min:.2f} – {d_max:.2f} m")

    # ── export PLY ────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_colored_ply(out_path, points_xyz, colors)
    print(f"\n[out]   PLY       -> {out_path}")

    # ── export overlay image ──────────────────────────────────────────────
    overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
    overlay = _draw_overlay(image, uv_all, depth_all, visible, radius=2)
    try:
        _save_image(overlay_path, overlay)
        print(f"[out]   Overlay   -> {overlay_path}")
    except ImportError as e:
        print(f"[warn]  Could not save overlay ({e})")

    print("\nDone.")


if __name__ == "__main__":
    main()

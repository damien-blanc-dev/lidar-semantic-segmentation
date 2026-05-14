"""
scripts/eval_projection.py
───────────────────────────
Calibration sensitivity analysis for camera-LiDAR projection.

For each perturbation level the script reports:
  - visibility change  (Δ visible points vs nominal)
  - mean pixel displacement of visible points
  - fraction of points that leave the image after the perturbation

Outputs
───────
  outputs/fusion/sensitivity.csv  — per-perturbation statistics
  outputs/fusion/sensitivity.png  — bar chart (matplotlib, optional)
  outputs/fusion/overlay_*.png    — nominal + worst-case overlay images

Usage examples
──────────────
# Synthetic demo (no real data needed)
python scripts/eval_projection.py --demo

# Real data
python scripts/eval_projection.py \
    --points data/demo/scan.npy \
    --image  data/demo/frame.png \
    --calib  data/demo/calibration.json \
    --out    outputs/fusion
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fusion.projection import (
    colorize_pointcloud,
    filter_visible_points,
    load_camera_calibration,
    perturb_extrinsics,
    project_points,
    transform_points,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Perturbation grid
# ─────────────────────────────────────────────────────────────────────────────

PERTURBATIONS: list[dict] = [
    {"label": "nominal",           "trans_m": 0.00, "rot_deg": 0.0},
    {"label": "trans +1 cm",       "trans_m": 0.01, "rot_deg": 0.0},
    {"label": "trans +5 cm",       "trans_m": 0.05, "rot_deg": 0.0},
    {"label": "trans -5 cm",       "trans_m":-0.05, "rot_deg": 0.0},
    {"label": "rot  +0.5°",        "trans_m": 0.00, "rot_deg": 0.5},
    {"label": "rot  +1.0°",        "trans_m": 0.00, "rot_deg": 1.0},
    {"label": "rot  +2.0°",        "trans_m": 0.00, "rot_deg": 2.0},
    {"label": "rot  -2.0°",        "trans_m": 0.00, "rot_deg":-2.0},
    {"label": "trans+1cm rot+0.5°","trans_m": 0.01, "rot_deg": 0.5},
    {"label": "trans+5cm rot+2.0°","trans_m": 0.05, "rot_deg": 2.0},
]


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (reused from colorize_pointcloud, kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path)
        return pts[:, :3].astype(np.float64)
    raise ValueError(f"Unsupported format: {path.suffix}")


def _load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    try:
        import matplotlib.pyplot as plt
        img = plt.imread(str(path))
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img[:, :, :3]
    except ImportError:
        pass
    raise ImportError("Install Pillow to load images.")


def _depth_colormap(depth: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    t = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _draw_overlay(image: np.ndarray, uv: np.ndarray, depth: np.ndarray,
                  mask: np.ndarray, radius: int = 2) -> np.ndarray:
    overlay = image.copy()
    H, W = overlay.shape[:2]
    if not mask.any():
        return overlay
    vis_uv  = uv[mask]
    vis_dep = depth[mask]
    colors  = _depth_colormap(vis_dep, vis_dep.min(), vis_dep.max())
    ui = np.round(vis_uv[:, 0]).astype(np.int32)
    vi = np.round(vis_uv[:, 1]).astype(np.int32)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            overlay[np.clip(vi + dr, 0, H - 1), np.clip(ui + dc, 0, W - 1)] = colors
    return overlay


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


# ─────────────────────────────────────────────────────────────────────────────
#  Projection statistics
# ─────────────────────────────────────────────────────────────────────────────

def _project_with_T(points_xyz: np.ndarray, calib: dict,
                    T_override: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (uv, depth, visible_mask) for points_xyz with optional T override."""
    T   = T_override if T_override is not None else calib["T_lidar_to_cam"]
    pts = transform_points(points_xyz, T)
    uv, depth = project_points(pts, calib["K"])
    mask = filter_visible_points(uv, depth, calib["image_size"])
    return uv, depth, mask


def _compute_stats(
    uv_nom: np.ndarray,
    depth_nom: np.ndarray,
    mask_nom: np.ndarray,
    uv_pert: np.ndarray,
    depth_pert: np.ndarray,
    mask_pert: np.ndarray,
    n_total: int,
) -> dict:
    n_nom  = int(mask_nom.sum())
    n_pert = int(mask_pert.sum())

    # Pixel displacement only for points visible in BOTH projections
    both = mask_nom & mask_pert
    if both.any():
        disp = np.linalg.norm(uv_pert[both] - uv_nom[both], axis=1)
        mean_disp = float(disp.mean())
        p95_disp  = float(np.percentile(disp, 95))
    else:
        mean_disp = float("nan")
        p95_disp  = float("nan")

    # Points that were visible nominally but fall outside after perturbation
    lost = mask_nom & ~mask_pert
    n_lost = int(lost.sum())

    return {
        "n_visible_nominal":    n_nom,
        "n_visible_perturbed":  n_pert,
        "delta_visible":        n_pert - n_nom,
        "pct_visible_nominal":  round(100.0 * n_nom  / n_total, 2),
        "pct_visible_perturbed":round(100.0 * n_pert / n_total, 2),
        "mean_pixel_disp":      round(mean_disp, 3) if not np.isnan(mean_disp) else "",
        "p95_pixel_disp":       round(p95_disp,  3) if not np.isnan(p95_disp)  else "",
        "n_lost_points":        n_lost,
        "pct_lost":             round(100.0 * n_lost / max(n_nom, 1), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic demo scene
# ─────────────────────────────────────────────────────────────────────────────

def _make_demo_data(calib: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    W, H = calib["image_size"]
    N = 8_000
    pts_cam = np.column_stack([
        rng.uniform(-6, 6, N),
        rng.uniform(-4, 4, N),
        rng.uniform(3, 25, N),
    ])
    img = np.zeros((H, W, 3), dtype=np.uint8)
    sq = 80
    for r in range(H // sq + 1):
        for c in range(W // sq + 1):
            col = (200, 200, 200) if (r + c) % 2 == 0 else (60, 60, 60)
            img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = col
    img[H // 2 - 20:H // 2 + 20, :] = np.linspace([255, 0, 0], [0, 0, 255], W, dtype=np.uint8)

    T_inv = np.linalg.inv(calib["T_lidar_to_cam"])
    pts_lidar = pts_cam @ T_inv[:3, :3].T + T_inv[:3, 3]
    return pts_lidar, img


# ─────────────────────────────────────────────────────────────────────────────
#  Optional matplotlib figure
# ─────────────────────────────────────────────────────────────────────────────

def _save_sensitivity_figure(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    labels    = [r["label"]         for r in rows]
    mean_disp = [float(r["mean_pixel_disp"]) if r["mean_pixel_disp"] != "" else 0.0 for r in rows]
    pct_lost  = [float(r["pct_lost"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))

    axes[0].bar(x, mean_disp, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean pixel displacement")
    axes[0].set_title("Projection shift per perturbation")
    axes[0].axhline(0, color="k", linewidth=0.5)

    axes[1].bar(x, pct_lost, color="tomato")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("% points leaving FOV")
    axes[1].set_title("Visibility loss per perturbation")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[out]   Figure    -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration sensitivity analysis for camera-LiDAR projection."
    )
    parser.add_argument("--points", type=Path)
    parser.add_argument("--image",  type=Path)
    parser.add_argument("--calib",  type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/fusion"),
        help="Output directory (default: outputs/fusion)",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data (no real files needed)")
    args = parser.parse_args()

    # ── data loading ──────────────────────────────────────────────────────
    calib_path = args.calib or Path(__file__).resolve().parents[1] / "data/demo/calibration.json"
    calib = load_camera_calibration(calib_path)

    if args.demo:
        print(f"[demo]  Generating synthetic scene  (calib: {calib_path})")
        points_xyz, image = _make_demo_data(calib)
    else:
        if args.points is None or args.image is None:
            parser.error("--points and --image are required unless --demo is set")
        points_xyz = _load_points(args.points)
        image      = _load_image(args.image)

    args.out.mkdir(parents=True, exist_ok=True)
    n_total = len(points_xyz)
    print(f"[info]  Total points : {n_total:,}")

    # ── nominal projection ────────────────────────────────────────────────
    uv_nom, depth_nom, mask_nom = _project_with_T(points_xyz, calib)
    print(f"[nom]   Visible : {mask_nom.sum():,} / {n_total:,}  "
          f"({100*mask_nom.mean():.1f}%)")

    # ── perturbation sweep ────────────────────────────────────────────────
    rows: list[dict] = []
    header  = (
        f"{'Label':<28} {'Vis_nom':>8} {'Vis_pert':>9} {'dVis':>6} "
        f"{'mean_px':>8} {'p95_px':>7} {'lost%':>7}"
    )
    divider = "-" * len(header)
    print(f"\n{header}")
    print(divider)

    overlay_saved: list[str] = []

    for p in PERTURBATIONS:
        T_pert = perturb_extrinsics(
            calib["T_lidar_to_cam"],
            translation_noise_m=p["trans_m"],
            rotation_noise_deg=p["rot_deg"],
        )
        uv_p, depth_p, mask_p = _project_with_T(points_xyz, calib, T_override=T_pert)
        stats = _compute_stats(uv_nom, depth_nom, mask_nom, uv_p, depth_p, mask_p, n_total)
        row = {**p, **stats}
        rows.append(row)

        print(
            f"  {p['label']:<26} "
            f"{stats['n_visible_nominal']:>8,} "
            f"{stats['n_visible_perturbed']:>9,} "
            f"{stats['delta_visible']:>+6,} "
            f"{str(stats['mean_pixel_disp']):>8} "
            f"{str(stats['p95_pixel_disp']):>7} "
            f"{stats['pct_lost']:>6.2f}%"
        )

        # Save overlay for nominal + worst rotation + worst combined
        if p["label"] in ("nominal", "rot  +2.0°", "trans+5cm rot+2.0°"):
            mask_draw = mask_p if p["label"] != "nominal" else mask_nom
            uv_draw   = uv_p   if p["label"] != "nominal" else uv_nom
            dep_draw  = depth_p if p["label"] != "nominal" else depth_nom
            slug  = p["label"].replace(" ", "_").replace("+", "p").replace("°", "deg")
            opath = args.out / f"overlay_{slug}.png"
            ov    = _draw_overlay(image, uv_draw, dep_draw, mask_draw)
            try:
                _save_image(opath, ov)
                overlay_saved.append(str(opath))
            except Exception:
                pass

    print(divider)

    # ── CSV export ────────────────────────────────────────────────────────
    csv_path = args.out / "sensitivity.csv"
    csv_fields = [
        "label", "trans_m", "rot_deg",
        "n_visible_nominal", "n_visible_perturbed", "delta_visible",
        "pct_visible_nominal", "pct_visible_perturbed",
        "mean_pixel_disp", "p95_pixel_disp",
        "n_lost_points", "pct_lost",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[out]   CSV       -> {csv_path}")

    for p in overlay_saved:
        print(f"[out]   Overlay   -> {p}")

    # ── optional figure ───────────────────────────────────────────────────
    _save_sensitivity_figure(rows, args.out / "sensitivity.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

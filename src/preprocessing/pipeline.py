"""
src/preprocessing/pipeline.py
──────────────────────────────
Preprocessing pipeline: raw PLY → feature-rich numpy arrays ready for training.

Pipeline per scan:
    1. Voxel downsampling      (0.05 m)   — uniform density, ~10× fewer points
    2. Normal estimation       (r=0.3 m)  — local surface orientation
    3. Feature assembly                   — [x, y, z, reflectance, nx, ny, nz]
    4. Save to disk                       — points.npy (N,7) + labels.npy (N,)

Block cropping (4 m × 4 m) and height-above-ground are computed on-the-fly
in the Dataset class, not here. This keeps preprocessing fast and lets you
change block parameters without re-running this script.

Analogy to CT: this is equivalent to resampling a volume to isotropic voxels
and computing gradient magnitude — you do it once and cache the result.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Voxel downsampling
# ─────────────────────────────────────────────────────────────────────────────

def voxel_downsample(
    xyz: np.ndarray,
    reflectance: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    voxel_size: float = 0.05,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Voxel grid downsampling with label and reflectance preservation.

    Strategy:
      - Geometry (xyz): centroid of all points in the voxel
      - Reflectance:    mean of all points in the voxel
      - Label:          majority vote within the voxel

    Open3D's built-in voxel_down_sample is fast but discards scalar fields,
    so we implement the grid directly with numpy. For 30M → 3M points at
    0.05 m, this runs in ~15 s on a modern CPU.

    Parameters
    ----------
    xyz         : (N, 3) float32
    reflectance : (N,)   float32, or None
    labels      : (N,)   int32, or None
    voxel_size  : leaf size in meters

    Returns
    -------
    xyz_down, reflectance_down, labels_down  (same dtypes, fewer rows)
    """
    t0 = time.time()
    n_in = len(xyz)

    # Compute voxel index for every point: (N, 3) int32
    origin = xyz.min(axis=0)
    voxel_idx = np.floor((xyz - origin) / voxel_size).astype(np.int32)

    # Encode 3D voxel index as a single int64 for fast grouping
    # Use a large prime stride to avoid collisions
    strides = np.array([1, 100_003, 100_003 ** 2], dtype=np.int64)
    keys = (voxel_idx.astype(np.int64) * strides).sum(axis=1)

    # Sort by key to group points belonging to the same voxel
    sort_order = np.argsort(keys, kind="stable")
    keys_sorted = keys[sort_order]
    xyz_sorted = xyz[sort_order]

    # Find voxel boundaries
    _, first_occurrence = np.unique(keys_sorted, return_index=True)
    voxel_counts = np.diff(
        np.concatenate([first_occurrence, [len(keys_sorted)]])
    )
    n_voxels = len(first_occurrence)

    # ── Centroid xyz ──────────────────────────────────────────────────────
    xyz_down = np.zeros((n_voxels, 3), dtype=np.float32)
    for v, (start, count) in enumerate(zip(first_occurrence, voxel_counts)):
        xyz_down[v] = xyz_sorted[start : start + count].mean(axis=0)

    # ── Mean reflectance ──────────────────────────────────────────────────
    refl_down = None
    if reflectance is not None:
        refl_sorted = reflectance[sort_order]
        refl_down = np.zeros(n_voxels, dtype=np.float32)
        for v, (start, count) in enumerate(zip(first_occurrence, voxel_counts)):
            refl_down[v] = refl_sorted[start : start + count].mean()

    # ── Majority-vote label ───────────────────────────────────────────────
    labels_down = None
    if labels is not None:
        labels_sorted = labels[sort_order]
        labels_down = np.zeros(n_voxels, dtype=np.int32)
        for v, (start, count) in enumerate(zip(first_occurrence, voxel_counts)):
            block = labels_sorted[start : start + count]
            labels_down[v] = np.bincount(block).argmax()

    elapsed = time.time() - t0
    logger.info(
        f"  Voxel downsample: {n_in:,} → {n_voxels:,} pts "
        f"(ratio {n_voxels/n_in:.2%}, {elapsed:.1f}s)"
    )
    return xyz_down, refl_down, labels_down


def voxel_downsample_fast(
    xyz: np.ndarray,
    reflectance: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    voxel_size: float = 0.05,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Faster voxel downsampling using Open3D for geometry, then KNN for labels.

    Open3D's C++ implementation is ~5× faster than the pure-numpy version above
    for large clouds. We use it for xyz, then assign reflectance and labels to
    each downsampled point via nearest-neighbor lookup in the original cloud.

    Use this when processing full scans (>10M points). The numpy version above
    is used as fallback when Open3D is unavailable.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    t0 = time.time()
    n_in = len(xyz)

    # Downsample geometry with Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    xyz_down = np.asarray(pcd_down.points, dtype=np.float32)
    n_out = len(xyz_down)

    # KNN: for each downsampled point, find its nearest original neighbor
    tree = cKDTree(xyz)
    _, nn_idx = tree.query(xyz_down, k=1, workers=-1)

    refl_down = reflectance[nn_idx] if reflectance is not None else None
    labels_down = labels[nn_idx].astype(np.int32) if labels is not None else None

    elapsed = time.time() - t0
    logger.info(
        f"  Voxel downsample (fast): {n_in:,} → {n_out:,} pts "
        f"(ratio {n_out/n_in:.2%}, {elapsed:.1f}s)"
    )
    return xyz_down, refl_down, labels_down


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Normal estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_normals(
    xyz: np.ndarray,
    radius: float = 0.3,
    max_nn: int = 30,
) -> np.ndarray:
    """Estimate surface normals via PCA on local neighborhoods.

    Uses Open3D's KD-tree radius search. Normals are oriented consistently
    toward the positive Z axis (upward sensor assumption).

    The analogy to CT: normals encode local surface orientation, similar to
    the image gradient direction in edge-based segmentation — they help the
    model distinguish flat ground from vertical walls even when xyz is noisy.

    Parameters
    ----------
    xyz     : (N, 3) float32 — downsampled coordinates
    radius  : neighborhood radius in meters
    max_nn  : maximum neighbors in the radius ball

    Returns
    -------
    normals : (N, 3) float32 in [-1, 1]
    """
    import open3d as o3d

    t0 = time.time()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    # Orient normals consistently upward (Z+ = toward sky)
    pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])

    normals = np.asarray(pcd.normals, dtype=np.float32)

    elapsed = time.time() - t0
    logger.info(f"  Normal estimation: {len(xyz):,} pts, r={radius}m ({elapsed:.1f}s)")
    return normals


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Feature assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_features(
    xyz: np.ndarray,
    reflectance: Optional[np.ndarray],
    normals: Optional[np.ndarray],
) -> np.ndarray:
    """Stack per-point features into a single (N, C) array.

    Feature layout (C = 7):
        [0:3]  xyz           — raw coordinates (Lambert-93 meters)
        [3]    reflectance   — normalized [0, 1], or 0 if unavailable
        [4:7]  nx, ny, nz   — unit normal vector, or 0 if unavailable

    Note: height-above-ground is NOT computed here because it requires
    knowledge of the local block (z_min within a 4 m × 4 m window).
    It is added in the Dataset class at block-crop time.

    Returns
    -------
    features : (N, 7) float32
    """
    n = len(xyz)
    features = np.zeros((n, 7), dtype=np.float32)

    features[:, 0:3] = xyz

    if reflectance is not None:
        features[:, 3] = reflectance

    if normals is not None:
        features[:, 4:7] = normals

    return features


# ─────────────────────────────────────────────────────────────────────────────
#  Full pipeline for one file
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_file(
    input_path: str | Path,
    output_dir: str | Path,
    voxel_size: float = 0.05,
    normal_radius: float = 0.3,
    normal_max_nn: int = 30,
    use_fast_downsample: bool = True,
    overwrite: bool = False,
) -> dict:
    """Run the full preprocessing pipeline on one PLY file.

    Output files written to output_dir/<stem>/:
        points.npy  — (N, 7) float32  [x, y, z, reflectance, nx, ny, nz]
        labels.npy  — (N,)   int32    coarse class labels (0–9)
        stats.npz   — bounding box, point count, class counts (for DataLoader)

    Parameters
    ----------
    input_path          : path to a training_10_classes .ply file
    output_dir          : root output directory (e.g. data/processed/)
    voxel_size          : leaf size for voxel downsampling
    normal_radius       : radius for normal estimation
    normal_max_nn       : max neighbors for normal estimation
    use_fast_downsample : use Open3D + KNN (faster) vs pure numpy
    overwrite           : re-process even if output already exists

    Returns
    -------
    dict with keys: n_points, class_counts, elapsed_total
    """
    from src.data.loader import load_ply, NUM_CLASSES

    input_path = Path(input_path)
    output_dir = Path(output_dir) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    points_path = output_dir / "points.npy"
    labels_path = output_dir / "labels.npy"

    if points_path.exists() and labels_path.exists() and not overwrite:
        logger.info(f"  Skipping {input_path.name} (already processed, use --overwrite)")
        points = np.load(points_path)
        labels = np.load(labels_path)
        return {"n_points": len(points), "skipped": True}

    t_total = time.time()
    logger.info("─" * 60)
    logger.info(f"Processing: {input_path.name}")
    logger.info("─" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    pc = load_ply(input_path)
    xyz = pc.xyz
    reflectance = pc.reflectance
    labels = pc.labels

    # ── Downsample ────────────────────────────────────────────────────────
    fn_down = voxel_downsample_fast if use_fast_downsample else voxel_downsample
    xyz, reflectance, labels = fn_down(xyz, reflectance, labels, voxel_size)

    # ── Normals ───────────────────────────────────────────────────────────
    normals = estimate_normals(xyz, radius=normal_radius, max_nn=normal_max_nn)

    # ── Features ──────────────────────────────────────────────────────────
    points = assemble_features(xyz, reflectance, normals)

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(points_path, points)
    logger.info(f"  Saved points: {points_path} — shape {points.shape}")

    if labels is not None:
        np.save(labels_path, labels.astype(np.int32))
        logger.info(f"  Saved labels: {labels_path} — shape {labels.shape}")

    # Stats for the DataLoader (bounding box, class balance)
    class_counts = np.bincount(labels, minlength=NUM_CLASSES) if labels is not None else None
    stats = {
        "n_points": len(points),
        "xyz_min": xyz.min(axis=0),
        "xyz_max": xyz.max(axis=0),
        "voxel_size": voxel_size,
    }
    if class_counts is not None:
        stats["class_counts"] = class_counts
    np.savez(output_dir / "stats.npz", **stats)

    elapsed = time.time() - t_total
    logger.info(f"  Done in {elapsed:.1f}s  →  {output_dir}")

    return {
        "n_points": len(points),
        "class_counts": class_counts,
        "elapsed_total": elapsed,
        "skipped": False,
    }

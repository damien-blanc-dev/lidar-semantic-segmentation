"""
src/fusion/projection.py
─────────────────────────
Core camera–LiDAR geometry utilities.

Coordinate conventions (OpenCV / ROS camera standard):
  Camera frame : +X right, +Y down, +Z into scene (forward)
  Image frame  : origin top-left,  u = column (→),  v = row (↓)
  LiDAR frame  : defined by T_lidar_to_cam (sensor-specific)

All functions are pure NumPy — no OpenCV, Open3D, or scipy dependency.

Public API
──────────
  load_camera_calibration(path)
  transform_points(points_xyz, T)
  project_points(points_cam, K)
  filter_visible_points(uv, depth, image_size, min_depth)
  sample_colors(image, uv, mode)
  colorize_pointcloud(points_xyz, image, calib, ...)
  perturb_extrinsics(T, translation_noise_m, rotation_noise_deg, ...)
  export_colored_ply(path, xyz, rgb)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_camera_calibration(path: str | Path) -> dict:
    """Load camera intrinsics and LiDAR-to-camera extrinsics from JSON.

    JSON schema (see data/demo/calibration.json for a full annotated example):

        {
            "image_size"     : [W, H],
            "K"              : [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist_coeffs"    : [k1, k2, p1, p2, k3],   # optional
            "T_lidar_to_cam" : [[r00, r01, r02, tx],    # 4×4 SE(3)
                                 [r10, r11, r12, ty],
                                 [r20, r21, r22, tz],
                                 [  0,   0,   0,  1]]
        }

    Returns
    -------
    dict with keys:
        K               (3, 3) float64 — intrinsic matrix
        dist_coeffs     (5,)   float64 — radial/tangential distortion (zeros = none)
        T_lidar_to_cam  (4, 4) float64 — rigid transform LiDAR → camera frame
        image_size      (W, H) int     — image resolution
    """
    with open(Path(path)) as f:
        raw = json.load(f)

    K = np.array(raw["K"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3×3, got {K.shape}")

    T = np.array(raw["T_lidar_to_cam"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_lidar_to_cam must be 4×4, got {T.shape}")

    dist = np.array(raw.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    W, H = int(raw["image_size"][0]), int(raw["image_size"][1])

    return {"K": K, "dist_coeffs": dist, "T_lidar_to_cam": T, "image_size": (W, H)}


# ─────────────────────────────────────────────────────────────────────────────
#  Rigid-body transform
# ─────────────────────────────────────────────────────────────────────────────

def transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 SE(3) transform to an (N, 3) point array.

    Equivalent to  P_target = R @ P_source + t  for each point.

    Parameters
    ----------
    points_xyz : (N, 3) float — points in source frame
    T          : (4, 4) float — T_target_from_source

    Returns
    -------
    (N, 3) float — points in target (camera) frame
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3), got {points_xyz.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    return points_xyz @ R.T + t


# ─────────────────────────────────────────────────────────────────────────────
#  Perspective projection
# ─────────────────────────────────────────────────────────────────────────────

def project_points(
    points_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame 3-D points onto the image plane.

    Uses the standard pinhole model:
        u = fx · (X/Z) + cx
        v = fy · (Y/Z) + cy

    Note: distortion is NOT applied here. For significant lens distortion,
    apply undistortion to the image before calling this function (OpenCV
    cv2.undistort / cv2.initUndistortRectifyMap).

    Parameters
    ----------
    points_cam : (N, 3) float — points in camera frame (X right, Y down, Z fwd)
    K          : (3, 3) float — intrinsic matrix

    Returns
    -------
    uv    : (N, 2) float — pixel coordinates [u=col, v=row]
             Entries for points with Z ≤ 0 are numerically garbage;
             use filter_visible_points() to mask them out.
    depth : (N,)   float — Z coordinate in camera frame (positive = in front)
    """
    depth = points_cam[:, 2]
    safe_z = np.where(depth > 0, depth, 1.0)   # avoid divide-by-zero

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (points_cam[:, 0] / safe_z) + cx
    v = fy * (points_cam[:, 1] / safe_z) + cy

    return np.stack([u, v], axis=1), depth


def filter_visible_points(
    uv: np.ndarray,
    depth: np.ndarray,
    image_size: tuple[int, int],
    min_depth: float = 0.1,
) -> np.ndarray:
    """Return a boolean mask for points that project inside the image.

    Parameters
    ----------
    uv         : (N, 2) float  — projected pixel coordinates
    depth      : (N,)   float  — depth in camera frame
    image_size : (W, H)        — image width and height in pixels
    min_depth  : float         — minimum valid depth in metres (clips near noise)

    Returns
    -------
    mask : (N,) bool — True for points with valid, in-frame projections
    """
    W, H = image_size
    u, v = uv[:, 0], uv[:, 1]
    return (depth >= min_depth) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


# ─────────────────────────────────────────────────────────────────────────────
#  Color sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_colors(
    image: np.ndarray,
    uv: np.ndarray,
    mode: str = "bilinear",
) -> np.ndarray:
    """Sample RGB colors from an image at continuous pixel coordinates.

    Pure-NumPy implementation — no OpenCV or scipy required.

    Parameters
    ----------
    image : (H, W, 3) uint8 — RGB image
    uv    : (N, 2)   float  — pixel coordinates [u=col, v=row]
    mode  : 'nearest' | 'bilinear'

    Returns
    -------
    colors : (N, 3) uint8 — sampled RGB values
    """
    H, W = image.shape[:2]
    u, v = uv[:, 0], uv[:, 1]

    if mode == "nearest":
        ui = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, H - 1)
        return image[vi, ui].copy()

    # Bilinear interpolation — clamp to [0, W-2] × [0, H-2] so u1/v1 stay in bounds
    u0 = np.clip(np.floor(u).astype(np.int32), 0, W - 2)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H - 2)
    u1, v1 = u0 + 1, v0 + 1

    # Fractional weights — shape (N, 1) for broadcasting over RGB channels
    wu = (u - u0)[:, np.newaxis].astype(np.float32)
    wv = (v - v0)[:, np.newaxis].astype(np.float32)

    c00 = image[v0, u0].astype(np.float32)
    c10 = image[v0, u1].astype(np.float32)
    c01 = image[v1, u0].astype(np.float32)
    c11 = image[v1, u1].astype(np.float32)

    blended = (c00 * (1 - wu) * (1 - wv)
             + c10 * wu       * (1 - wv)
             + c01 * (1 - wu) * wv
             + c11 * wu       * wv)

    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  High-level colorization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def colorize_pointcloud(
    points_xyz: np.ndarray,
    image: np.ndarray,
    calib: dict,
    sample_mode: str = "bilinear",
    min_depth: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full colorization pipeline: transform → project → filter → sample.

    Parameters
    ----------
    points_xyz  : (N, 3) float — LiDAR points in sensor frame
    image       : (H, W, 3) uint8 — RGB image
    calib       : dict from load_camera_calibration()
    sample_mode : 'nearest' | 'bilinear'
    min_depth   : minimum valid depth in metres

    Returns
    -------
    colors       : (N, 3) uint8 — RGB for all N points; (0, 0, 0) for invisible
    visible_mask : (N,)   bool  — True for points with valid projection
    uv_all       : (N, 2) float — pixel coords for ALL points (invalid for invisible)
    depth_all    : (N,)   float — depth for ALL points (may be ≤ 0 for invisible)
    """
    pts_cam = transform_points(points_xyz, calib["T_lidar_to_cam"])
    uv, depth = project_points(pts_cam, calib["K"])
    visible = filter_visible_points(uv, depth, calib["image_size"], min_depth)

    colors = np.zeros((len(points_xyz), 3), dtype=np.uint8)
    if visible.any():
        colors[visible] = sample_colors(image, uv[visible], mode=sample_mode)

    return colors, visible, uv, depth


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration perturbation (sensitivity analysis)
# ─────────────────────────────────────────────────────────────────────────────

def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: (unit axis, angle) → 3×3 rotation matrix.

    R = I + sin(θ)·[axis]× + (1 − cos(θ))·[axis]×²
    where [axis]× is the skew-symmetric cross-product matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0    ],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def perturb_extrinsics(
    T: np.ndarray,
    translation_noise_m: float = 0.0,
    rotation_noise_deg: float = 0.0,
    trans_axis: Optional[np.ndarray] = None,
    rot_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a small SE(3) perturbation to T_lidar_to_cam.

    Models calibration estimation uncertainty. The perturbation is
    left-multiplied:  T_perturbed = ΔT · T_nominal

    This corresponds to an error in the *camera-side* frame estimate, which is
    the dominant error source in hand-eye / target-based calibration.

    Parameters
    ----------
    T                   : (4, 4) original T_lidar_to_cam
    translation_noise_m : shift magnitude along trans_axis in metres
    rotation_noise_deg  : rotation magnitude around rot_axis in degrees
    trans_axis          : (3,) unit vector in camera frame; default = X (lateral)
    rot_axis            : (3,) unit vector in camera frame; default = Y (yaw/pan)

    Returns
    -------
    T_perturbed : (4, 4) float — perturbed extrinsic transform
    """
    # Defaults: most sensitive axes for a forward-facing camera
    if trans_axis is None:
        trans_axis = np.array([1.0, 0.0, 0.0])   # camera X = lateral
    if rot_axis is None:
        rot_axis = np.array([0.0, 1.0, 0.0])      # camera Y = pan / yaw

    T_out = T.copy().astype(np.float64)

    if rotation_noise_deg != 0.0:
        R_noise = _rodrigues(np.asarray(rot_axis, float), np.radians(rotation_noise_deg))
        T_out[:3, :3] = R_noise @ T_out[:3, :3]
        T_out[:3, 3]  = R_noise @ T_out[:3, 3]

    if translation_noise_m != 0.0:
        T_out[:3, 3] += np.asarray(trans_axis, float) * translation_noise_m

    return T_out


# ─────────────────────────────────────────────────────────────────────────────
#  PLY export  (binary little-endian, no Open3D dependency)
# ─────────────────────────────────────────────────────────────────────────────

def export_colored_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> None:
    """Write a binary little-endian PLY with XYZ (float32) + RGB (uint8).

    Readable by CloudCompare, MeshLab, Open3D, and any standard PLY viewer.

    Parameters
    ----------
    path : output path (.ply)
    xyz  : (N, 3) float — 3-D coordinates (any unit)
    rgb  : (N, 3) uint8 — RGB colors in [0, 255]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(xyz)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    buf = np.empty(N, dtype=dtype)
    buf["x"]     = xyz[:, 0].astype(np.float32)
    buf["y"]     = xyz[:, 1].astype(np.float32)
    buf["z"]     = xyz[:, 2].astype(np.float32)
    buf["red"]   = rgb[:, 0].astype(np.uint8)
    buf["green"] = rgb[:, 1].astype(np.uint8)
    buf["blue"]  = rgb[:, 2].astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())

"""
src/data/loader.py
──────────────────
Load Paris-Lille-3D point clouds.

PLY header (training_10_classes):
    property float x
    property float y
    property float z
    property uchar reflectance   ← laser intensity, 0–255
    property int   class         ← coarse label, 0–9

We use `plyfile` (not Open3D) to read these files because Open3D's
`read_point_cloud` only exposes xyz+colors and ignores custom scalar fields
like `class`. plyfile reads the binary PLY exactly as structured.

Label mapping (10 coarse classes, already in the dataset files):
    0  unclassified
    1  ground
    2  building
    3  pole / road sign / traffic light
    4  bollard / small pole
    5  trash can
    6  barrier
    7  pedestrian
    8  car
    9  natural / vegetation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Class metadata  (10 coarse classes, Paris-Lille-3D convention)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "unclassified",     # 0
    "ground",           # 1
    "building",         # 2
    "pole/sign",        # 3
    "bollard",          # 4
    "trash can",        # 5
    "barrier",          # 6
    "pedestrian",       # 7
    "car",              # 8
    "vegetation",       # 9
]

NUM_CLASSES = len(CLASS_NAMES)

# RGB colors for visualization, one per class, float32 in [0, 1]
CLASS_COLORS = np.array([
    [105, 105, 105],   # 0 unclassified  — dim grey
    [139, 115,  85],   # 1 ground        — sandy brown
    [160, 160, 160],   # 2 building      — light grey
    [255, 215,   0],   # 3 pole/sign     — gold
    [255, 165,   0],   # 4 bollard       — orange
    [210, 105,  30],   # 5 trash can     — chocolate
    [139,  69,  19],   # 6 barrier       — saddle brown
    [255,  69,   0],   # 7 pedestrian    — orange-red
    [ 65, 105, 225],   # 8 car           — royal blue
    [ 34, 139,  34],   # 9 vegetation    — forest green
], dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
#  Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PointCloud:
    """Container for a loaded point cloud.

    Attributes
    ----------
    xyz        : (N, 3) float32 — 3D coordinates (meters, Lambert-93)
    reflectance: (N,)   float32 — laser intensity normalized to [0, 1], or None
    labels     : (N,)   int32   — coarse class label (0–9), or None if unlabeled
    path       : source file path
    """
    xyz: np.ndarray
    reflectance: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.xyz.ndim == 2 and self.xyz.shape[1] == 3, \
            f"xyz must be (N, 3), got {self.xyz.shape}"

    @property
    def num_points(self) -> int:
        return len(self.xyz)

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def __repr__(self) -> str:
        label_info = (
            f", classes={np.unique(self.labels).tolist()}"
            if self.has_labels else ""
        )
        return (
            f"PointCloud({self.num_points:,} pts"
            f"{label_info}"
            f", path={self.path.name if self.path else 'None'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ply(path: str | Path) -> PointCloud:
    """Load a Paris-Lille-3D PLY file using plyfile.

    Reads `x`, `y`, `z`, `reflectance`, and `class` scalar fields directly
    from the binary PLY. The `class` field contains the 10-class coarse labels
    as distributed in training_10_classes/.

    Parameters
    ----------
    path : path to the .ply file

    Returns
    -------
    PointCloud
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Install plyfile: pip install plyfile")

    path = Path(path)
    logger.info(f"Loading: {path.name} ...")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]

    xyz = np.stack([
        np.asarray(vertex["x"], dtype=np.float32),
        np.asarray(vertex["y"], dtype=np.float32),
        np.asarray(vertex["z"], dtype=np.float32),
    ], axis=1)

    # reflectance: uchar (0–255) → float32 [0, 1]
    reflectance = None
    if "reflectance" in vertex.data.dtype.names:
        raw = np.asarray(vertex["reflectance"], dtype=np.float32)
        reflectance = raw / 255.0

    # labels: the field is literally named "class" in the PLY
    labels = None
    if "class" in vertex.data.dtype.names:
        labels = np.asarray(vertex["class"], dtype=np.int32)

    logger.info(f"  {len(xyz):,} points loaded")
    if labels is not None:
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        for i, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
            if count > 0:
                logger.info(f"  [{i}] {name:<20} {count:>10,}  ({count/len(xyz)*100:5.1f}%)")

    return PointCloud(xyz=xyz, reflectance=reflectance, labels=labels, path=path)


def load_pointcloud(path: str | Path) -> PointCloud:
    """Dispatch loader by file extension. Supports .ply, .las, .laz."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".ply":
        return load_ply(path)
    elif ext in (".las", ".laz"):
        return _load_las(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def _load_las(path: Path) -> PointCloud:
    try:
        import laspy
    except ImportError:
        raise ImportError("Install laspy: pip install laspy lazrs")

    logger.info(f"Loading LAS/LAZ: {path.name} ...")
    with laspy.open(str(path)) as f:
        las = f.read()

    xyz = np.stack([
        np.asarray(las.x, dtype=np.float32),
        np.asarray(las.y, dtype=np.float32),
        np.asarray(las.z, dtype=np.float32),
    ], axis=1)

    reflectance = None
    if hasattr(las, "intensity"):
        raw = np.asarray(las.intensity, dtype=np.float32)
        reflectance = raw / raw.max() if raw.max() > 0 else raw

    labels = None
    if hasattr(las, "classification"):
        labels = np.asarray(las.classification, dtype=np.int32)

    logger.info(f"  {len(xyz):,} points loaded")
    return PointCloud(xyz=xyz, reflectance=reflectance, labels=labels, path=path)

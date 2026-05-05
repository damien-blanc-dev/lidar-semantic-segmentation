"""
src/data/dataset.py
────────────────────
PyTorch Dataset for Paris-Lille-3D.

Reads preprocessed .npy files (output of scripts/preprocess.py) and serves
fixed-size point blocks to the training loop.

Block cropping logic (done here, not in preprocessing):
  1. Pick a random point as block center
  2. Collect all points within a 4 m × 4 m XY window around it
  3. Compute height = z - z_block_min  (height above lowest point in block)
  4. Normalize XY within the block to [-1, 1]  (so the model sees local geometry)
  5. Sample exactly `num_points` points (random subsample or repeat-pad if sparse)

Final feature vector per point — shape (num_points, 8):
  [0]   x_norm       — X normalized within block [-1, 1]
  [1]   y_norm       — Y normalized within block [-1, 1]
  [2]   z            — raw Z (absolute height, Lambert-93)
  [3]   height       — Z - Z_block_min  (relative height above local ground)
  [4]   reflectance  — laser intensity [0, 1]
  [5]   nx           — normal X
  [6]   ny           — normal Y
  [7]   nz           — normal Z

Analogy to CT: this is equivalent to extracting a random patch from a volume
at each training step. The block size (4 m) is your receptive field equivalent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Feature indices in the preprocessed points.npy  (N, 7)
#   [x, y, z, reflectance, nx, ny, nz]
_IDX_XYZ   = slice(0, 3)
_IDX_REFL  = 3
_IDX_NORM  = slice(4, 7)


class PL3DDataset(Dataset):
    """Paris-Lille-3D Dataset.

    Parameters
    ----------
    processed_dir : path to data/processed/ root
    split_files   : list of scan stems to include, e.g. ["Lille1_1", "Lille1_2"]
    num_points    : points per block (default 4096)
    block_size    : spatial block side length in meters (default 4.0)
    min_points    : discard blocks with fewer points than this (default 512)
    augment       : apply random augmentation (train only)
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_files: list[str],
        num_points: int = 4096,
        block_size: float = 4.0,
        min_points: int = 512,
        augment: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.num_points = num_points
        self.block_size = block_size
        self.min_points = min_points
        self.augment = augment

        # Load all scans into memory (downsampled, so ~100-300 MB per scan)
        self.scans = []
        for stem in split_files:
            scan = self._load_scan(stem)
            if scan is not None:
                self.scans.append(scan)
                logger.info(
                    f"  Loaded {stem}: {scan['n_points']:,} pts"
                )

        if not self.scans:
            raise RuntimeError(
                f"No scans loaded from {processed_dir}. "
                "Run scripts/preprocess.py first."
            )

        # Pre-build a sampling index: for each epoch step, which scan + which
        # center point to use. We rebuild this at the start of each epoch.
        self._build_index(n_samples_per_scan=10_000)

    def _load_scan(self, stem: str) -> Optional[dict]:
        scan_dir = self.processed_dir / stem
        points_path = scan_dir / "points.npy"
        labels_path = scan_dir / "labels.npy"

        if not points_path.exists():
            logger.warning(f"  Missing points.npy for {stem}, skipping.")
            return None

        points = np.load(points_path)   # (N, 7): x,y,z,refl,nx,ny,nz
        labels = np.load(labels_path) if labels_path.exists() else None

        return {
            "stem": stem,
            "points": points,
            "labels": labels,
            "n_points": len(points),
            "xy": points[:, :2],  # keep a separate XY view for fast block lookup
        }

    def _build_index(self, n_samples_per_scan: int = 10_000):
        """Pre-compute (scan_idx, center_point_idx) pairs for __getitem__."""
        self.index = []
        for scan_idx, scan in enumerate(self.scans):
            n = scan["n_points"]
            sample_n = min(n_samples_per_scan, n)
            center_idxs = np.random.choice(n, sample_n, replace=False)
            for ci in center_idxs:
                self.index.append((scan_idx, ci))
        np.random.shuffle(self.index)
        logger.debug(f"Dataset index: {len(self.index):,} blocks")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        scan_idx, center_idx = self.index[idx]
        scan = self.scans[scan_idx]

        points = scan["points"]   # (N, 7)
        labels = scan["labels"]   # (N,)
        xy = scan["xy"]           # (N, 2)

        # ── Block cropping ────────────────────────────────────────────────
        center_xy = xy[center_idx]
        half = self.block_size / 2.0

        mask = (
            (xy[:, 0] >= center_xy[0] - half) & (xy[:, 0] < center_xy[0] + half) &
            (xy[:, 1] >= center_xy[1] - half) & (xy[:, 1] < center_xy[1] + half)
        )
        block_pts = points[mask]    # (M, 7)
        block_lbl = labels[mask]    # (M,)

        # Fallback: if block is too sparse, return a random block elsewhere
        if len(block_pts) < self.min_points:
            return self.__getitem__(np.random.randint(len(self)))

        # ── Sample / pad to num_points ────────────────────────────────────
        m = len(block_pts)
        if m >= self.num_points:
            chosen = np.random.choice(m, self.num_points, replace=False)
        else:
            # Repeat-pad: sample with replacement to reach num_points
            chosen = np.random.choice(m, self.num_points, replace=True)

        block_pts = block_pts[chosen]   # (num_points, 7)
        block_lbl = block_lbl[chosen]   # (num_points,)

        # ── Feature engineering (on the block) ───────────────────────────
        xyz   = block_pts[:, _IDX_XYZ]          # (P, 3)
        refl  = block_pts[:, _IDX_REFL]          # (P,)
        norms = block_pts[:, _IDX_NORM]          # (P, 3)

        # Height above local ground (5th percentile of Z in block, robust to noise)
        z_ground = np.percentile(xyz[:, 2], 5)
        height = (xyz[:, 2] - z_ground).clip(min=0).astype(np.float32)

        # Normalize XY to block-local coordinates [-1, 1]
        x_norm = ((xyz[:, 0] - center_xy[0]) / half).astype(np.float32)
        y_norm = ((xyz[:, 1] - center_xy[1]) / half).astype(np.float32)

        # Assemble final feature vector (P, 8)
        features = np.stack([
            x_norm,          # 0: local X
            y_norm,          # 1: local Y
            xyz[:, 2],       # 2: raw Z
            height,          # 3: height above local ground
            refl,            # 4: reflectance
            norms[:, 0],     # 5: nx
            norms[:, 1],     # 6: ny
            norms[:, 2],     # 7: nz
        ], axis=1).astype(np.float32)   # (P, 8)

        # ── Augmentation (training only) ──────────────────────────────────
        if self.augment:
            features = _augment(features)

        return (
            torch.from_numpy(features),         # (num_points, 8)
            torch.from_numpy(block_lbl.astype(np.int64)),  # (num_points,)
        )

    def on_epoch_end(self):
        """Call at the end of each epoch to reshuffle the block sampling index."""
        self._build_index()


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _augment(features: np.ndarray) -> np.ndarray:
    """Random augmentation on a single block.

    Applied only during training. Keeps augmentations physically plausible
    for outdoor LiDAR (no vertical flips, no scaling of height).

    Augmentations:
      - Random rotation around Z axis (full 360°)
      - Random jitter on XYZ (Gaussian noise σ=0.01 m)
      - Random reflectance drop (zero out reflectance with p=0.1)
    """
    features = features.copy()

    # Z-axis rotation (applies to x_norm, y_norm and normals nx, ny)
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x, y = features[:, 0].copy(), features[:, 1].copy()
    features[:, 0] = cos_t * x - sin_t * y
    features[:, 1] = sin_t * x + cos_t * y

    nx, ny = features[:, 5].copy(), features[:, 6].copy()
    features[:, 5] = cos_t * nx - sin_t * ny
    features[:, 6] = sin_t * nx + cos_t * ny

    # XYZ jitter
    features[:, 0:3] += np.random.normal(0, 0.01, size=(len(features), 3)).astype(np.float32)

    # Reflectance dropout
    if np.random.rand() < 0.1:
        features[:, 4] = 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str | Path,
    train_files: list[str],
    val_files: list[str],
    num_points: int = 4096,
    block_size: float = 4.0,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple:
    """Build train and validation DataLoaders.

    Returns
    -------
    (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = PL3DDataset(
        processed_dir=processed_dir,
        split_files=[Path(f).stem for f in train_files],
        num_points=num_points,
        block_size=block_size,
        augment=True,
    )
    val_dataset = PL3DDataset(
        processed_dir=processed_dir,
        split_files=[Path(f).stem for f in val_files],
        num_points=num_points,
        block_size=block_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready — "
        f"train: {len(train_dataset):,} blocks, "
        f"val: {len(val_dataset):,} blocks"
    )
    return train_loader, val_loader

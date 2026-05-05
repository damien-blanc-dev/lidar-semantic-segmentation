"""
src/models/pointnet2.py
────────────────────────
PointNet++ (SSG — Single Scale Grouping) for point cloud semantic segmentation.
Implementation from scratch with pure PyTorch (no torch-geometric dependency).

Architecture — encoder-decoder with skip connections:
┌─────────────────────────────────────────────────────────────┐
│  Input: (B, N=4096, C=8) features                           │
│                                                             │
│  ENCODER (Set Abstraction)                                  │
│    SA1: 4096 → 1024 pts  r=0.2  k=32  →  64 features       │
│    SA2: 1024 →  256 pts  r=0.4  k=32  → 128 features       │
│    SA3:  256 →   64 pts  r=0.8  k=32  → 256 features       │
│    SA4:   64 →    1 pt   global        → 512 features       │
│                                                             │
│  DECODER (Feature Propagation, KNN interpolation)          │
│    FP3:   1 →   64 pts  [512+256 → 256]                    │
│    FP2:  64 →  256 pts  [256+128 → 128]                    │
│    FP1: 256 → 1024 pts  [128+64  → 128]                    │
│    FP0: 1024→ 4096 pts  [128+C   → 128]                    │
│                                                             │
│  HEAD: MLP(128, 128) → Dropout → Linear(128, num_classes)  │
└─────────────────────────────────────────────────────────────┘

Analogy to CT: this is a 3D U-Net but on irregular points.
  SA layers = encoder with learned 3D pooling (instead of max-pool on voxels)
  FP layers = decoder with KNN interpolation (instead of transposed conv)
  Skip connections carry fine-grained local features to the decoder.

Coordinate convention (input features):
  pos  = features[:, [0, 1, 3]] → (x_norm, y_norm, height)  used for geometry
  feat = features[:, :]          → all 8 channels            used as descriptors
  Height is divided by 5.0 to bring it to roughly the same scale as x_norm/y_norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

def farthest_point_sample(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Select n_samples points via Farthest Point Sampling.

    Iteratively picks the point farthest from all already-selected points.
    This gives a spatially uniform subset, unlike random sampling.

    Parameters
    ----------
    xyz       : (B, N, 3)  input positions
    n_samples : number of points to select

    Returns
    -------
    idx : (B, n_samples) indices into the N dimension
    """
    B, N, _ = xyz.shape
    device = xyz.device

    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)

    # Random starting point for each batch element
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_samples):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B,1,3)
        d = ((xyz - centroid) ** 2).sum(dim=-1)  # (B, N)
        dist = torch.min(dist, d)
        farthest = dist.argmax(dim=-1)           # (B,)

    return idx


def ball_query(
    xyz: torch.Tensor,
    centroids: torch.Tensor,
    radius: float,
    max_samples: int,
) -> torch.Tensor:
    """Find up to max_samples neighbors within a sphere of given radius.

    For points with fewer than max_samples neighbors, the nearest point
    is repeated to maintain fixed tensor shape (standard PointNet++ trick).

    Parameters
    ----------
    xyz        : (B, N, 3)  all input points
    centroids  : (B, S, 3)  query centers
    radius     : float — sphere radius
    max_samples: max neighbors to return per centroid

    Returns
    -------
    group_idx : (B, S, max_samples) indices into xyz
    """
    B, N, _ = xyz.shape
    _, S, _ = centroids.shape
    device = xyz.device

    # Pairwise squared distances: (B, S, N)
    # Chunked computation to avoid OOM for large S and N
    chunk = 128  # process this many centroids at a time
    group_idx = torch.zeros(B, S, max_samples, dtype=torch.long, device=device)

    for s_start in range(0, S, chunk):
        s_end = min(s_start + chunk, S)
        c = centroids[:, s_start:s_end, :]           # (B, chunk, 3)
        dists = torch.cdist(c, xyz)                   # (B, chunk, N)

        sorted_dists, sorted_idx = dists.sort(dim=-1)
        nn_idx = sorted_idx[:, :, :max_samples]       # (B, chunk, max_samples)
        within = sorted_dists[:, :, :max_samples] <= radius  # bool

        # Repeat-pad: replace out-of-radius indices with the nearest point
        nearest = sorted_idx[:, :, :1].expand(-1, -1, max_samples)
        nn_idx = torch.where(within, nn_idx, nearest)
        group_idx[:, s_start:s_end, :] = nn_idx

    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Parameters
    ----------
    points : (B, N, C)
    idx    : (B, ...) long tensor — any shape of indices into N

    Returns
    -------
    (B, ..., C)
    """
    B = points.shape[0]
    *rest, _ = points.shape
    idx_expanded = idx.unsqueeze(-1).expand(*idx.shape, points.shape[-1])
    return points.gather(1, idx_expanded.view(B, -1, points.shape[-1])).view(
        B, *idx.shape[1:], points.shape[-1]
    )


def knn_interpolate(
    xyz_fine: torch.Tensor,
    xyz_coarse: torch.Tensor,
    feat_coarse: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Propagate features from a coarser set of points to a finer set via KNN.

    Uses inverse-distance weighting: closer points contribute more.
    This is the Feature Propagation step in PointNet++.

    Parameters
    ----------
    xyz_fine    : (B, N_fine,   3)  positions of the fine-grained points
    xyz_coarse  : (B, N_coarse, 3)  positions of the coarser points
    feat_coarse : (B, N_coarse, C)  features to propagate
    k           : number of neighbors for interpolation

    Returns
    -------
    (B, N_fine, C)  interpolated features
    """
    B, N_fine, _ = xyz_fine.shape
    _, N_coarse, C = feat_coarse.shape
    device = xyz_fine.device

    k = min(k, N_coarse)

    dists = torch.cdist(xyz_fine, xyz_coarse)           # (B, N_fine, N_coarse)
    knn_dists, knn_idx = dists.topk(k, dim=-1, largest=False)  # (B, N_fine, k)

    # Inverse distance weights (add small eps for stability)
    weights = 1.0 / (knn_dists + 1e-8)                  # (B, N_fine, k)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Gather k neighbor features and compute weighted sum
    knn_feat = index_points(feat_coarse, knn_idx)        # (B, N_fine, k, C)
    interpolated = (knn_feat * weights.unsqueeze(-1)).sum(dim=2)  # (B, N_fine, C)

    return interpolated


# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(channels: list[int], bn: bool = True) -> nn.Sequential:
    """Build a shared MLP (1×1 convolution on the feature axis) with BN + ReLU."""
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """One Set Abstraction layer (SSG).

    Samples centroids → groups neighbors → PointNet on each group → max pool.

    Parameters
    ----------
    n_centroids : number of output points (FPS)
    radius      : ball query radius (in normalized 3D coordinate space)
    max_samples : max neighbors per ball
    in_channels : number of input feature channels
    mlp_channels: list of hidden/output channel sizes for the PointNet MLP
    """

    def __init__(
        self,
        n_centroids: int,
        radius: float,
        max_samples: int,
        in_channels: int,
        mlp_channels: list[int],
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.radius = radius
        self.max_samples = max_samples

        # Input to the MLP = grouped input features + relative xyz offsets (3 extra)
        self.mlp = build_mlp([in_channels + 3] + mlp_channels)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz      : (B, N, 3) point positions
        features : (B, N, C) per-point features

        Returns
        -------
        new_xyz      : (B, n_centroids, 3)
        new_features : (B, n_centroids, mlp_channels[-1])
        """
        B, N, _ = xyz.shape

        # 1 — Farthest Point Sampling
        fps_idx = farthest_point_sample(xyz, self.n_centroids)   # (B, S)
        new_xyz = index_points(xyz, fps_idx)                      # (B, S, 3)

        # 2 — Ball query
        group_idx = ball_query(xyz, new_xyz, self.radius, self.max_samples)
        # (B, S, K)

        # 3 — Gather features and encode relative positions
        grouped_xyz = index_points(xyz, group_idx)          # (B, S, K, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)                 # relative offsets
        grouped_feat = index_points(features, group_idx)    # (B, S, K, C)

        # Concatenate relative xyz + features → (B, S, K, 3+C)
        x = torch.cat([grouped_xyz, grouped_feat], dim=-1)

        # 4 — Shared MLP + max pool across neighborhood
        # Conv1d expects (B, C, N) → reshape to (B*S, 3+C, K)
        B, S, K, D = x.shape
        x = x.view(B * S, K, D).permute(0, 2, 1)   # (B*S, D, K)
        x = self.mlp(x)                              # (B*S, C_out, K)
        x = x.max(dim=-1).values                     # (B*S, C_out)
        new_features = x.view(B, S, -1)              # (B, S, C_out)

        return new_xyz, new_features


class GlobalAbstraction(nn.Module):
    """Set Abstraction with global pooling (no sampling/grouping).
    Used as the bottleneck layer — aggregates everything into a single vector.
    """

    def __init__(self, in_channels: int, mlp_channels: list[int]):
        super().__init__()
        self.mlp = build_mlp([in_channels + 3] + mlp_channels)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        # Concatenate absolute xyz + features
        x = torch.cat([xyz, features], dim=-1)             # (B, N, 3+C)
        x = x.permute(0, 2, 1)                             # (B, 3+C, N)
        x = self.mlp(x)                                    # (B, C_out, N)
        x = x.max(dim=-1).values.unsqueeze(1)              # (B, 1, C_out)
        xyz_out = torch.zeros(B, 1, 3, device=xyz.device)
        return xyz_out, x


class FeaturePropagation(nn.Module):
    """Feature Propagation layer: upsample from coarse to fine via KNN interpolation.

    1. Interpolate coarse features onto fine point positions
    2. Concatenate with skip connection from the encoder
    3. Apply shared MLP
    """

    def __init__(self, in_channels: int, mlp_channels: list[int]):
        super().__init__()
        self.mlp = build_mlp([in_channels] + mlp_channels)

    def forward(
        self,
        xyz_fine: torch.Tensor,
        xyz_coarse: torch.Tensor,
        feat_fine: torch.Tensor,
        feat_coarse: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz_fine    : (B, N_fine,   3)  — fine level xyz (from encoder skip)
        xyz_coarse  : (B, N_coarse, 3)  — coarse level xyz
        feat_fine   : (B, N_fine,   C1) — skip connection features
        feat_coarse : (B, N_coarse, C2) — features to upsample

        Returns
        -------
        (B, N_fine, mlp_channels[-1])
        """
        # Interpolate coarse features → fine resolution
        interp = knn_interpolate(xyz_fine, xyz_coarse, feat_coarse, k=3)

        # Concatenate with skip connection
        x = torch.cat([feat_fine, interp], dim=-1)   # (B, N_fine, C1+C2)

        # Shared MLP
        x = x.permute(0, 2, 1)                       # (B, C1+C2, N_fine)
        x = self.mlp(x)                               # (B, C_out, N_fine)
        x = x.permute(0, 2, 1)                        # (B, N_fine, C_out)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Full PointNet++ model
# ─────────────────────────────────────────────────────────────────────────────

class PointNet2(nn.Module):
    """PointNet++ SSG for semantic segmentation.

    Parameters
    ----------
    in_channels : number of input features per point (default 8)
    num_classes : number of output semantic classes (default 10)
    dropout     : dropout rate before the final linear layer

    Input
    -----
    features : (B, N, in_channels) — full feature tensor from the Dataset
               pos is extracted internally as features[:, :, [0, 1, 3]]
               (x_norm, y_norm, height/5.0)
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels

        # ── Encoder ───────────────────────────────────────────────────────
        self.sa1 = SetAbstraction(
            n_centroids=1024, radius=0.2, max_samples=32,
            in_channels=in_channels, mlp_channels=[32, 32, 64],
        )
        self.sa2 = SetAbstraction(
            n_centroids=256, radius=0.4, max_samples=32,
            in_channels=64, mlp_channels=[64, 64, 128],
        )
        self.sa3 = SetAbstraction(
            n_centroids=64, radius=0.8, max_samples=32,
            in_channels=128, mlp_channels=[128, 128, 256],
        )
        self.sa4 = GlobalAbstraction(
            in_channels=256, mlp_channels=[256, 512, 512],
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.fp3 = FeaturePropagation(
            in_channels=256 + 512, mlp_channels=[256, 256],
        )
        self.fp2 = FeaturePropagation(
            in_channels=128 + 256, mlp_channels=[128, 128],
        )
        self.fp1 = FeaturePropagation(
            in_channels=64 + 128, mlp_channels=[128, 128],
        )
        self.fp0 = FeaturePropagation(
            in_channels=in_channels + 128, mlp_channels=[128, 128],
        )

        # ── Segmentation head ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def _extract_pos(self, features: torch.Tensor) -> torch.Tensor:
        """Extract 3D position from the feature tensor.

        Uses [x_norm, y_norm, height/5.0] — all three in roughly [-1, 2] range.
        """
        pos = features[:, :, [0, 1, 3]].clone()
        pos[:, :, 2] = pos[:, :, 2] / 5.0   # scale height to match xy
        return pos

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, N, C)

        Returns
        -------
        logits : (B, N, num_classes)
        """
        xyz0 = self._extract_pos(features)   # (B, N, 3) — positions for geometry
        feat0 = features                     # (B, N, C) — full features

        # ── Encode ────────────────────────────────────────────────────────
        xyz1, feat1 = self.sa1(xyz0, feat0)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        xyz4, feat4 = self.sa4(xyz3, feat3)

        # ── Decode (with skip connections) ────────────────────────────────
        feat3_up = self.fp3(xyz3, xyz4, feat3, feat4)
        feat2_up = self.fp2(xyz2, xyz3, feat2, feat3_up)
        feat1_up = self.fp1(xyz1, xyz2, feat1, feat2_up)
        feat0_up = self.fp0(xyz0, xyz1, feat0, feat1_up)   # (B, N, 128)

        # ── Head ──────────────────────────────────────────────────────────
        x = feat0_up.permute(0, 2, 1)   # (B, 128, N)
        logits = self.head(x)            # (B, num_classes, N)
        logits = logits.permute(0, 2, 1) # (B, N, num_classes)

        return logits


def build_model(cfg: dict) -> PointNet2:
    """Instantiate model from config dict."""
    return PointNet2(
        in_channels=cfg.get("in_channels", 8),
        num_classes=cfg.get("num_classes", 10),
        dropout=cfg.get("dropout", 0.5),
    )

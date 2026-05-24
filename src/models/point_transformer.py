"""
src/models/point_transformer.py
─────────────────────────────────
Point Transformer: Self-Attention for Point Cloud Learning
(Zhao et al., ICCV 2021 — https://arxiv.org/abs/2012.09164)

Key innovations over PointNet++:
  - Vector self-attention (per-channel weights, not scalar)
  - Subtracted position encoding (relative geometry awareness)
  - Attention computed locally in KNN neighborhoods

Architecture (adapted for Paris-Lille-3D):
  Input embedding: MLP(C → 32)

  Encoder (TransitionDown — FPS + KNN + attention pooling):
    PT Block(32)  → TransitionDown  N→N/4    32→64
    PT Block(64)  → TransitionDown  N/4→N/16 64→128
    PT Block(128) → TransitionDown  N/16→N/64 128→256
    PT Block(256) → TransitionDown  N/64→N/256 256→512

  Decoder (TransitionUp — KNN interpolation + skip):
    TransitionUp 512+256→256 → PT Block(256)
    TransitionUp 256+128→128 → PT Block(128)
    TransitionUp 128+64→64   → PT Block(64)
    TransitionUp 64+32→32    → PT Block(32)

  Head: MLP(32→64→num_classes)

Input:  (B, N, C)
Output: (B, N, num_classes)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import knn_points, index_points, knn_interpolate


# ─────────────────────────────────────────────────────────────────────────────
#  Point Transformer Layer
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformerLayer(nn.Module):
    """Vector self-attention within a KNN neighborhood.

    For each query point i with neighbors {j}:
        y_i = sum_j  softmax( φ(x_i) - ψ(x_j) + δ_ij ) ⊙ (α(x_j) + δ_ij)

    where:
        φ, ψ, α : linear projections (query/key/value)
        δ_ij    : position encoding MLP applied to (p_i - p_j)
        ⊙       : element-wise product (vector attention)

    Parameters
    ----------
    channels : feature dimension (same in and out — residual block)
    k        : neighborhood size
    """

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k = k

        self.linear_q = nn.Linear(channels, channels, bias=False)
        self.linear_k = nn.Linear(channels, channels, bias=False)
        self.linear_v = nn.Linear(channels, channels, bias=False)

        # Position encoding: relative xyz (3-dim) → channels
        self.pos_enc = nn.Sequential(
            nn.Linear(3, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
        )

        # Attention weight MLP (after subtraction + pos encoding)
        self.attn_mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
        )

        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        xyz: torch.Tensor,      # (B, N, 3)
        features: torch.Tensor, # (B, N, C)
    ) -> torch.Tensor:          # (B, N, C)
        B, N, C = features.shape

        _, idx = knn_points(xyz, xyz, k=self.k)    # (B, N, k)
        neighbor_xyz  = index_points(xyz, idx)      # (B, N, k, 3)
        neighbor_feat = index_points(features, idx) # (B, N, k, C)

        # Relative position encoding — δ_ij
        xyz_exp = xyz.unsqueeze(2).expand_as(neighbor_xyz)   # (B, N, k, 3)
        rel_pos = (xyz_exp - neighbor_xyz).reshape(B * N * self.k, 3)
        delta   = self.pos_enc(rel_pos).reshape(B, N, self.k, C)  # (B, N, k, C)

        # Query / key / value projections
        q = self.linear_q(features).unsqueeze(2)              # (B, N, 1, C)
        k = self.linear_k(neighbor_feat)                      # (B, N, k, C)
        v = self.linear_v(neighbor_feat)                      # (B, N, k, C)

        # Attention weights: φ(x_i) - ψ(x_j) + δ_ij
        attn_in = (q - k + delta).reshape(B * N * self.k, C)
        attn    = self.attn_mlp(attn_in).reshape(B, N, self.k, C)
        attn    = F.softmax(attn, dim=2)                      # (B, N, k, C)

        # Aggregate: element-wise with value + position encoding
        out = (attn * (v + delta)).sum(dim=2)                 # (B, N, C)

        # Residual + BN + ReLU
        out = self.act(self.bn(out.reshape(B * N, C)).reshape(B, N, C) + features)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  TransitionDown  (FPS + KNN pooling)
# ─────────────────────────────────────────────────────────────────────────────

def _farthest_point_sample(xyz: torch.Tensor, n_out: int) -> torch.Tensor:
    """FPS via torch_cluster.fps — CUDA-native kernel, ~10× faster than iterative.

    Returns (B, n_out) local indices into xyz.
    """
    from torch_cluster import fps as tc_fps

    B, N, _ = xyz.shape
    xyz_flat  = xyz.reshape(B * N, 3)
    batch_vec = torch.arange(B, device=xyz.device).repeat_interleave(N)
    ratio     = n_out / N

    sel_global = tc_fps(xyz_flat, batch_vec, ratio=ratio, random_start=True)
    offsets    = torch.arange(B, device=xyz.device).repeat_interleave(n_out) * N
    return (sel_global - offsets).reshape(B, n_out)


class TransitionDown(nn.Module):
    """Downsample via FPS, then pool KNN neighborhood with max + MLP.

    in_channels  → out_channels,  N → N // stride
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 4, k: int = 16):
        super().__init__()
        self.stride = stride
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz: torch.Tensor,      # (B, N, 3)
        features: torch.Tensor, # (B, N, C_in)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = features.shape
        n_out = max(1, N // self.stride)

        # FPS centroids
        fps_idx    = _farthest_point_sample(xyz, n_out)        # (B, n_out)
        xyz_down   = index_points(xyz, fps_idx)                 # (B, n_out, 3)

        # KNN around each centroid
        _, knn_idx = knn_points(xyz_down, xyz, k=self.k)       # (B, n_out, k)
        knn_feat   = index_points(features, knn_idx)            # (B, n_out, k, C)

        # MLP + max pooling over k
        x = knn_feat.permute(0, 3, 1, 2)                       # (B, C, n_out, k)
        x = self.mlp(x)                                         # (B, C_out, n_out, k)
        x = x.max(dim=-1).values                                # (B, C_out, n_out)
        feat_down = x.permute(0, 2, 1)                         # (B, n_out, C_out)

        return xyz_down, feat_down


# ─────────────────────────────────────────────────────────────────────────────
#  TransitionUp  (interpolation + linear projection)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionUp(nn.Module):
    """Upsample via KNN interpolation, concat skip, project.

    (in_channels + skip_channels) → out_channels
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz_fine:    torch.Tensor,   # (B, N_fine,   3)
        xyz_coarse:  torch.Tensor,   # (B, N_coarse, 3)
        feat_coarse: torch.Tensor,   # (B, N_coarse, in_channels)
        feat_skip:   torch.Tensor,   # (B, N_fine,   skip_channels)
    ) -> torch.Tensor:               # (B, N_fine, out_channels)
        interp = knn_interpolate(xyz_fine, xyz_coarse, feat_coarse, k=3)
        x = torch.cat([interp, feat_skip], dim=-1)
        return self.mlp(x.permute(0, 2, 1)).permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Point Transformer
# ─────────────────────────────────────────────────────────────────────────────

class PointTransformer(nn.Module):
    """Point Transformer for semantic segmentation of point clouds.

    Parameters
    ----------
    in_channels  : input feature dimensions
    num_classes  : number of output classes
    k            : KNN neighborhood size (default 16)
    dropout      : dropout in head
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        k: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Input embedding
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Encoder PT blocks + TransitionDowns
        self.pt1  = PointTransformerLayer(32,  k)
        self.td1  = TransitionDown(32,  64,  stride=4, k=k)

        self.pt2  = PointTransformerLayer(64,  k)
        self.td2  = TransitionDown(64,  128, stride=4, k=k)

        self.pt3  = PointTransformerLayer(128, k)
        self.td3  = TransitionDown(128, 256, stride=4, k=k)

        self.pt4  = PointTransformerLayer(256, k)
        self.td4  = TransitionDown(256, 512, stride=4, k=k)

        # Bottleneck PT block
        self.pt5  = PointTransformerLayer(512, k)

        # Decoder TransitionUps + PT blocks
        self.tu4  = TransitionUp(512, 256, 256)
        self.pt_d4 = PointTransformerLayer(256, k)

        self.tu3  = TransitionUp(256, 128, 128)
        self.pt_d3 = PointTransformerLayer(128, k)

        self.tu2  = TransitionUp(128,  64,  64)
        self.pt_d2 = PointTransformerLayer(64, k)

        self.tu1  = TransitionUp(64,  32,  32)
        self.pt_d1 = PointTransformerLayer(32, k)

        # Classification head
        self.head = nn.Sequential(
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1),
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, N, 32) with BatchNorm applied per-point."""
        B, N, C = x.shape
        out = self.input_mlp[0](x.reshape(B * N, C))
        out = self.input_mlp[1](out)
        out = self.input_mlp[2](out)
        return out.reshape(B, N, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, C)

        Returns
        -------
        (B, N, num_classes)
        """
        xyz0 = x[:, :, :3]   # use first 3 feature dims as geometry proxy

        # ── Input embedding ───────────────────────────────────────────────
        f0 = self._embed(x)                          # (B, N, 32)

        # ── Encoder ──────────────────────────────────────────────────────
        f0 = self.pt1(xyz0, f0)
        xyz1, f1 = self.td1(xyz0, f0)
        f1 = self.pt2(xyz1, f1)

        xyz2, f2 = self.td2(xyz1, f1)
        f2 = self.pt3(xyz2, f2)

        xyz3, f3 = self.td3(xyz2, f2)
        f3 = self.pt4(xyz3, f3)

        xyz4, f4 = self.td4(xyz3, f3)
        f4 = self.pt5(xyz4, f4)

        # ── Decoder ──────────────────────────────────────────────────────
        d3 = self.tu4(xyz3, xyz4, f4, f3)
        d3 = self.pt_d4(xyz3, d3)

        d2 = self.tu3(xyz2, xyz3, d3, f2)
        d2 = self.pt_d3(xyz2, d2)

        d1 = self.tu2(xyz1, xyz2, d2, f1)
        d1 = self.pt_d2(xyz1, d1)

        d0 = self.tu1(xyz0, xyz1, d1, f0)
        d0 = self.pt_d1(xyz0, d0)

        # ── Head ──────────────────────────────────────────────────────────
        out = self.head(d0.permute(0, 2, 1))         # (B, num_classes, N)
        return out.permute(0, 2, 1)                  # (B, N, num_classes)

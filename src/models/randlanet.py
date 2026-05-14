"""
src/models/randlanet.py
────────────────────────
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
(Hu et al., CVPR 2020 — https://arxiv.org/abs/1911.11236)

Key differences from PointNet++:
  - Random sampling instead of FPS  → O(1) vs O(N²) sampling
  - Local Spatial Encoding (LocSE)  → explicit relative geometry encoding
  - Attentive Pooling               → learned aggregation weights
  - 4× downsampling per stage       → very aggressive, compensated by LFA

Architecture (matching original paper for Paris-Lille-3D scale):
  Encoder:
    LFA(16→32)  + downsample N→N/4
    LFA(32→64)  + downsample N/4→N/16
    LFA(64→128) + downsample N/16→N/64
    LFA(128→256)+ downsample N/64→N/256

  Decoder (KNN interpolation + skip connections):
    UP: N/256→N/64  concat 256+256→128
    UP: N/64→N/16   concat 128+128→64
    UP: N/16→N/4    concat 64+64→32
    UP: N/4→N       concat 32+32→16

  Head: FC(16→64) → FC(64→num_classes)

Input:  (B, N, C)   — same format as PointNet++
Output: (B, N, num_classes)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import knn_points, index_points, random_downsample, knn_interpolate


# ─────────────────────────────────────────────────────────────────────────────
#  Local Spatial Encoding
# ─────────────────────────────────────────────────────────────────────────────

class LocalSpatialEncoding(nn.Module):
    """Encode relative positions of k-NN neighbors into a feature vector.

    For each query point p_i with neighbor p_j, builds a 10-dim descriptor:
        [p_i | p_j | (p_i - p_j) | ||p_i - p_j||]
    then lifts it to `out_channels` with a shared MLP.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        # 10 geometric dims + in_channels per neighbor
        self.mlp = nn.Sequential(
            nn.Conv2d(10 + in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz: torch.Tensor,      # (B, N, 3)
        features: torch.Tensor, # (B, N, C)
    ) -> torch.Tensor:          # (B, N, k, out_channels)
        B, N, C = features.shape

        _, idx = knn_points(xyz, xyz, k=self.k)    # (B, N, k)
        neighbor_xyz  = index_points(xyz, idx)      # (B, N, k, 3)
        neighbor_feat = index_points(features, idx) # (B, N, k, C)

        # Relative geometry: 10 dims
        xyz_tiled = xyz.unsqueeze(2).expand_as(neighbor_xyz)   # (B, N, k, 3)
        diff      = xyz_tiled - neighbor_xyz                    # (B, N, k, 3)
        dist      = diff.norm(dim=-1, keepdim=True)             # (B, N, k, 1)
        geom      = torch.cat([xyz_tiled, neighbor_xyz, diff, dist], dim=-1)  # (B, N, k, 10)

        # Concat with neighbor features
        x = torch.cat([geom, neighbor_feat], dim=-1)            # (B, N, k, 10+C)

        # Apply shared MLP via Conv2d (treats k as width, N as height)
        x = x.permute(0, 3, 1, 2)                              # (B, 10+C, N, k)
        x = self.mlp(x)                                         # (B, out_channels, N, k)
        x = x.permute(0, 2, 3, 1)                              # (B, N, k, out_channels)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Attentive Pooling
# ─────────────────────────────────────────────────────────────────────────────

class AttentivePooling(nn.Module):
    """Learn attention weights over k neighbors, then aggregate.

    Input : (B, N, k, C)
    Output: (B, N, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Score function: shared MLP → softmax over k
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Softmax(dim=-1),   # over k neighbors
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, k, C)
        y = x.permute(0, 3, 1, 2)          # (B, C, N, k)
        scores = self.score_fn(y)           # (B, C, N, k) — softmax over k
        agg = (y * scores).sum(dim=-1)      # (B, C, N)
        out = self.mlp(agg)                 # (B, out_channels, N)
        return out.permute(0, 2, 1)         # (B, N, out_channels)


# ─────────────────────────────────────────────────────────────────────────────
#  Local Feature Aggregation
# ─────────────────────────────────────────────────────────────────────────────

class LocalFeatureAggregation(nn.Module):
    """Two stacked (LocSE + AttentivePooling) blocks with a residual shortcut.

    This is the core building block of RandLA-Net. Doubles the feature
    dimensionality: in_channels → out_channels.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        mid = out_channels // 2

        self.lse1 = LocalSpatialEncoding(in_channels, mid, k)
        self.att1 = AttentivePooling(mid, mid)

        self.lse2 = LocalSpatialEncoding(mid, out_channels, k)
        self.att2 = AttentivePooling(out_channels, out_channels)

        # Shortcut projection
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        xyz: torch.Tensor,      # (B, N, 3)
        features: torch.Tensor, # (B, N, C_in)
    ) -> torch.Tensor:          # (B, N, C_out)
        shortcut = self.shortcut(features.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.lse1(xyz, features)   # (B, N, k, mid)
        x = self.att1(x)               # (B, N, mid)

        x = self.lse2(xyz, x)          # (B, N, k, out)
        x = self.att2(x)               # (B, N, out)

        return self.lrelu(x + shortcut)


# ─────────────────────────────────────────────────────────────────────────────
#  RandLA-Net
# ─────────────────────────────────────────────────────────────────────────────

class RandLANet(nn.Module):
    """RandLA-Net for point cloud semantic segmentation.

    Parameters
    ----------
    in_channels  : input feature dimensions (8 with normals, 5 without)
    num_classes  : number of output classes
    k            : number of KNN neighbors in LocSE blocks (default 16)
    decimation   : subsampling factor per encoder stage (default 4)
    dropout      : dropout rate in the classification head
    """

    def __init__(
        self,
        in_channels: int = 8,
        num_classes: int = 10,
        k: int = 16,
        decimation: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.decimation = decimation

        # Input projection: lift to 32-dim before any LFA
        self.input_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Encoder — each stage: LFA then random downsample (/4)
        self.enc1 = LocalFeatureAggregation(32,  64,  k)   # N   → N/4
        self.enc2 = LocalFeatureAggregation(64,  128, k)   # N/4 → N/16
        self.enc3 = LocalFeatureAggregation(128, 256, k)   # N/16→ N/64
        self.enc4 = LocalFeatureAggregation(256, 512, k)   # N/64→ N/256

        # Decoder — KNN interpolation + skip concat + MLP
        self.dec4 = self._make_decoder_mlp(512 + 256, 256)
        self.dec3 = self._make_decoder_mlp(256 + 128, 128)
        self.dec2 = self._make_decoder_mlp(128 + 64,  64)
        self.dec1 = self._make_decoder_mlp(64  + 32,  32)

        # Classification head
        self.head = nn.Sequential(
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1),
        )

    @staticmethod
    def _make_decoder_mlp(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, C) — point features (same convention as PointNet++)

        Returns
        -------
        (B, N, num_classes)
        """
        # Extract XYZ from features for geometric operations
        xyz0 = x[:, :, :3]   # (B, N, 3) — x_norm, y_norm, z_raw as proxy geometry

        # Input projection
        feat0 = self.input_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, 32)

        # ── Encoder ──────────────────────────────────────────────────────
        # Stage 1
        feat1 = self.enc1(xyz0, feat0)                                  # (B, N, 64)
        xyz1, feat1, idx1 = random_downsample(xyz0, feat1, xyz0.shape[1] // self.decimation)

        # Stage 2
        feat2 = self.enc2(xyz1, feat1)                                  # (B, N/4, 128)
        xyz2, feat2, idx2 = random_downsample(xyz1, feat2, xyz1.shape[1] // self.decimation)

        # Stage 3
        feat3 = self.enc3(xyz2, feat2)                                  # (B, N/16, 256)
        xyz3, feat3, idx3 = random_downsample(xyz2, feat3, xyz2.shape[1] // self.decimation)

        # Stage 4
        feat4 = self.enc4(xyz3, feat3)                                  # (B, N/64, 512)
        xyz4, feat4, _   = random_downsample(xyz3, feat4, xyz3.shape[1] // self.decimation)

        # ── Decoder (KNN interpolation + skip connection) ─────────────────
        # UP 4: N/256 → N/64
        up4 = knn_interpolate(xyz3, xyz4, feat4, k=3)                  # (B, N/64, 512)
        up4 = torch.cat([up4, feat3], dim=-1)                           # (B, N/64, 512+256)
        up4 = self.dec4(up4.permute(0, 2, 1)).permute(0, 2, 1)        # (B, N/64, 256)

        # UP 3: N/64 → N/16
        up3 = knn_interpolate(xyz2, xyz3, up4, k=3)                    # (B, N/16, 256)
        up3 = torch.cat([up3, feat2], dim=-1)                           # (B, N/16, 256+128)
        up3 = self.dec3(up3.permute(0, 2, 1)).permute(0, 2, 1)        # (B, N/16, 128)

        # UP 2: N/16 → N/4
        up2 = knn_interpolate(xyz1, xyz2, up3, k=3)                    # (B, N/4, 128)
        up2 = torch.cat([up2, feat1], dim=-1)                           # (B, N/4, 128+64)
        up2 = self.dec2(up2.permute(0, 2, 1)).permute(0, 2, 1)        # (B, N/4, 64)

        # UP 1: N/4 → N
        up1 = knn_interpolate(xyz0, xyz1, up2, k=3)                    # (B, N, 64)
        up1 = torch.cat([up1, feat0], dim=-1)                           # (B, N, 64+32)
        up1 = self.dec1(up1.permute(0, 2, 1)).permute(0, 2, 1)        # (B, N, 32)

        # ── Head ──────────────────────────────────────────────────────────
        out = self.head(up1.permute(0, 2, 1))                          # (B, num_classes, N)
        return out.permute(0, 2, 1)                                     # (B, N, num_classes)

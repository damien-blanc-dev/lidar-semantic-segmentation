"""
src/models/utils.py
────────────────────
Shared geometric primitives used by all point cloud models.

Extracted from pointnet2.py to avoid duplication across RandLA-Net and
PointTransformer. All functions operate on batched tensors (B, N, ...).
"""

from __future__ import annotations
import torch
import torch.nn as nn


def knn_points(
    query: torch.Tensor,
    source: torch.Tensor,
    k: int,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-Nearest Neighbors with chunked distance computation to avoid OOM.

    Parameters
    ----------
    query  : (B, M, 3) — query positions
    source : (B, N, 3) — source positions to search in
    k      : number of neighbors
    chunk_size : process this many queries at a time

    Returns
    -------
    dists : (B, M, k) squared distances
    idx   : (B, M, k) indices into source
    """
    B, M, _ = query.shape
    _, N, _ = source.shape
    device  = query.device
    k = min(k, N)

    all_dists = torch.zeros(B, M, k, device=device)
    all_idx   = torch.zeros(B, M, k, dtype=torch.long, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        q   = query[:, start:end, :]             # (B, chunk, 3)
        d   = torch.cdist(q, source)             # (B, chunk, N)
        top_d, top_idx = d.topk(k, dim=-1, largest=False)
        all_dists[:, start:end] = top_d
        all_idx[:, start:end]   = top_idx

    return all_dists, all_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Parameters
    ----------
    points : (B, N, C)
    idx    : (B, ...) long indices into N

    Returns
    -------
    (B, ..., C)
    """
    B, N, C = points.shape
    idx_flat = idx.reshape(B, -1)                          # (B, M)
    gathered = points.gather(
        1, idx_flat.unsqueeze(-1).expand(-1, -1, C)        # (B, M, C)
    )
    return gathered.reshape(B, *idx.shape[1:], C)


def random_downsample(
    xyz: torch.Tensor,
    features: torch.Tensor,
    n_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly subsample to n_out points (independent per batch element).

    Returns
    -------
    xyz_sub  : (B, n_out, 3)
    feat_sub : (B, n_out, C)
    idx      : (B, n_out) — sampled indices (for upsampling path)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    n_out = min(n_out, N)

    idx = torch.stack([
        torch.randperm(N, device=device)[:n_out] for _ in range(B)
    ])                                                      # (B, n_out)

    xyz_sub  = index_points(xyz, idx)
    feat_sub = index_points(features, idx)
    return xyz_sub, feat_sub, idx


def knn_interpolate(
    xyz_fine: torch.Tensor,
    xyz_coarse: torch.Tensor,
    feat_coarse: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Inverse-distance weighted KNN interpolation (shared with PointNet++).

    Parameters
    ----------
    xyz_fine    : (B, N_fine,   3)
    xyz_coarse  : (B, N_coarse, 3)
    feat_coarse : (B, N_coarse, C)

    Returns
    -------
    (B, N_fine, C)
    """
    dists, idx = knn_points(xyz_fine, xyz_coarse, k=k)   # (B, N_fine, k)
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    neighbor_feats = index_points(feat_coarse, idx)        # (B, N_fine, k, C)
    interpolated   = (neighbor_feats * weights.unsqueeze(-1)).sum(dim=2)
    return interpolated


def build_shared_mlp(channels: list[int], bn: bool = True) -> nn.Sequential:
    """Conv1d shared MLP with optional BN + ReLU."""
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i+1], 1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm1d(channels[i+1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

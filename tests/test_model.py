"""
tests/test_model.py
───────────────────
Smoke tests: verify tensor shapes end-to-end without loading real data.
Run with: python tests/test_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.pointnet2 import PointNet2, farthest_point_sample, ball_query


def test_fps():
    xyz = torch.randn(2, 4096, 3)
    idx = farthest_point_sample(xyz, 1024)
    assert idx.shape == (2, 1024), f"FPS shape error: {idx.shape}"
    print("  ✓ FPS: (2, 4096, 3) → (2, 1024)")


def test_ball_query():
    xyz = torch.randn(2, 4096, 3)
    centroids = xyz[:, :1024, :]
    idx = ball_query(xyz, centroids, radius=0.2, max_samples=32)
    assert idx.shape == (2, 1024, 32), f"Ball query shape error: {idx.shape}"
    print("  ✓ Ball query: (2, 1024, 32)")


def test_model_forward():
    model = PointNet2(in_channels=8, num_classes=10, dropout=0.0)
    model.eval()

    B, N, C = 2, 4096, 8
    features = torch.randn(B, N, C)

    with torch.no_grad():
        logits = model(features)

    assert logits.shape == (B, N, 10), f"Model output shape error: {logits.shape}"
    print(f"  ✓ Forward pass: ({B}, {N}, {C}) → ({B}, {N}, 10)")


def test_model_params():
    model = PointNet2(in_channels=8, num_classes=10)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parameters: {n:,}")
    assert n > 100_000, "Model seems too small"


if __name__ == "__main__":
    print("Running model smoke tests ...\n")
    test_fps()
    test_ball_query()
    test_model_forward()
    test_model_params()
    print("\nAll tests passed.")

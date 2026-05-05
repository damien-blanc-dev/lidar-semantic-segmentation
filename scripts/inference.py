"""
scripts/inference.py
─────────────────────
Run inference on a preprocessed scan and export visualizations.

Slides a 4m × 4m block window over the full scan, runs the model on each block,
and merges predictions by majority vote (a point can fall in multiple blocks).

Usage:
    python scripts/inference.py --scan Paris
    python scripts/inference.py --scan Lille1_1 --checkpoint outputs/checkpoints/pointnet2_ssg_pl3d/best.pth
    python scripts/inference.py --scan Paris --save_ply          # export colored PLY
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
CHECKPOINT    = Path("outputs/checkpoints/pointnet2_ssg_pl3d/best.pth")
FIGURE_DIR    = Path("outputs/figures")


def parse_args():
    p = argparse.ArgumentParser(description="PointNet++ inference on Paris-Lille-3D")
    p.add_argument("--scan",       type=str, default="Paris",
                   help="Scan stem to run inference on (default: Paris)")
    p.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    p.add_argument("--processed_dir", type=str, default=str(PROCESSED_DIR))
    p.add_argument("--block_size", type=float, default=4.0)
    p.add_argument("--num_points", type=int,   default=4096)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--stride",     type=float, default=2.0,
                   help="Sliding window stride in meters (default: 2.0)")
    p.add_argument("--save_ply",   action="store_true",
                   help="Export a colored PLY file for CloudCompare")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding-window inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    points: "np.ndarray",       # (N, 7)
    labels: "np.ndarray",       # (N,)
    model,
    device,
    block_size: float = 4.0,
    stride: float = 2.0,
    num_points: int = 4096,
    batch_size: int = 32,
) -> "np.ndarray":
    """Sliding-window inference → per-point predicted labels (N,).

    Each point accumulates votes from all blocks it falls into.
    Final prediction = class with the most votes.

    This is the LiDAR equivalent of sliding-window inference on CT volumes.
    """
    import numpy as np
    import torch

    N = len(points)
    xyz    = points[:, :3]          # x, y, z
    xy     = points[:, :2]          # x, y  (for block lookup)
    half   = block_size / 2.0

    # Build grid of block centers covering the full scan XY extent
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    centers_x = np.arange(x_min + half, x_max, stride)
    centers_y = np.arange(y_min + half, y_max, stride)
    grid = [(cx, cy) for cx in centers_x for cy in centers_y]
    logger.info(f"  {len(grid):,} blocks to process (stride={stride}m)")

    from src.data.dataset import _augment
    from src.data.loader import NUM_CLASSES

    # Accumulators
    vote_counts  = np.zeros((N, NUM_CLASSES), dtype=np.int32)

    # Collect blocks into batches
    batch_feats  = []
    batch_masks  = []

    def flush_batch():
        """Run model on accumulated batch, update vote_counts."""
        if not batch_feats:
            return
        feat_tensor = torch.from_numpy(
            np.stack(batch_feats, axis=0)
        ).to(device)                                    # (B, num_points, 8)

        with torch.no_grad():
            logits = model(feat_tensor)                 # (B, num_points, C)
            preds  = logits.argmax(dim=-1).cpu().numpy()  # (B, num_points)

        for pred, mask_idx in zip(preds, batch_masks):
            # pred shape: (num_points,) — indices back into the full scan
            np.add.at(vote_counts, (mask_idx,), np.eye(NUM_CLASSES, dtype=np.int32)[pred])

        batch_feats.clear()
        batch_masks.clear()

    model.eval()
    processed = 0

    for cx, cy in grid:
        mask = (
            (xy[:, 0] >= cx - half) & (xy[:, 0] < cx + half) &
            (xy[:, 1] >= cy - half) & (xy[:, 1] < cy + half)
        )
        block_pts = points[mask]
        m = mask.sum()

        if m < 64:   # skip nearly-empty blocks
            continue

        # Sample / pad to num_points
        if m >= num_points:
            chosen = np.random.choice(m, num_points, replace=False)
        else:
            chosen = np.random.choice(m, num_points, replace=True)

        block = block_pts[chosen]                       # (num_points, 7)
        orig_idx = np.where(mask)[0][chosen]            # original point indices

        # Feature engineering (same as Dataset)
        z_ground = np.percentile(block[:, 2], 5)
        height   = (block[:, 2] - z_ground).clip(min=0).astype(np.float32)
        x_norm   = ((block[:, 0] - cx) / half).astype(np.float32)
        y_norm   = ((block[:, 1] - cy) / half).astype(np.float32)

        feat = np.stack([
            x_norm, y_norm, block[:, 2],
            height, block[:, 3],
            block[:, 4], block[:, 5], block[:, 6],
        ], axis=1).astype(np.float32)                   # (num_points, 8)

        batch_feats.append(feat)
        batch_masks.append(orig_idx)

        if len(batch_feats) >= batch_size:
            flush_batch()

        processed += 1
        if processed % 500 == 0:
            logger.info(f"  {processed}/{len(grid)} blocks done ...")

    flush_batch()  # remaining blocks

    # Final prediction = argmax of vote counts
    pred_labels = vote_counts.argmax(axis=1).astype(np.int32)

    # Points with no votes (not covered by any block) → unclassified = 0
    no_vote = vote_counts.sum(axis=1) == 0
    pred_labels[no_vote] = 0

    return pred_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_colored_ply(
    xyz: "np.ndarray",
    labels: "np.ndarray",
    save_path: Path,
) -> None:
    """Write a colored PLY file viewable in CloudCompare."""
    import numpy as np
    from src.data.loader import CLASS_COLORS

    colors_uint8 = (CLASS_COLORS[labels] * 255).astype(np.uint8)

    with open(save_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

    with open(save_path, "ab") as f:
        data = np.hstack([xyz, colors_uint8]).astype(object)
        np.savetxt(f, data, fmt=["%f", "%f", "%f", "%d", "%d", "%d"])

    logger.info(f"  Colored PLY saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import numpy as np
    import torch

    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    logger.info(f"Loaded checkpoint — epoch {ckpt['epoch']}, mIoU {ckpt['miou']*100:.2f}%")

    from src.models.pointnet2 import PointNet2
    from src.data.loader import CLASS_NAMES, NUM_CLASSES

    model = PointNet2(in_channels=8, num_classes=NUM_CLASSES, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # ── Load scan ─────────────────────────────────────────────────────────
    scan_dir = Path(args.processed_dir) / args.scan
    points   = np.load(scan_dir / "points.npy")   # (N, 7)
    labels   = np.load(scan_dir / "labels.npy")   # (N,)
    logger.info(f"Scan: {args.scan}  —  {len(points):,} points")

    # ── Inference ─────────────────────────────────────────────────────────
    logger.info("Running sliding-window inference ...")
    pred_labels = run_inference(
        points, labels, model, device,
        block_size=args.block_size,
        stride=args.stride,
        num_points=args.num_points,
        batch_size=args.batch_size,
    )

    # ── Save predictions (for error analysis) ────────────────────────────
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{args.scan}_pred.npy", pred_labels)
    np.save(pred_dir / f"{args.scan}_gt.npy",   labels)
    logger.info(f"Predictions saved → {pred_dir}/{args.scan}_pred.npy")

    # ── Metrics ───────────────────────────────────────────────────────────
    from src.training.metrics import MetricTracker
    tracker = MetricTracker(NUM_CLASSES, CLASS_NAMES, ignore_index=0)
    tracker.update(pred_labels, labels)
    metrics = tracker.compute()
    logger.info("Results on full scan:")
    logger.info(tracker.log_str(metrics))

    # ── Figures ───────────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    xyz = points[:, :3]

    from src.visualization.visualizer import plot_topdown, plot_class_distribution

    # Subsample for plotting (12M points is too many for matplotlib)
    n_plot = min(500_000, len(xyz))
    idx_plot = np.random.choice(len(xyz), n_plot, replace=False)

    # Build fake PointCloud objects for the visualizer
    from src.data.loader import PointCloud

    pc_gt   = PointCloud(xyz=xyz[idx_plot], labels=labels[idx_plot])
    pc_pred = PointCloud(xyz=xyz[idx_plot], labels=pred_labels[idx_plot])

    logger.info("Generating figures ...")

    plot_topdown(pc_gt,   mode="labels", subsample=n_plot,
                 save_path=FIGURE_DIR / f"{args.scan}_gt.png")
    plot_topdown(pc_pred, mode="labels", subsample=n_plot,
                 save_path=FIGURE_DIR / f"{args.scan}_pred.png")
    plot_class_distribution(pc_pred,
                 save_path=FIGURE_DIR / f"{args.scan}_pred_distribution.png")

    logger.info(f"Figures saved to {FIGURE_DIR}/")

    # ── Optional PLY export ───────────────────────────────────────────────
    if args.save_ply:
        logger.info("Exporting colored PLY (this may take a few minutes for large scans) ...")
        export_colored_ply(
            xyz, pred_labels,
            FIGURE_DIR / f"{args.scan}_predictions.ply",
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

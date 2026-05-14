"""
scripts/inference.py
─────────────────────
Run inference on a preprocessed scan and export visualizations.

Slides a 4m × 4m block window over the full scan, runs the model on each block,
and merges predictions by majority vote (a point can fall in multiple blocks).

Usage:
    python scripts/inference.py --scan Paris
    python scripts/inference.py --scan Paris --save_probs           # also save softmax probs for Exp 5
    python scripts/inference.py --scan Paris --model randlanet \\
        --checkpoint outputs/checkpoints/exp4_randlanet/best.pth
    python scripts/inference.py --scan Paris --save_ply             # export colored PLY
"""

import argparse
import logging
import sys
import time
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
CHECKPOINT    = Path("outputs/checkpoints/pointnet2_pl3d/best.pth")
FIGURE_DIR    = Path("outputs/figures")
MODELS        = ["pointnet2", "randlanet", "point_transformer"]


def parse_args():
    p = argparse.ArgumentParser(description="Inference on Paris-Lille-3D")
    p.add_argument("--scan",        type=str, default="Paris")
    p.add_argument("--checkpoint",  type=str, default=str(CHECKPOINT))
    p.add_argument("--processed_dir", type=str, default=str(PROCESSED_DIR))
    p.add_argument("--block_size",  type=float, default=4.0)
    p.add_argument("--num_points",  type=int,   default=4096)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--stride",      type=float, default=2.0,
                   help="Sliding window stride in meters (default: 2.0)")
    p.add_argument("--model",       type=str,   default="pointnet2", choices=MODELS,
                   help="Architecture — auto-detected from checkpoint when possible")
    p.add_argument("--save_ply",    action="store_true",
                   help="Export a colored PLY file for CloudCompare")
    p.add_argument("--save_probs",  action="store_true",
                   help="Save per-point softmax probabilities to *_probs.npy (Exp 5)")
    p.add_argument("--experiment",  type=str, default=None,
                   help="Experiment name to log inference_time_s into results.csv")
    p.add_argument("--device",      type=str, default="auto")
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
    save_probs: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Sliding-window inference → per-point predicted labels (N,).

    Each point accumulates votes from all blocks it falls into.
    Final prediction = class with the most votes.

    When save_probs=True, also returns averaged softmax probabilities (N, C).
    Softmax probs are averaged (not voted) across overlapping windows, which
    gives a calibrated confidence signal for uncertainty analysis (Exp 5).
    """
    import numpy as np
    import torch

    N = len(points)
    xy   = points[:, :2]
    half = block_size / 2.0

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    centers_x = np.arange(x_min + half, x_max, stride)
    centers_y = np.arange(y_min + half, y_max, stride)
    grid = [(cx, cy) for cx in centers_x for cy in centers_y]
    logger.info(f"  {len(grid):,} blocks to process (stride={stride}m)")

    from src.data.loader import NUM_CLASSES

    vote_counts  = np.zeros((N, NUM_CLASSES), dtype=np.int32)
    prob_accum   = np.zeros((N, NUM_CLASSES), dtype=np.float32) if save_probs else None
    visit_counts = np.zeros(N, dtype=np.int32)                  if save_probs else None

    batch_feats: list = []
    batch_masks: list = []

    def flush_batch():
        if not batch_feats:
            return
        feat_tensor = torch.from_numpy(np.stack(batch_feats, axis=0)).to(device)

        with torch.no_grad():
            logits = model(feat_tensor)                              # (B, P, C)
            preds  = logits.argmax(dim=-1).cpu().numpy()            # (B, P)
            if save_probs:
                probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, P, C)

        for i, (pred, mask_idx) in enumerate(zip(preds, batch_masks)):
            np.add.at(vote_counts, (mask_idx,), np.eye(NUM_CLASSES, dtype=np.int32)[pred])
            if save_probs:
                np.add.at(prob_accum, (mask_idx,), probs_batch[i])
                np.add.at(visit_counts, mask_idx, 1)

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

        if m < 64:
            continue

        if m >= num_points:
            chosen = np.random.choice(m, num_points, replace=False)
        else:
            chosen = np.random.choice(m, num_points, replace=True)

        block     = block_pts[chosen]
        orig_idx  = np.where(mask)[0][chosen]

        # Feature engineering (mirrors PL3DDataset.__getitem__)
        z_ground = np.percentile(block[:, 2], 5)
        height   = (block[:, 2] - z_ground).clip(min=0).astype(np.float32)
        x_norm   = ((block[:, 0] - cx) / half).astype(np.float32)
        y_norm   = ((block[:, 1] - cy) / half).astype(np.float32)

        feat = np.stack([
            x_norm, y_norm, block[:, 2],
            height, block[:, 3],
            block[:, 4], block[:, 5], block[:, 6],
        ], axis=1).astype(np.float32)

        batch_feats.append(feat)
        batch_masks.append(orig_idx)

        if len(batch_feats) >= batch_size:
            flush_batch()

        processed += 1
        if processed % 500 == 0:
            logger.info(f"  {processed}/{len(grid)} blocks done ...")

    flush_batch()

    pred_labels = vote_counts.argmax(axis=1).astype(np.int32)
    no_vote     = vote_counts.sum(axis=1) == 0
    pred_labels[no_vote] = 0

    if save_probs:
        safe = np.maximum(visit_counts, 1).astype(np.float32)[:, np.newaxis]
        avg_probs = prob_accum / safe
        avg_probs[no_vote] = 0.0
        return pred_labels, avg_probs

    return pred_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_colored_ply(xyz: "np.ndarray", labels: "np.ndarray", save_path: Path) -> None:
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint — epoch {ckpt['epoch']}, mIoU {ckpt['miou']*100:.2f}%")

    # ── Model selection ───────────────────────────────────────────────────
    # Prefer architecture recorded in the checkpoint cfg; fall back to --model
    model_name = args.model
    if "cfg" in ckpt and "model" in ckpt["cfg"]:
        model_name = ckpt["cfg"]["model"]
        logger.info(f"  Architecture (from checkpoint): {model_name}")

    from src.data.loader import CLASS_NAMES, NUM_CLASSES

    if model_name == "pointnet2":
        from src.models.pointnet2 import PointNet2
        model = PointNet2(in_channels=8, num_classes=NUM_CLASSES, dropout=0.0)
    elif model_name == "randlanet":
        from src.models.randlanet import RandLANet
        model = RandLANet(in_channels=8, num_classes=NUM_CLASSES, dropout=0.0)
    elif model_name == "point_transformer":
        from src.models.point_transformer import PointTransformer
        model = PointTransformer(in_channels=8, num_classes=NUM_CLASSES, dropout=0.0)
    else:
        logger.warning(f"Unknown model '{model_name}', defaulting to PointNet2")
        from src.models.pointnet2 import PointNet2
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
    t_inf = time.time()
    result = run_inference(
        points, labels, model, device,
        block_size=args.block_size,
        stride=args.stride,
        num_points=args.num_points,
        batch_size=args.batch_size,
        save_probs=args.save_probs,
    )
    inference_time_s = time.time() - t_inf
    logger.info(f"  Inference done in {inference_time_s:.0f}s")

    if args.save_probs:
        pred_labels, probs = result
    else:
        pred_labels = result

    # ── Save predictions ──────────────────────────────────────────────────
    pred_dir = Path("outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{args.scan}_pred.npy", pred_labels)
    np.save(pred_dir / f"{args.scan}_gt.npy",   labels)
    logger.info(f"Predictions saved → {pred_dir}/{args.scan}_pred.npy")

    if args.save_probs:
        np.save(pred_dir / f"{args.scan}_probs.npy", probs)
        logger.info(f"Softmax probs saved → {pred_dir}/{args.scan}_probs.npy  shape={probs.shape}")

    # ── Metrics ───────────────────────────────────────────────────────────
    from src.training.metrics import MetricTracker
    tracker = MetricTracker(NUM_CLASSES, CLASS_NAMES, ignore_index=0)
    tracker.update(pred_labels, labels)
    metrics = tracker.compute()
    logger.info("Results on full scan:")
    logger.info(tracker.log_str(metrics))

    # ── Log inference time into results.csv (optional) ───────────────────
    if args.experiment:
        from src.utils.results_logger import log_result
        log_result({
            "experiment":       args.experiment,
            "variant":          "inference",
            "model":            model_name,
            "mIoU":             round(metrics["miou"] * 100, 2),
            "OA":               round(metrics["overall_acc"] * 100, 2),
            "inference_time_s": round(inference_time_s, 1),
        })
        logger.info("Inference result logged → outputs/results.csv")

    # ── Figures ───────────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    xyz = points[:, :3]

    from src.visualization.visualizer import plot_topdown, plot_class_distribution
    from src.data.loader import PointCloud

    n_plot  = min(500_000, len(xyz))
    idx_plt = np.random.choice(len(xyz), n_plot, replace=False)

    pc_gt   = PointCloud(xyz=xyz[idx_plt], labels=labels[idx_plt])
    pc_pred = PointCloud(xyz=xyz[idx_plt], labels=pred_labels[idx_plt])

    logger.info("Generating figures ...")
    plot_topdown(pc_gt,   mode="labels", subsample=n_plot,
                 save_path=FIGURE_DIR / f"{args.scan}_gt.png")
    plot_topdown(pc_pred, mode="labels", subsample=n_plot,
                 save_path=FIGURE_DIR / f"{args.scan}_pred.png")
    plot_class_distribution(pc_pred,
                 save_path=FIGURE_DIR / f"{args.scan}_pred_distribution.png")

    logger.info(f"Figures saved to {FIGURE_DIR}/")

    if args.save_ply:
        logger.info("Exporting colored PLY ...")
        export_colored_ply(xyz, pred_labels, FIGURE_DIR / f"{args.scan}_predictions.ply")

    logger.info("Done.")


if __name__ == "__main__":
    main()

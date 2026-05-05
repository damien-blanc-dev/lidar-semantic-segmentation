"""
scripts/train.py
─────────────────
Launch PointNet++ training on Paris-Lille-3D.

Usage:
    python scripts/train.py                          # default config
    python scripts/train.py --epochs 50              # override one param
    python scripts/train.py --batch_size 8           # for smaller GPU
    python scripts/train.py --device cpu             # CPU-only debug

Monitor training:
    tensorboard --logdir outputs/logs
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet++ on Paris-Lille-3D")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--num_points",    type=int,   default=4096)
    parser.add_argument("--block_size",    type=float, default=4.0)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="auto",
                        help="cuda | cpu | auto")
    parser.add_argument("--experiment",    type=str,   default="pointnet2_ssg_pl3d")
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    import torch, numpy as np, random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Config ────────────────────────────────────────────────────────────
    from src.data.loader import CLASS_NAMES, NUM_CLASSES

    TRAIN_STEMS = ["Lille1_1", "Lille1_2", "Lille2"]
    VAL_STEMS   = ["Paris"]

    cfg = {
        # Data
        "processed_dir":    args.processed_dir,
        "train_stems":      TRAIN_STEMS,
        "val_stems":        VAL_STEMS,
        "num_points":       args.num_points,
        "block_size":       args.block_size,
        # Model
        "in_channels":      8,
        "num_classes":      NUM_CLASSES,
        "dropout":          0.5,
        # Training
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "learning_rate":    args.lr,
        "weight_decay":     1e-4,
        "early_stopping_patience": 15,
        "ignore_index":     0,
        # Hardware
        "device":           device,
        "num_workers":      args.num_workers,
        # Outputs
        "experiment_name":  args.experiment,
        "checkpoint_dir":   "outputs/checkpoints",
        "log_dir":          "outputs/logs",
    }

    # ── Class weights ─────────────────────────────────────────────────────
    from src.training.trainer import compute_class_weights
    class_weights = compute_class_weights(
        processed_dir=args.processed_dir,
        train_stems=TRAIN_STEMS,
        num_classes=NUM_CLASSES,
        ignore_index=0,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    from src.data.dataset import PL3DDataset
    from torch.utils.data import DataLoader

    logger.info("Building datasets ...")
    train_dataset = PL3DDataset(
        processed_dir=args.processed_dir,
        split_files=TRAIN_STEMS,
        num_points=args.num_points,
        block_size=args.block_size,
        augment=True,
    )
    val_dataset = PL3DDataset(
        processed_dir=args.processed_dir,
        split_files=VAL_STEMS,
        num_points=args.num_points,
        block_size=args.block_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    logger.info(f"  Train: {len(train_dataset):,} blocks  |  Val: {len(val_dataset):,} blocks")

    # ── Model ─────────────────────────────────────────────────────────────
    from src.models.pointnet2 import PointNet2

    model = PointNet2(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: PointNet2-SSG  |  Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_names=CLASS_NAMES,
        class_weights=class_weights,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

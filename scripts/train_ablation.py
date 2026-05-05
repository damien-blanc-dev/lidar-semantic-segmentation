"""
scripts/train_ablation.py
──────────────────────────
Train ablation variants to measure feature importance.

Variants:
  --variant no_normals      : features = [x, y, z, height, reflectance]  (5 dims)
  --variant no_weighted_loss: standard CE, no class weighting
  --variant no_augmentation : no rotation / jitter / dropout
  --variant baseline        : all three disabled simultaneously

Results feed into the comparison table in the README.

Usage:
    python scripts/train_ablation.py --variant no_normals
    python scripts/train_ablation.py --variant no_weighted_loss
    python scripts/train_ablation.py --variant no_augmentation
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

VARIANTS = ["no_normals", "no_weighted_loss", "no_augmentation", "baseline"]


def parse_args():
    p = argparse.ArgumentParser(description="Ablation study training")
    p.add_argument("--variant",      type=str, required=True, choices=VARIANTS)
    p.add_argument("--processed_dir",type=str, default="data/processed")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    import torch
    import numpy as np
    import random

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Variant config ────────────────────────────────────────────────────
    use_normals       = args.variant not in ("no_normals", "baseline")
    use_weighted_loss = args.variant not in ("no_weighted_loss", "baseline")
    use_augmentation  = args.variant not in ("no_augmentation", "baseline")

    in_channels = 5 if not use_normals else 8
    exp_name    = f"pointnet2_ssg_{args.variant}"

    logger.info(f"Ablation variant : {args.variant}")
    logger.info(f"  use_normals       = {use_normals}   → in_channels={in_channels}")
    logger.info(f"  use_weighted_loss = {use_weighted_loss}")
    logger.info(f"  use_augmentation  = {use_augmentation}")

    from src.data.loader import CLASS_NAMES, NUM_CLASSES
    from src.data.dataset import PL3DDataset
    from src.models.pointnet2 import PointNet2
    from src.training.trainer import Trainer, compute_class_weights
    from torch.utils.data import DataLoader

    TRAIN_STEMS = ["Lille1_1", "Lille1_2", "Lille2"]
    VAL_STEMS   = ["Paris"]

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = PL3DDataset(
        processed_dir=args.processed_dir,
        split_files=TRAIN_STEMS,
        num_points=4096, block_size=4.0,
        augment=use_augmentation,
        use_normals=use_normals,
    )
    val_dataset = PL3DDataset(
        processed_dir=args.processed_dir,
        split_files=VAL_STEMS,
        num_points=4096, block_size=4.0,
        augment=False,
        use_normals=use_normals,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))

    # ── Class weights ─────────────────────────────────────────────────────
    class_weights = None
    if use_weighted_loss:
        class_weights = compute_class_weights(
            args.processed_dir, TRAIN_STEMS, NUM_CLASSES, ignore_index=0
        )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PointNet2(in_channels=in_channels, num_classes=NUM_CLASSES, dropout=0.5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    cfg = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
        "ignore_index": 0,
        "experiment_name": exp_name,
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_names=CLASS_NAMES,
        class_weights=class_weights,
    )
    trainer.fit()

    logger.info(f"Best mIoU ({args.variant}): {trainer.best_miou*100:.2f}%")


if __name__ == "__main__":
    main()

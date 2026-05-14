"""
scripts/train.py
─────────────────
Launch training on Paris-Lille-3D with a selectable model architecture.

Usage:
    python scripts/train.py                                    # PointNet++ (default)
    python scripts/train.py --model randlanet                  # RandLA-Net
    python scripts/train.py --model point_transformer          # PointTransformer
    python scripts/train.py --model pointnet2 --epochs 50      # override one param
    python scripts/train.py --batch_size 8                     # for smaller GPU
    python scripts/train.py --device cpu                       # CPU-only debug

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


MODELS = ["pointnet2", "randlanet", "point_transformer"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a point cloud model on Paris-Lille-3D")
    parser.add_argument("--model",         type=str,   default="pointnet2", choices=MODELS,
                        help="Model architecture to train")
    parser.add_argument("--processed_dir", type=str,   default="data/processed")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--num_points",    type=int,   default=4096)
    parser.add_argument("--block_size",    type=float, default=4.0)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="auto",
                        help="cuda | cpu | auto")
    parser.add_argument("--experiment",    type=str,   default=None,
                        help="Experiment name (default: <model>_pl3d)")
    parser.add_argument("--resume",        type=str,   default=None,
                        help="Path to checkpoint to resume from (e.g. outputs/checkpoints/randlanet_pl3d/best.pth)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--loss",          type=str,   default="weighted_ce",
                        choices=["ce", "weighted_ce", "focal", "cb_focal"],
                        help="Loss function (default: weighted_ce)")
    parser.add_argument("--normal_radius", type=float, default=0.3,
                        help="(metadata) Normal estimation radius used in preprocessing — logged to results.csv only")
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
    if args.experiment is None:
        args.experiment = f"{args.model}_pl3d"

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
        # Experiment metadata (persisted in checkpoint cfg)
        "model":            args.model,
        "loss":             args.loss,
    }

    # ── Class weights ─────────────────────────────────────────────────────
    from src.training.trainer import compute_class_weights, compute_cb_weights
    class_weights = compute_class_weights(
        processed_dir=args.processed_dir,
        train_stems=TRAIN_STEMS,
        num_classes=NUM_CLASSES,
        ignore_index=0,
    )
    cb_weights = None
    if args.loss == "cb_focal":
        cb_weights = compute_cb_weights(
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
    if args.model == "pointnet2":
        from src.models.pointnet2 import PointNet2
        model = PointNet2(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        model_label = "PointNet2-SSG"
    elif args.model == "randlanet":
        from src.models.randlanet import RandLANet
        model = RandLANet(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        model_label = "RandLA-Net"
    elif args.model == "point_transformer":
        from src.models.point_transformer import PointTransformer
        model = PointTransformer(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        model_label = "PointTransformer"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_label}  |  Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_names=CLASS_NAMES,
        class_weights=class_weights,
        cb_weights=cb_weights,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.fit()

    # ── Log result ────────────────────────────────────────────────────────
    from src.utils.results_logger import log_result
    _m   = trainer.best_metrics
    _cls = _m.get("per_class_iou", {})
    log_result({
        "experiment":      args.experiment,
        "variant":         "",
        "model":           args.model,
        "blocksize":       args.block_size,
        "numpoints":       args.num_points,
        "loss":            args.loss,
        "normal_radius":   args.normal_radius,
        "mIoU":            round(_m.get("miou", float("nan")) * 100, 2),
        "OA":              round(_m.get("overall_acc", float("nan")) * 100, 2),
        "pedestrian_iou":  round(_cls.get("pedestrian", float("nan")) * 100, 2),
        "bollard_iou":     round(_cls.get("bollard", float("nan")) * 100, 2),
        "polesign_iou":    round(_cls.get("pole/sign", float("nan")) * 100, 2),
        "trashcan_iou":    round(_cls.get("trash can", float("nan")) * 100, 2),
        "params":          n_params,
        "train_time_s":    round(trainer.fit_time_s, 1),
        "inference_time_s": "",
    })
    logger.info(f"Result logged → outputs/results.csv")


if __name__ == "__main__":
    main()

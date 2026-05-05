"""
src/training/trainer.py
────────────────────────
Training loop for PointNet++ semantic segmentation.

Features:
  - Weighted cross-entropy loss (inverse class frequency for imbalance)
  - Adam + cosine LR schedule
  - Per-epoch mIoU on validation set
  - TensorBoard logging
  - Checkpoint saving (best val mIoU)
  - Early stopping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    processed_dir: str | Path,
    train_stems: list[str],
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training scans.

    Weight_c = 1 / (freq_c + ε), then normalized so weights sum to num_classes.
    This is the standard approach for class-imbalanced segmentation.

    The severe imbalance in Paris-Lille-3D (ground=46%, bollard=0.0%) makes
    this critical — without it, the model collapses to predicting only
    ground/building and achieves high OA but terrible mIoU.
    """
    processed_dir = Path(processed_dir)
    total_counts = np.zeros(num_classes, dtype=np.int64)

    for stem in train_stems:
        stats_path = processed_dir / stem / "stats.npz"
        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True)
            if "class_counts" in stats:
                total_counts += stats["class_counts"]

    # Ignore the unclassified class
    total_counts[ignore_index] = 0
    freq = total_counts.astype(float)

    weights = np.where(freq > 0, 1.0 / (freq + 1.0), 0.0)
    weights[ignore_index] = 0.0

    # Normalize so that the mean non-zero weight = 1
    valid = freq > 0
    valid[ignore_index] = False
    if valid.sum() > 0:
        weights[valid] /= weights[valid].mean()

    logger.info("Class weights:")
    for i, w in enumerate(weights):
        logger.info(f"  [{i}] w={w:.3f}  (count={total_counts[i]:,})")

    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Training + validation steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)               # (B, N, C)
        labels   = labels.to(device).long()          # (B, N)

        optimizer.zero_grad()
        logits = model(features)                     # (B, N, num_classes)

        # Reshape for cross-entropy: (B*N, num_classes) and (B*N,)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        loss.backward()

        # Gradient clipping — prevents exploding gradients on sparse classes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  Epoch {epoch} [{batch_idx}/{n_batches}]  "
                f"loss={loss.item():.4f}  ({elapsed:.0f}s)"
            )

    mean_loss = total_loss / n_batches
    writer.add_scalar("Loss/train", mean_loss, epoch)
    return mean_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    tracker: MetricTracker,
) -> tuple[float, dict]:
    """Run validation. Returns (mean_loss, metrics_dict)."""
    model.eval()
    tracker.reset()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.to(device).long()

        logits = model(features)                     # (B, N, num_classes)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        total_loss += loss.item()

        pred = logits.argmax(dim=-1)                 # (B, N)
        tracker.update(pred, labels)

    mean_loss = total_loss / len(loader)
    metrics = tracker.compute()

    writer.add_scalar("Loss/val", mean_loss, epoch)
    writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
    writer.add_scalar("Metrics/OA", metrics["overall_acc"], epoch)
    for name, iou in metrics["per_class_iou"].items():
        if not np.isnan(iou):
            writer.add_scalar(f"IoU/{name}", iou, epoch)

    return mean_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates training, validation, checkpointing and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        class_names: list[str],
        class_weights: torch.Tensor | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=cfg.get("ignore_index", 0),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.get("learning_rate", 0.001),
            weight_decay=cfg.get("weight_decay", 0.0001),
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get("epochs", 100),
            eta_min=1e-6,
        )

        # Directories
        self.ckpt_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir  = Path(cfg.get("log_dir", "outputs/logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        exp_name = cfg.get("experiment_name", "pointnet2")
        self.writer = SummaryWriter(self.log_dir / exp_name)
        self.tracker = MetricTracker(
            num_classes=self.num_classes,
            class_names=class_names,
            ignore_index=cfg.get("ignore_index", 0),
        )

        self.best_miou = 0.0
        self.patience_counter = 0
        self.patience = cfg.get("early_stopping_patience", 15)

    def fit(self) -> None:
        n_epochs = self.cfg.get("epochs", 100)
        logger.info(f"Starting training for {n_epochs} epochs")

        for epoch in range(1, n_epochs + 1):
            t_epoch = time.time()
            logger.info(f"\n{'═'*60}")
            logger.info(f"Epoch {epoch}/{n_epochs}  —  lr={self.optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.device, epoch, self.writer,
            )

            # Validate
            val_loss, metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, epoch, self.writer, self.tracker,
            )

            self.scheduler.step()

            # Log
            miou = metrics["miou"]
            logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
            logger.info(self.tracker.log_str(metrics))
            logger.info(f"  Epoch time: {time.time() - t_epoch:.0f}s")

            # Checkpoint
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics, is_best=True)
                logger.info(f"  ★ New best mIoU: {miou*100:.2f}%  — checkpoint saved")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement ({self.patience_counter}/{self.patience}). "
                    f"Best: {self.best_miou*100:.2f}%"
                )

            # Save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, metrics, is_best=False)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

            # Reshuffle dataset blocks for next epoch
            if hasattr(self.train_loader.dataset, "on_epoch_end"):
                self.train_loader.dataset.on_epoch_end()

        self.writer.close()
        logger.info(f"\nTraining complete. Best val mIoU: {self.best_miou*100:.2f}%")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        exp_name = self.cfg.get("experiment_name", "pointnet2")
        fname = "best.pth" if is_best else f"epoch_{epoch:03d}.pth"
        path = self.ckpt_dir / exp_name / fname
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "miou": metrics["miou"],
            "cfg": self.cfg,
        }, path)

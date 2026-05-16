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
import torch.nn.functional as F
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


def _load_class_counts(
    processed_dir: str | Path,
    train_stems: list[str],
    num_classes: int,
) -> np.ndarray:
    """Load per-class point counts from stats.npz files."""
    processed_dir = Path(processed_dir)
    counts = np.zeros(num_classes, dtype=np.int64)
    for stem in train_stems:
        p = processed_dir / stem / "stats.npz"
        if p.exists():
            s = np.load(p, allow_pickle=True)
            if "class_counts" in s:
                counts += s["class_counts"]
    return counts


def compute_cb_weights(
    processed_dir: str | Path,
    train_stems: list[str],
    num_classes: int,
    ignore_index: int = 0,
    beta: float | None = None,
) -> torch.Tensor:
    """Class-balanced weights (Cui et al. 2019).

    w_c = (1 - β) / (1 - β^n_c),  β = (N-1)/N.
    Compresses the long-tail imbalance less aggressively than inverse frequency,
    which avoids over-weighting very rare classes at the expense of common ones.
    """
    counts = _load_class_counts(processed_dir, train_stems, num_classes)
    counts[ignore_index] = 0

    if beta is None:
        total = max(int(counts.sum()), 1)
        beta = (total - 1.0) / total

    eff = 1.0 - np.power(beta, counts.astype(float))
    weights = np.where(eff > 0, (1.0 - beta) / eff, 0.0)
    weights[ignore_index] = 0.0

    valid = (counts > 0)
    valid[ignore_index] = False
    if valid.sum() > 0:
        weights[valid] /= weights[valid].mean()

    logger.info("Class-balanced (CB) weights:")
    for i, w in enumerate(weights):
        logger.info(f"  [{i}] w={w:.3f}  (count={counts[i]:,})")

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss for point-cloud segmentation (Lin et al. 2017).

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Accepts the same `weight` and `ignore_index` arguments as CrossEntropyLoss
    so it can be swapped in transparently.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (N, C)   targets : (N,)

        valid = targets != self.ignore_index

        # Clamp ignored indices so gather never reads an out-of-range target.
        # These positions are masked out before the final reduction so their
        # loss values are irrelevant; we just need the indexing to be safe.
        targets_safe = targets.masked_fill(~valid, 0)

        # --- pt must be derived from raw logits, NOT from weighted CE ----------
        # F.cross_entropy(weight=w) computes  -w_y * log(p_y), so
        #   torch.exp(-ce) = p_y^{w_y}  ≠  p_y
        # For high-weight classes (bollard w≈4.56) this pushes pt^w toward 0
        # even for well-classified examples, breaking the focal modulation.
        # The fix: compute log_p independently, then gather p_t from raw logits.
        log_p  = F.log_softmax(logits, dim=1)                                  # (N, C)
        log_pt = log_p.gather(1, targets_safe.unsqueeze(1)).squeeze(1)         # (N,)
        pt     = log_pt.exp()                                                   # (N,)

        # Focal modulation: (1 - p_t)^γ  —  based on true probability
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Per-point loss: class weight is applied as a plain multiplicative
        # scalar on the final term, NOT inside the log computation
        loss = -focal_weight * log_pt                                           # (N,)
        if self.weight is not None:
            alpha = self.weight[targets_safe]
            loss  = alpha * loss

        return loss[valid].mean() if valid.any() else loss.sum() * 0.0


def _build_criterion(
    loss_type: str,
    class_weights: torch.Tensor | None,
    cb_weights: torch.Tensor | None,
    ignore_index: int,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory that returns the requested loss module."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma, weight=class_weights, ignore_index=ignore_index)
    elif loss_type == "cb_focal":
        if cb_weights is None:
            raise ValueError("cb_weights must be provided when loss='cb_focal'")
        return FocalLoss(gamma=gamma, weight=cb_weights, ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Unknown loss '{loss_type}'. Choose: ce | weighted_ce | focal | cb_focal"
        )


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
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> float:
    """Run one training epoch. Returns mean loss.

    Pass a GradScaler (from Trainer) to enable automatic mixed precision (AMP).
    AMP halves VRAM usage and speeds up training ~1.5-2× on Ampere+ GPUs.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()
    use_amp = scaler is not None

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)               # (B, N, C)
        labels   = labels.to(device).long()          # (B, N)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(features)                 # (B, N, num_classes)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
    use_amp: bool = False,
) -> tuple[float, dict]:
    """Run validation. Returns (mean_loss, metrics_dict)."""
    model.eval()
    tracker.reset()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.to(device).long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(features)                 # (B, N, num_classes)
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
        cb_weights: torch.Tensor | None = None,
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
        loss_type   = cfg.get("loss", "weighted_ce")
        ignore_idx  = cfg.get("ignore_index", 0)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        if cb_weights is not None:
            cb_weights = cb_weights.to(self.device)
        self.criterion = _build_criterion(loss_type, class_weights, cb_weights, ignore_idx)
        logger.info(f"Loss: {loss_type}")

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
        self.best_metrics: dict = {}
        self.fit_time_s: float = 0.0
        self.patience_counter = 0
        self.patience = cfg.get("early_stopping_patience", 15)
        self.start_epoch = 1

        # Automatic mixed precision — enabled on CUDA by default, no-op on CPU.
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("AMP enabled (autocast + GradScaler)")

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a saved checkpoint."""
        path = Path(path)
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_miou = ckpt.get("miou", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        # Fast-forward the LR scheduler to match the saved epoch
        for _ in range(self.start_epoch - 1):
            self.scheduler.step()
        logger.info(
            f"  Resuming from epoch {self.start_epoch}  |  "
            f"Best mIoU so far: {self.best_miou*100:.2f}%"
        )

    def fit(self) -> None:
        n_epochs = self.cfg.get("epochs", 100)
        logger.info(f"Starting training for {n_epochs} epochs")
        t_fit_start = time.time()

        for epoch in range(self.start_epoch, n_epochs + 1):
            t_epoch = time.time()
            logger.info(f"\n{'═'*60}")
            logger.info(f"Epoch {epoch}/{n_epochs}  —  lr={self.optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer,
                self.criterion, self.device, epoch, self.writer,
                scaler=self.scaler,
            )

            torch.cuda.empty_cache()

            # Validate
            val_loss, metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, epoch, self.writer, self.tracker,
                use_amp=self.use_amp,
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
                self.best_metrics = metrics
                self.patience_counter = 0
                self._save_checkpoint(epoch, metrics, is_best=True)
                logger.info(f"  ★ New best mIoU: {miou*100:.2f}%  — checkpoint saved")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  No improvement ({self.patience_counter}/{self.patience}). "
                    f"Best: {self.best_miou*100:.2f}%"
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

            # Reshuffle dataset blocks for next epoch
            if hasattr(self.train_loader.dataset, "on_epoch_end"):
                self.train_loader.dataset.on_epoch_end()

        self.fit_time_s = time.time() - t_fit_start
        self.writer.close()
        logger.info(f"\nTraining complete. Best val mIoU: {self.best_miou*100:.2f}%  ({self.fit_time_s/60:.1f} min)")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        exp_name = self.cfg.get("experiment_name", "pointnet2")
        fname = "best.pth" if is_best else f"epoch_{epoch:03d}.pth"
        path = self.ckpt_dir / exp_name / fname
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: save to .tmp then rename to avoid corrupted files on OOM/disk-full
        tmp_path = path.with_suffix(".tmp")
        try:
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "miou": metrics["miou"],
                "cfg": self.cfg,
            }, tmp_path)
            tmp_path.replace(path)
            logger.info(f"  Checkpoint saved -> {path}")
        except (RuntimeError, OSError) as e:
            logger.error(f"  Checkpoint save failed ({e}). Skipping — training continues.")
            # On Windows, torch.save's C++ zip writer may keep the file handle open
            # for a brief moment after an exception, causing PermissionError on unlink.
            # Retry a few times before giving up — orphaned .tmp files are harmless.
            for _attempt in range(4):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.3)

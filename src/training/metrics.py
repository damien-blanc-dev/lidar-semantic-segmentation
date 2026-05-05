"""
src/training/metrics.py
────────────────────────
Segmentation metrics: mIoU, per-class IoU, Overall Accuracy, confusion matrix.

All functions work on raw numpy arrays or torch tensors (converted internally).
The confusion matrix is the single source of truth — all other metrics derive from it.
"""

from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> np.ndarray:
    """Accumulate a (num_classes, num_classes) confusion matrix.

    Parameters
    ----------
    pred         : (N,) predicted class indices
    target       : (N,) ground-truth class indices
    num_classes  : total number of classes
    ignore_index : class to exclude (default 0 = "unclassified")

    Returns
    -------
    cm : (num_classes, num_classes) int64 array
         cm[true, pred] = count
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.flatten().astype(np.int64)
    target = target.flatten().astype(np.int64)

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    cm = np.bincount(
        num_classes * target + pred,
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return cm


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU from a confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
         = cm[c,c] / (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp          # column sum minus diagonal
    fn = cm.sum(axis=1) - tp          # row sum minus diagonal
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def compute_metrics(
    cm: np.ndarray,
    class_names: list[str],
    ignore_index: int = 0,
) -> dict:
    """Compute all segmentation metrics from a confusion matrix.

    Returns
    -------
    dict with keys:
        miou          : float — mean IoU over valid classes
        overall_acc   : float — fraction of correctly classified points
        per_class_iou : dict[class_name → float]
        per_class_acc : dict[class_name → float]
    """
    # Exclude ignore_index from mIoU computation
    valid = [i for i in range(len(cm)) if i != ignore_index]
    cm_valid = cm[valid, :][:, valid]

    iou = iou_from_confusion(cm_valid)
    miou = float(np.nanmean(iou))

    # Overall accuracy (on all valid classes)
    total = cm_valid.sum()
    correct = np.diag(cm_valid).sum()
    overall_acc = float(correct / total) if total > 0 else 0.0

    # Per-class accuracy = recall = TP / (TP + FN)
    row_sums = cm_valid.sum(axis=1).astype(float)
    per_class_acc_arr = np.where(
        row_sums > 0,
        np.diag(cm_valid) / row_sums,
        np.nan,
    )

    valid_names = [class_names[i] for i in valid]
    per_class_iou  = {n: float(v) for n, v in zip(valid_names, iou)}
    per_class_acc  = {n: float(v) for n, v in zip(valid_names, per_class_acc_arr)}

    return {
        "miou": miou,
        "overall_acc": overall_acc,
        "per_class_iou": per_class_iou,
        "per_class_acc": per_class_acc,
    }


class MetricTracker:
    """Accumulates predictions across batches and computes metrics at epoch end.

    Usage:
        tracker = MetricTracker(num_classes=10, class_names=CLASS_NAMES)
        for pred, target in loader:
            tracker.update(pred, target)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        ignore_index: int = 0,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        self.cm += confusion_matrix(
            pred, target, self.num_classes, self.ignore_index
        )

    def compute(self) -> dict:
        return compute_metrics(self.cm, self.class_names, self.ignore_index)

    def reset(self) -> None:
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def log_str(self, metrics: dict | None = None) -> str:
        """Return a formatted string suitable for logging."""
        m = metrics or self.compute()
        lines = [
            f"  mIoU: {m['miou']*100:.2f}%   OA: {m['overall_acc']*100:.2f}%",
            "  Per-class IoU:",
        ]
        for name, iou in m["per_class_iou"].items():
            acc = m["per_class_acc"].get(name, float("nan"))
            iou_str = f"{iou*100:5.1f}%" if not np.isnan(iou) else "  N/A "
            acc_str = f"{acc*100:5.1f}%" if not np.isnan(acc) else "  N/A "
            lines.append(f"    {name:<20}  IoU={iou_str}  Acc={acc_str}")
        return "\n".join(lines)

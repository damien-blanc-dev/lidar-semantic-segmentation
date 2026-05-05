"""
scripts/error_analysis.py
──────────────────────────
Error analysis for PointNet++ predictions on Paris-Lille-3D.

Generates 4 figures saved to outputs/figures/error_analysis/:
  1. confusion_matrix.png  — normalized confusion matrix (recall per class)
  2. error_map.png         — top-down view: correct=class color, wrong=red
  3. confusion_pairs.png   — top 10 most frequent confusion pairs (bar chart)
  4. hard_classes.png      — zoom on pedestrian & bollard prediction vs GT

Usage:
    # Run inference first to generate predictions:
    python scripts/inference.py --scan Paris

    # Then run error analysis:
    python scripts/error_analysis.py --scan Paris
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

PRED_DIR   = Path("outputs/predictions")
OUT_DIR    = Path("outputs/figures/error_analysis")
PROC_DIR   = Path("data/processed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=str, default="Paris")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: "np.ndarray", class_names: list, save_path: Path):
    """Recall-normalized confusion matrix (row = ground truth, col = predicted).

    Normalization by row means each cell shows:
    'Given GT class X, what fraction did the model predict as class Y?'
    The diagonal = per-class recall.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Exclude class 0 (unclassified) — not meaningful for segmentation quality
    idx = list(range(1, len(class_names)))
    cm_sub  = cm[np.ix_(idx, idx)].astype(float)
    names   = [class_names[i] for i in idx]

    # Row-normalize (recall)
    row_sums = cm_sub.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_sub / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_norm[i, j]
            count = int(cm_sub[i, j])
            text_color = "white" if val > 0.55 else "#ccc"
            ax.text(
                j, i,
                f"{val:.2f}\n({count:,})" if count > 0 else "",
                ha="center", va="center",
                fontsize=7, color=text_color,
            )

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9, color="white")
    ax.set_yticklabels(names, fontsize=9, color="white")
    ax.set_xlabel("Predicted class", fontsize=11, color="white")
    ax.set_ylabel("Ground truth class", fontsize=11, color="white")
    ax.set_title("Confusion matrix (row-normalized = recall per class)",
                 fontsize=12, fontweight="bold", color="white", pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Recall", color="white")

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Error map
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_map(
    xyz: "np.ndarray",
    gt: "np.ndarray",
    pred: "np.ndarray",
    save_path: Path,
    subsample: int = 400_000,
):
    """Top-down view where:
      - Correctly classified points → their semantic class color (faded)
      - Misclassified points        → bright red

    This immediately reveals *where* the model struggles spatially —
    typically at class boundaries, occluded zones, and rare object locations.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.data.loader import CLASS_COLORS

    correct = (pred == gt)
    wrong   = ~correct & (gt != 0)   # exclude unclassified from error count

    # Subsample for rendering
    n = len(xyz)
    if n > subsample:
        # Stratified: keep all wrong points (they're the interesting ones)
        wrong_idx   = np.where(wrong)[0]
        correct_idx = np.where(correct & (gt != 0))[0]
        n_correct   = min(subsample - len(wrong_idx), len(correct_idx))
        if n_correct > 0:
            correct_idx = np.random.choice(correct_idx, n_correct, replace=False)
        plot_idx = np.concatenate([correct_idx, wrong_idx])
    else:
        plot_idx = np.arange(n)

    plot_correct = correct[plot_idx]
    plot_gt      = gt[plot_idx]
    plot_pred    = pred[plot_idx]
    plot_xyz     = xyz[plot_idx]

    # Colors: correct → class color (50% opacity equivalent via lightening)
    colors = np.ones((len(plot_idx), 3), dtype=np.float32)
    for i, (is_correct, gt_cls) in enumerate(zip(plot_correct, plot_gt)):
        if gt_cls == 0:
            colors[i] = [0.2, 0.2, 0.2]
        elif is_correct:
            # Lighten correct predictions slightly (blend with dark bg)
            colors[i] = CLASS_COLORS[gt_cls] * 0.6 + 0.1
        else:
            colors[i] = [1.0, 0.1, 0.1]  # bright red for errors

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, title, c in [
        (axes[0], "GT class colors (correct) + errors (red)", colors),
        (axes[1], "Error map only", np.where(
            plot_correct[:, None], np.array([[0.15, 0.15, 0.15]]), np.array([[1.0, 0.1, 0.1]])
        )),
    ]:
        ax.scatter(
            plot_xyz[:, 0], plot_xyz[:, 1],
            c=c, s=0.3, linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#333")
        ax.set_xlabel("X (m)", color="white", fontsize=9)
        ax.set_ylabel("Y (m)", color="white", fontsize=9)

    n_wrong = wrong.sum()
    n_valid = (gt != 0).sum()
    error_rate = n_wrong / n_valid * 100 if n_valid > 0 else 0
    fig.suptitle(
        f"Error map — Paris scan  |  {n_wrong:,} misclassified / {n_valid:,} valid points ({error_rate:.1f}% error rate)",
        fontsize=13, fontweight="bold", color="white",
    )
    fig.patch.set_facecolor("#0d0d1a")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Top confusion pairs
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_pairs(cm: "np.ndarray", class_names: list, save_path: Path, top_k: int = 10):
    """Horizontal bar chart of the most frequent off-diagonal confusion pairs.

    Answers: 'Which specific GT→Pred substitutions happen most often?'
    More actionable than the full matrix for understanding failure modes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and i != 0 and j != 0:   # skip diagonal and unclassified
                pairs.append((cm[i, j], class_names[i], class_names[j]))

    pairs.sort(reverse=True)
    top = pairs[:top_k]

    counts  = [p[0] for p in top]
    labels  = [f"{p[1]}  →  {p[2]}" for p in top]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top)), counts, color="#E05252", edgecolor="white", linewidth=0.4)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9, color="white")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of misclassified points", fontsize=10, color="white")
    ax.set_title(f"Top {top_k} confusion pairs  (GT → Predicted)",
                 fontsize=12, fontweight="bold", color="white")
    ax.invert_yaxis()
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#444")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Hard class zoom
# ─────────────────────────────────────────────────────────────────────────────

def plot_hard_classes(
    xyz: "np.ndarray",
    gt: "np.ndarray",
    pred: "np.ndarray",
    save_path: Path,
    hard_classes: list = None,
):
    """For each hard class: zoom into a region containing GT instances and
    compare GT (left) vs prediction (right).

    Shows whether errors are isolated points, boundary effects, or systematic.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.data.loader import CLASS_COLORS, CLASS_NAMES

    if hard_classes is None:
        hard_classes = [7, 4, 3]   # pedestrian, bollard, pole/sign

    n_classes = len(hard_classes)
    fig, axes = plt.subplots(n_classes, 2, figsize=(14, 4 * n_classes))
    if n_classes == 1:
        axes = axes[None, :]

    CONTEXT_RADIUS = 20.0   # meters around the class instances

    for row, cls_id in enumerate(hard_classes):
        cls_mask = (gt == cls_id)
        n_instances = cls_mask.sum()

        if n_instances == 0:
            axes[row, 0].text(0.5, 0.5, f"No {CLASS_NAMES[cls_id]} instances",
                              ha="center", va="center", color="white", transform=axes[row, 0].transAxes)
            continue

        # Find a 20m × 20m window with the highest concentration of this class
        # Use the centroid of all instances as center
        center_xy = xyz[cls_mask, :2].mean(axis=0)
        cx, cy = center_xy

        window_mask = (
            (xyz[:, 0] >= cx - CONTEXT_RADIUS) & (xyz[:, 0] < cx + CONTEXT_RADIUS) &
            (xyz[:, 1] >= cy - CONTEXT_RADIUS) & (xyz[:, 1] < cy + CONTEXT_RADIUS)
        )
        w_xyz  = xyz[window_mask]
        w_gt   = gt[window_mask]
        w_pred = pred[window_mask]

        for col, (labels_arr, title) in enumerate([
            (w_gt,   f"Ground truth — {CLASS_NAMES[cls_id]}"),
            (w_pred, f"Prediction   — {CLASS_NAMES[cls_id]}"),
        ]):
            ax = axes[row, col]
            colors = CLASS_COLORS[np.clip(labels_arr, 0, len(CLASS_COLORS) - 1)]

            # Highlight the target class with full brightness, dim others
            is_target = (labels_arr == cls_id)
            colors[~is_target] *= 0.25   # dim non-target classes

            ax.scatter(w_xyz[:, 0], w_xyz[:, 1],
                       c=colors, s=1.5, linewidths=0)

            n_target = is_target.sum()
            ax.set_title(f"{title}  ({n_target} pts in window)",
                         fontsize=10, color="white", fontweight="bold")
            ax.set_aspect("equal")
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#333")

    fig.suptitle(
        "Hard class analysis — 20m × 20m zoom around highest-density instance clusters",
        fontsize=12, fontweight="bold", color="white",
    )
    fig.patch.set_facecolor("#0d0d1a")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────
    pred_path = PRED_DIR / f"{args.scan}_pred.npy"
    gt_path   = PRED_DIR / f"{args.scan}_gt.npy"

    if not pred_path.exists():
        logger.error(
            f"Predictions not found: {pred_path}\n"
            f"Run first: python scripts/inference.py --scan {args.scan}"
        )
        sys.exit(1)

    pred   = np.load(pred_path)
    gt     = np.load(gt_path)
    points = np.load(PROC_DIR / args.scan / "points.npy")
    xyz    = points[:, :3]

    logger.info(f"Loaded {len(pred):,} predictions for scan '{args.scan}'")

    # ── Confusion matrix ──────────────────────────────────────────────────
    from src.training.metrics import confusion_matrix, compute_metrics
    from src.data.loader import CLASS_NAMES, NUM_CLASSES

    cm = confusion_matrix(pred, gt, NUM_CLASSES, ignore_index=0)
    metrics = compute_metrics(cm, CLASS_NAMES, ignore_index=0)

    logger.info(f"mIoU: {metrics['miou']*100:.2f}%  OA: {metrics['overall_acc']*100:.2f}%")

    # ── Generate all figures ──────────────────────────────────────────────
    logger.info("Generating figures ...")

    plot_confusion_matrix(cm, CLASS_NAMES,
                          OUT_DIR / "confusion_matrix.png")

    plot_error_map(xyz, gt, pred,
                   OUT_DIR / "error_map.png")

    plot_confusion_pairs(cm, CLASS_NAMES,
                         OUT_DIR / "confusion_pairs.png")

    plot_hard_classes(xyz, gt, pred,
                      OUT_DIR / "hard_classes.png",
                      hard_classes=[7, 4, 3])   # pedestrian, bollard, pole/sign

    plt.close("all")

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Top confusion pairs:")
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and i != 0 and j != 0:
                pairs.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    pairs.sort(reverse=True)
    for count, gt_cls, pred_cls in pairs[:8]:
        logger.info(f"  {gt_cls:<20} → {pred_cls:<20} : {count:>10,} pts")

    logger.info(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

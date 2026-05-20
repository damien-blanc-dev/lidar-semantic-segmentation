"""
scripts/uncertainty_analysis.py
---------------------------------
Exp 5 - Predictive uncertainty analysis from averaged softmax probabilities.

Requires running inference first with --save_probs:
    python scripts/inference.py --scan Paris \\
        --checkpoint outputs/checkpoints/exp4_pn2_wce_znorm/best.pth \\
        --model pointnet2 --save_probs

Generates 4 figures saved to outputs/figures/uncertainty/:
  1. uncertainty_map.png      - top-down entropy map (spatial view)
  2. per_class_entropy.png    - entropy distribution per GT class
  3. entropy_vs_error.png     - entropy histogram: correct vs incorrect predictions
  4. coverage_accuracy.png    - OA and mIoU as a function of coverage threshold

Usage:
    python scripts/uncertainty_analysis.py
    python scripts/uncertainty_analysis.py --scan Paris
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PRED_DIR = Path("outputs/predictions")
PROC_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/figures/uncertainty")

BG      = "#0d1117"
FG      = "#c9d1d9"
GRID    = "#30363d"
PANEL   = "#161b22"
C_LOW   = "#3fb950"
C_HIGH  = "#f78166"
C_BLUE  = "#58a6ff"


# -----------------------------------------------------------------------------
#  Data loading
# -----------------------------------------------------------------------------

def load_data(scan: str):
    probs_path = PRED_DIR / f"{scan}_probs.npy"
    pred_path  = PRED_DIR / f"{scan}_pred.npy"
    gt_path    = PRED_DIR / f"{scan}_gt.npy"
    pts_path   = PROC_DIR / scan / "points.npy"

    if not probs_path.exists():
        logger.error(
            f"Probs file not found: {probs_path}\n"
            "Run inference with --save_probs first."
        )
        sys.exit(1)

    logger.info("Loading data ...")
    xyz   = np.load(pts_path)[:, :3].astype(np.float32)
    pred  = np.load(pred_path).astype(np.int32)
    gt    = np.load(gt_path).astype(np.int32)
    probs = np.load(probs_path).astype(np.float32)
    logger.info(f"  {len(xyz):,} points | probs shape: {probs.shape}")
    return xyz, pred, gt, probs


# -----------------------------------------------------------------------------
#  Entropy
# -----------------------------------------------------------------------------

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-point predictive entropy normalized to [0, 1].

    H(x) = -sum_c p_c * log(p_c) / log(C)
    """
    C   = probs.shape[1]
    eps = 1e-10
    H   = -(probs * np.log(probs + eps)).sum(axis=1)
    return (H / np.log(C)).astype(np.float32)


# -----------------------------------------------------------------------------
#  Figure 1 - Spatial uncertainty map
# -----------------------------------------------------------------------------

def plot_uncertainty_map(xyz, entropy, save_path: Path, n_plot: int = 500_000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    rng = np.random.default_rng(42)
    idx = rng.choice(len(xyz), min(n_plot, len(xyz)), replace=False)
    x, y, h = xyz[idx, 0], xyz[idx, 1], entropy[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 9), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG)
        ax.spines[:].set_color(GRID)

    norm = Normalize(vmin=0, vmax=1)
    sc = axes[0].scatter(x, y, c=h, cmap="plasma", s=0.3, linewidths=0, norm=norm)
    axes[0].set_title("Predictive entropy", color=FG, fontsize=11)
    axes[0].set_xlabel("X (m)", color=FG)
    axes[0].set_ylabel("Y (m)", color=FG)
    axes[0].set_aspect("equal")
    cb = plt.colorbar(sc, ax=axes[0], fraction=0.03)
    cb.set_label("Normalized entropy", color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)

    threshold = np.percentile(h, 80)
    colors = np.where(h > threshold, C_HIGH, "#3a3f47")
    axes[1].scatter(x, y, c=colors, s=0.3, linewidths=0)
    axes[1].set_title(f"High-entropy points (top 20%, H > {threshold:.2f})", color=FG, fontsize=11)
    axes[1].set_xlabel("X (m)", color=FG)
    axes[1].set_ylabel("Y (m)", color=FG)
    axes[1].set_aspect("equal")

    fig.suptitle("Spatial uncertainty - Paris validation scan", color=FG, fontsize=13, y=1.01)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info(f"  Saved -> {save_path}")


# -----------------------------------------------------------------------------
#  Figure 2 - Per-class entropy distribution
# -----------------------------------------------------------------------------

def plot_per_class_entropy(entropy, pred, gt, class_names, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = (gt != 0) & (pred != 0)
    valid_classes = [i for i in range(1, len(class_names))]
    data, labels, means = [], [], []

    for c in valid_classes:
        mask = scored & (gt == c)
        if mask.sum() < 10:
            continue
        vals = entropy[mask]
        data.append(vals)
        labels.append(class_names[c])
        means.append(float(vals.mean()))

    order  = np.argsort(means)
    data   = [data[i] for i in order]
    labels = [labels[i] for i in order]
    means  = [means[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_BLUE)
        pc.set_edgecolor(GRID)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color(FG)

    ax.scatter(range(len(means)), means, color=C_HIGH, s=50, zorder=5, label="Mean entropy")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", color=FG, fontsize=10)
    ax.set_ylabel("Normalized entropy H(x)", color=FG, fontsize=11)
    ax.tick_params(colors=FG)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9)
    ax.set_title(
        "Predictive entropy per GT class (sorted by mean, easy left - hard right)",
        color=FG, fontsize=11,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info(f"  Saved -> {save_path}")


# -----------------------------------------------------------------------------
#  Figure 3 - Entropy vs prediction error
# -----------------------------------------------------------------------------

def plot_entropy_vs_error(entropy, pred, gt, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid     = (gt != 0) & (pred != 0)
    correct   = valid & (pred == gt)
    incorrect = valid & (pred != gt)

    bins = np.linspace(0, 1, 60)
    h_c  = entropy[correct]
    h_i  = entropy[incorrect]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)

    ax.hist(h_c, bins=bins, density=True, alpha=0.75,
            color=C_LOW,  label=f"Correct   (n={correct.sum():,})")
    ax.hist(h_i, bins=bins, density=True, alpha=0.75,
            color=C_HIGH, label=f"Incorrect (n={incorrect.sum():,})")

    ax.axvline(h_c.mean(), color=C_LOW,  linestyle="--", linewidth=1.5,
               label=f"Mean correct   {h_c.mean():.3f}")
    ax.axvline(h_i.mean(), color=C_HIGH, linestyle="--", linewidth=1.5,
               label=f"Mean incorrect {h_i.mean():.3f}")

    ax.set_xlabel("Normalized entropy H(x)", color=FG, fontsize=11)
    ax.set_ylabel("Density", color=FG, fontsize=11)
    ax.tick_params(colors=FG)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=9)
    ax.set_title(
        "Entropy distribution: correct vs incorrect predictions\n"
        "Higher entropy correlates with prediction errors",
        color=FG, fontsize=11,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info(f"  Saved -> {save_path}")


# -----------------------------------------------------------------------------
#  Figure 4 - Coverage / accuracy tradeoff
# -----------------------------------------------------------------------------

def plot_coverage_accuracy(entropy, pred, gt, num_classes, class_names,
                           save_path: Path, n_thresholds: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid            = (gt != 0) & (pred != 0)
    valid_idx        = np.where(valid)[0]
    sorted_valid_idx = valid_idx[np.argsort(entropy[valid_idx])]
    N_valid          = len(sorted_valid_idx)

    coverages, oas, mious = [], [], []

    for frac in np.linspace(0.1, 1.0, n_thresholds):
        n_keep     = max(1, int(frac * N_valid))
        kept_valid = sorted_valid_idx[:n_keep]
        if len(kept_valid) < 10:
            continue

        p = pred[kept_valid]
        g = gt[kept_valid]

        oa = float((p == g).mean())

        ious = []
        for c in range(1, num_classes):
            tp = int(((p == c) & (g == c)).sum())
            fp = int(((p == c) & (g != c)).sum())
            fn = int(((p != c) & (g == c)).sum())
            denom = tp + fp + fn
            if denom > 0:
                ious.append(tp / denom)
        miou = float(np.mean(ious)) if ious else 0.0

        coverage = len(kept_valid) / N_valid
        coverages.append(coverage * 100)
        oas.append(oa * 100)
        mious.append(miou * 100)

    coverages = np.array(coverages)
    oas       = np.array(oas)
    mious     = np.array(mious)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)

    ax.plot(coverages, oas,   color=C_BLUE, linewidth=2.0, label="OA (%)")
    ax.plot(coverages, mious, color=C_LOW,  linewidth=2.0, label="mIoU (%)")

    idx_80 = int(np.searchsorted(coverages, 80))
    if idx_80 < len(coverages):
        ax.axvline(80, color=GRID, linestyle=":", linewidth=1.2)
        ax.annotate(
            f"80% coverage\nOA   {oas[idx_80]:.1f}%\nmIoU {mious[idx_80]:.1f}%",
            xy=(80, (oas[idx_80] + mious[idx_80]) / 2),
            xytext=(58, (oas[idx_80] + mious[idx_80]) / 2 - 2),
            color=FG, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=FG, lw=0.8),
        )
        logger.info(
            f"  Coverage 80%: OA {oas[idx_80]:.1f}% (+{oas[idx_80]-oas[-1]:.1f}pp), "
            f"mIoU {mious[idx_80]:.1f}% (+{mious[idx_80]-mious[-1]:.1f}pp) "
            f"vs full coverage"
        )

    ax.set_xlabel("Coverage (% of valid points retained, most certain first)", color=FG, fontsize=11)
    ax.set_ylabel("Score (%)", color=FG, fontsize=11)
    ax.tick_params(colors=FG)
    ax.spines[:].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(coverages.min() - 1, 101)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=FG, fontsize=10)
    ax.set_title(
        "Coverage / accuracy tradeoff - entropy-based abstention\n"
        "Rejecting the most uncertain points raises accuracy on retained predictions",
        color=FG, fontsize=11,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logger.info(f"  Saved -> {save_path}")


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=str, default="Paris")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.data.loader import CLASS_NAMES, NUM_CLASSES

    xyz, pred, gt, probs = load_data(args.scan)
    entropy = compute_entropy(probs)
    logger.info(
        f"  Entropy stats: min={entropy.min():.3f}  "
        f"mean={entropy.mean():.3f}  max={entropy.max():.3f}"
    )

    logger.info("Generating figures ...")
    plot_uncertainty_map(xyz, entropy,     OUT_DIR / "uncertainty_map.png")
    plot_per_class_entropy(entropy, pred, gt, CLASS_NAMES, OUT_DIR / "per_class_entropy.png")
    plot_entropy_vs_error(entropy, pred, gt,         OUT_DIR / "entropy_vs_error.png")
    plot_coverage_accuracy(entropy, pred, gt, NUM_CLASSES, CLASS_NAMES,
                           OUT_DIR / "coverage_accuracy.png")

    logger.info(f"Done. All figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()

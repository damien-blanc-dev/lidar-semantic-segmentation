"""
scripts/uncertainty_analysis.py
─────────────────────────────────
Experiment 5: Uncertainty and error triage.

Requires inference.py to be run first with --save_probs:
    python scripts/inference.py --scan Paris --save_probs --experiment exp5_pointnet2

Computes:
  - Per-point softmax entropy  H = -Σ p_c · log₂(p_c + ε)
  - Top-1 / top-2 margin       Δ = p_max - p_second_max
  - AUROC: how well each signal predicts actual misclassifications
  - Precision@k on the most-uncertain subset

Requires: scikit-learn  (pip install scikit-learn)

Usage:
    python scripts/uncertainty_analysis.py --scan Paris
    python scripts/uncertainty_analysis.py --scan Paris --topk 20000
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

PRED_DIR = Path("outputs/predictions")
PROC_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/figures/uncertainty")


def parse_args():
    p = argparse.ArgumentParser(description="Uncertainty + error triage — Experiment 5")
    p.add_argument("--scan",      type=str, default="Paris")
    p.add_argument("--pred_dir",  type=str, default=str(PRED_DIR))
    p.add_argument("--proc_dir",  type=str, default=str(PROC_DIR))
    p.add_argument("--topk",      type=int, default=10_000,
                   help="Points to inspect for precision@k (default: 10000)")
    p.add_argument("--subsample", type=int, default=400_000,
                   help="Max points for map rendering (default: 400000)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Uncertainty signals
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy(probs: "np.ndarray", eps: float = 1e-8) -> "np.ndarray":
    """Per-point Shannon entropy (bits).  High entropy = uncertain.  Shape (N,)."""
    import numpy as np
    return -(probs * np.log2(probs + eps)).sum(axis=1)


def compute_margin(probs: "np.ndarray") -> "np.ndarray":
    """Per-point top-1 / top-2 margin.  Low margin = uncertain.  Shape (N,)."""
    import numpy as np
    sorted_p = np.sort(probs, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.error("scikit-learn is required: pip install scikit-learn")
        sys.exit(1)

    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.pred_dir)
    proc_dir = Path(args.proc_dir)

    # ── Load ──────────────────────────────────────────────────────────────
    probs_path = pred_dir / f"{args.scan}_probs.npy"
    gt_path    = pred_dir / f"{args.scan}_gt.npy"

    if not probs_path.exists():
        logger.error(
            f"Softmax probabilities not found: {probs_path}\n"
            f"Run first:\n"
            f"  python scripts/inference.py --scan {args.scan} --save_probs"
        )
        sys.exit(1)

    probs  = np.load(probs_path)                                   # (N, C)
    gt     = np.load(gt_path)                                      # (N,)
    points = np.load(proc_dir / args.scan / "points.npy")         # (N, 7)
    xyz    = points[:, :3]

    logger.info(f"Loaded {len(probs):,} points  |  probs shape: {probs.shape}")

    # ── Uncertainty signals ───────────────────────────────────────────────
    pred    = probs.argmax(axis=1).astype(np.int32)
    entropy = compute_entropy(probs)
    margin  = compute_margin(probs)

    valid  = gt != 0                         # exclude unclassified
    errors = (pred != gt) & valid            # True = misclassified valid point

    n_valid  = int(valid.sum())
    n_errors = int(errors.sum())
    logger.info(f"Valid points: {n_valid:,}  |  Errors: {n_errors:,}  ({n_errors/n_valid*100:.1f}%)")

    # ── AUROC ─────────────────────────────────────────────────────────────
    ent_v = entropy[valid]
    mar_v = margin[valid]
    err_v = errors[valid].astype(np.uint8)

    auroc_entropy = roc_auc_score(err_v, ent_v)
    auroc_margin  = roc_auc_score(err_v, -mar_v)   # low margin → high uncertainty
    logger.info(f"AUROC (entropy  → error): {auroc_entropy:.4f}")
    logger.info(f"AUROC (1-margin → error): {auroc_margin:.4f}")

    # ── Precision@k ───────────────────────────────────────────────────────
    k = min(args.topk, n_valid)
    valid_idxs = np.where(valid)[0]

    top_by_entropy = valid_idxs[np.argsort(ent_v)[::-1]][:k]
    prec_entropy   = errors[top_by_entropy].sum() / k

    top_by_margin  = valid_idxs[np.argsort(mar_v)][:k]
    prec_margin    = errors[top_by_margin].sum() / k

    baseline = errors[valid].mean()
    logger.info(f"Precision@{k:,} (entropy):    {prec_entropy*100:.1f}%  ({prec_entropy/baseline:.2f}x baseline)")
    logger.info(f"Precision@{k:,} (low-margin): {prec_margin*100:.1f}%  ({prec_margin/baseline:.2f}x baseline)")
    logger.info(f"Baseline precision (random):  {baseline*100:.1f}%")

    # ── Uncertainty maps ──────────────────────────────────────────────────
    logger.info("Rendering uncertainty maps ...")
    n   = len(xyz)
    idx = np.random.choice(n, min(args.subsample, n), replace=False)
    xs, ys = xyz[idx, 0], xyz[idx, 1]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [
        (entropy[idx],               "hot",  "Entropy (high = uncertain)"),
        (-margin[idx],               "hot",  "1 - Margin (high = uncertain)"),
        (errors[idx].astype(float),  "bwr",  "Error mask (1 = misclassified)"),
    ]
    for ax, (scalar, cmap, title) in zip(axes, panels):
        sc = ax.scatter(xs, ys, c=scalar, s=0.3, cmap=cmap, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#333")
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)

    fig.suptitle(
        f"Uncertainty — {args.scan}  |  "
        f"AUROC(entropy)={auroc_entropy:.3f}  AUROC(margin)={auroc_margin:.3f}  |  "
        f"Prec@{k:,}(ent)={prec_entropy*100:.1f}%  baseline={baseline*100:.1f}%",
        fontsize=11, fontweight="bold", color="white",
    )
    fig.patch.set_facecolor("#0d0d1a")
    plt.tight_layout()

    out_path = OUT_DIR / f"{args.scan}_uncertainty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Uncertainty map saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("EXPERIMENT 5 — UNCERTAINTY SUMMARY")
    logger.info(f"  Scan              : {args.scan}")
    logger.info(f"  Valid points      : {n_valid:,}")
    logger.info(f"  Error rate        : {n_errors/n_valid*100:.1f}%")
    logger.info(f"  AUROC (entropy)   : {auroc_entropy:.4f}")
    logger.info(f"  AUROC (1-margin)  : {auroc_margin:.4f}")
    logger.info(f"  Precision@{k:<6,}  entropy={prec_entropy*100:.1f}%  margin={prec_margin*100:.1f}%  baseline={baseline*100:.1f}%")
    logger.info(f"  Entropy lift      : {prec_entropy/baseline:.2f}x above random")
    logger.info("─" * 60)


if __name__ == "__main__":
    main()

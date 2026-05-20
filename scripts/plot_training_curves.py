"""
scripts/plot_training_curves.py
────────────────────────────────
Read TensorBoard event files and plot training curves for one or more experiments.

Outputs:
  outputs/figures/training_curves_<experiment>.png

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --experiments exp4_pn2_wce_znorm exp4_randlanet_wce_znorm
    python scripts/plot_training_curves.py --experiments exp2_weighted_ce exp2_focal_v2 --out training_loss_comparison.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOGS_DIR   = Path("outputs/logs")
FIGURE_DIR = Path("outputs/figures")


def load_scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Return list of (step, value) for a given tag from all event files in log_dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def merge_runs(
    log_dir: Path, tag: str
) -> tuple[list[tuple[int, float]], list[int]]:
    """Merge scalar data from multiple event files (restarts) in a log dir.

    Each restart resets the step counter.  We stitch runs together by
    offsetting later runs' steps so the timeline is continuous.

    Returns
    -------
    (events, restart_epochs) where restart_epochs is the list of epoch
    numbers at which a training restart was detected (hardware crash → resume).
    """
    all_events: list[tuple[int, float]] = []
    restart_epochs: list[int] = []
    offset = 0
    max_step = 0
    for ev_file in sorted(log_dir.glob("events.out.tfevents.*")):
        run_dir = ev_file.parent
        events = load_scalars(run_dir, tag)
        if not events:
            continue
        if all_events:
            first_step = events[0][0]
            if first_step <= max_step:
                offset = max_step
                restart_epochs.append(max_step)
        for step, val in events:
            adjusted = step + offset
            all_events.append((adjusted, val))
            max_step = max(max_step, adjusted)

    all_events.sort(key=lambda x: x[0])
    return all_events, restart_epochs


def plot_single(experiment: str, save_path: Path) -> None:
    """Plot train loss, val loss, and val mIoU for one experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    log_dir = LOGS_DIR / experiment
    if not log_dir.exists():
        print(f"  Log dir not found: {log_dir}", file=sys.stderr)
        return

    train_loss, restarts_tl = merge_runs(log_dir, "Loss/train")
    val_loss,   _            = merge_runs(log_dir, "Loss/val")
    val_miou,   restarts     = merge_runs(log_dir, "Metrics/mIoU")

    if not val_miou:
        print(f"  No mIoU data found for {experiment}", file=sys.stderr)
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")

    color_train   = "#4c9be8"
    color_val     = "#58a6ff"
    color_miou    = "#3fb950"
    color_restart = "#f78166"

    def unzip(pairs):
        if not pairs:
            return [], []
        xs, ys = zip(*pairs)
        return list(xs), list(ys)

    tl_x, tl_y = unzip(train_loss)
    vl_x, vl_y = unzip(val_loss)
    mi_x, mi_y = unzip(val_miou)

    if tl_x:
        ax1.plot(tl_x, tl_y, color=color_train, alpha=0.6, linewidth=1.0, label="Train loss")
    if vl_x:
        ax1.plot(vl_x, vl_y, color=color_val, linewidth=1.4, label="Val loss")
    ax1.set_xlabel("Epoch", color="white", fontsize=11)
    ax1.set_ylabel("Loss", color=color_val, fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#30363d")

    ax2 = ax1.twinx()
    ax2.set_facecolor("#0d1117")
    if mi_x:
        miou_pct = [v * 100 for v in mi_y]
        ax2.plot(mi_x, miou_pct, color=color_miou, linewidth=2.0, label="Val mIoU (%)")
        best_idx = int(np.argmax(miou_pct))
        ax2.scatter([mi_x[best_idx]], [miou_pct[best_idx]],
                    color=color_miou, s=80, zorder=5)
        ax2.annotate(f"  best: {miou_pct[best_idx]:.2f}%",
                     xy=(mi_x[best_idx], miou_pct[best_idx]),
                     color=color_miou, fontsize=10, fontweight="bold")
    ax2.set_ylabel("mIoU (%)", color=color_miou, fontsize=11)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#30363d")

    # Mark hardware restart points (checkpoint resume after PSU crash)
    for i, ep in enumerate(restarts):
        ax1.axvline(ep, color=color_restart, linewidth=1.0, linestyle="--", alpha=0.7)
        ax1.text(ep + 1, ax1.get_ylim()[1] * 0.97 if ax1.get_ylim()[1] else 0.8,
                 f"restart {i+1}", color=color_restart, fontsize=8, va="top")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=9)

    ax1.set_title(f"Training dynamics — {experiment}", color="white", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {save_path}")


def plot_multi_miou(experiments: list[str], save_path: Path) -> None:
    """Overlay val mIoU curves for several experiments on a single axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#3fb950", "#4c9be8", "#f78166", "#d2a8ff", "#ffa657"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, exp in enumerate(experiments):
        log_dir = LOGS_DIR / exp
        if not log_dir.exists():
            continue
        val_miou, _ = merge_runs(log_dir, "Metrics/mIoU")
        if not val_miou:
            continue
        xs, ys = zip(*val_miou)
        ax.plot(xs, [v * 100 for v in ys],
                color=palette[i % len(palette)],
                linewidth=1.8, label=exp)

    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Val mIoU (%)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)
    ax.set_title("Validation mIoU — experiment comparison", color="white", fontsize=11)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+",
                   default=["exp4_pn2_wce_znorm"],
                   help="Experiment name(s) matching subdirs in outputs/logs/")
    p.add_argument("--out", type=str, default=None,
                   help="Output filename (auto-generated if omitted)")
    p.add_argument("--compare", action="store_true",
                   help="Overlay mIoU curves for all listed experiments")
    return p.parse_args()


def main():
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare or len(args.experiments) > 1:
        out = FIGURE_DIR / (args.out or "training_curves_comparison.png")
        plot_multi_miou(args.experiments, out)
    else:
        exp = args.experiments[0]
        out = FIGURE_DIR / (args.out or f"training_curves_{exp}.png")
        plot_single(exp, out)


if __name__ == "__main__":
    main()

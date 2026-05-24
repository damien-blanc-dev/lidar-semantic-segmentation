"""
scripts/plot_architecture_comparison.py
────────────────────────────────────────
Generate a grouped bar chart comparing architectures from outputs/results.csv.

Outputs:
  outputs/figures/architecture_comparison.png

Usage:
    python scripts/plot_architecture_comparison.py
    python scripts/plot_architecture_comparison.py --experiments exp4_pn2_wce_znorm exp4_randlanet_wce_znorm
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_CSV = Path("outputs/results.csv")
FIGURE_DIR  = Path("outputs/figures")

# Metrics shown in the chart and their display labels
METRICS = [
    ("mIoU",           "mIoU"),
    ("pedestrian_iou", "Pedestrian"),
    ("bollard_iou",    "Bollard"),
    ("polesign_iou",   "Pole/sign"),
    ("trashcan_iou",   "Trash can"),
]

# Color palette per architecture (matched to class colors in the dataset where sensible)
ARCH_COLORS = {
    "pointnet2": "#4c9be8",
    "randlanet":  "#f78166",
    "point_transformer": "#3fb950",
}
ARCH_LABELS = {
    "pointnet2":          "PointNet++ SSG",
    "randlanet":           "RandLA-Net",
    "point_transformer":   "PointTransformer",
}


def load_results(csv_path: Path) -> list[dict]:
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments", nargs="+",
        default=["exp4_pn2_wce_znorm", "exp4_randlanet_wce_znorm", "exp4_pt_wce_znorm"],
        help="Experiment names to compare (must match the 'experiment' column in results.csv)",
    )
    p.add_argument("--out", type=str, default="architecture_comparison.png")
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()
    rows = load_results(RESULTS_CSV)

    # Build lookup: experiment → row (training rows only — skip inference-only rows
    # that lack per-class metrics and would overwrite the full training entry)
    lookup = {
        r["experiment"]: r for r in rows
        if r.get("pedestrian_iou", "").strip()
    }
    selected = []
    for exp in args.experiments:
        if exp not in lookup:
            print(f"  Warning: experiment '{exp}' not found in results.csv", file=sys.stderr)
            continue
        selected.append(lookup[exp])

    if not selected:
        print("No experiments found. Exiting.", file=sys.stderr)
        sys.exit(1)

    metric_keys   = [m[0] for m in METRICS]
    metric_labels = [m[1] for m in METRICS]
    n_metrics = len(METRICS)
    n_arch    = len(selected)

    x = np.arange(n_metrics)
    width = 0.75 / n_arch
    offsets = np.linspace(-(n_arch - 1) / 2, (n_arch - 1) / 2, n_arch) * width

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, row in enumerate(selected):
        arch  = row.get("model", "unknown")
        color = ARCH_COLORS.get(arch, "#aaaaaa")
        label = ARCH_LABELS.get(arch, arch)
        # Append experiment name for disambiguation when multiple runs of same arch
        if len([r for r in selected if r.get("model") == arch]) > 1:
            label = f"{label}\n({row['experiment']})"

        values = []
        for key in metric_keys:
            try:
                values.append(float(row.get(key, 0) or 0))
            except ValueError:
                values.append(0.0)

        bars = ax.bar(x + offsets[i], values, width * 0.92,
                      label=label, color=color, alpha=0.9,
                      edgecolor="#0d1117", linewidth=0.5)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f"{val:.1f}",
                    ha="center", va="bottom",
                    fontsize=8, color="white",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color="white", fontsize=10)
    ax.set_ylabel("IoU / mIoU (%)", color="white")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#30363d")
    ax.yaxis.grid(True, color="#30363d", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=10, loc="upper right")
    ax.set_title("Architecture comparison — same protocol (Exp 4)",
                 color="white", fontsize=12, pad=12)

    plt.tight_layout()
    out_path = FIGURE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

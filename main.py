"""
main.py
───────
Quick entry point for step-by-step exploration.
Run this file to load a Paris-Lille-3D scan and launch interactive visualizations.

Usage:
    python main.py --file data/raw/Lille1.ply --mode labels
    python main.py --file data/raw/Lille1.ply --mode height --no-interactive
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path (needed when running as a script)
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LiDAR Semantic Segmentation — Step 1: Load & Visualize"
    )
    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to a .ply / .las / .laz file",
    )
    parser.add_argument(
        "--mode", type=str, default="labels",
        choices=["labels", "reflectance", "height"],
        help="Coloring mode for the visualization",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip Open3D interactive viewer (useful in headless environments)",
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save static figures to outputs/figures/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    # ── Step 1a: Load ─────────────────────────────────────────────────────
    from src.data.loader import load_pointcloud
    logger.info("═" * 60)
    logger.info("STEP 1 — Load point cloud")
    logger.info("═" * 60)

    pc = load_pointcloud(file_path)
    logger.info(f"  {pc}")

    if pc.has_labels:
        import numpy as np
        from src.data.loader import CLASS_NAMES, NUM_CLASSES
        counts = np.bincount(pc.labels, minlength=NUM_CLASSES)
        logger.info("  Class counts:")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
            pct = count / pc.num_points * 100
            logger.info(f"    [{i}] {name:<12} : {count:>10,}  ({pct:5.1f}%)")

    # ── Step 1b: Static figures ────────────────────────────────────────────
    from src.visualization.visualizer import (
        plot_class_distribution,
        plot_class_samples,
        plot_topdown,
    )

    save_dir = Path("outputs/figures") if args.save_figures else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating top-down view …")
    plot_topdown(
        pc, mode=args.mode,
        save_path=save_dir / f"topdown_{args.mode}.png" if save_dir else None,
    )

    if pc.has_labels:
        logger.info("Generating class distribution chart …")
        plot_class_distribution(
            pc,
            save_path=save_dir / "class_distribution.png" if save_dir else None,
        )

        logger.info("Generating per-class sample grid …")
        plot_class_samples(
            pc,
            save_path=save_dir / "class_samples.png" if save_dir else None,
        )

    # Show matplotlib figures
    import matplotlib.pyplot as plt
    plt.show()

    # ── Step 1c: Interactive 3D viewer ────────────────────────────────────
    if not args.no_interactive:
        from src.visualization.visualizer import show_pointcloud
        logger.info(f"Launching Open3D viewer (mode={args.mode}) …")
        logger.info("  Controls: drag=rotate, scroll=zoom, shift+drag=pan, Q=quit")
        show_pointcloud(pc, mode=args.mode, window_name=f"Paris-Lille-3D — {file_path.name}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

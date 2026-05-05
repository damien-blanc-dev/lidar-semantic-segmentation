"""
scripts/preprocess.py
─────────────────────
CLI to run the full preprocessing pipeline on Paris-Lille-3D training files.

Reads from : D:/Forge/Data/Benchmark/Benchmark/training_10_classes/
Writes to  : data/processed/

Usage:
    # Full run (all 4 files)
    python scripts/preprocess.py

    # Single file — useful for a quick test
    python scripts/preprocess.py --file Lille1_1.ply

    # Re-process even if output already exists
    python scripts/preprocess.py --overwrite

    # Dry-run: print what would be done without processing
    python scripts/preprocess.py --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path so `src` is importable when running scripts/ directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default paths (from configs/default.yaml) ─────────────────────────────────
TRAINING_DIR = Path("D:/Forge/Data/Benchmark/Benchmark/training_10_classes")
OUTPUT_DIR   = Path("data/processed")

TRAIN_FILES = ["Lille1_1.ply", "Lille1_2.ply", "Lille2.ply"]
VAL_FILES   = ["Paris.ply"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paris-Lille-3D preprocessing pipeline"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Process a single file by name (e.g. Lille1_1.ply). Default: all files.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=str(TRAINING_DIR),
        help=f"Directory containing the training .ply files. Default: {TRAINING_DIR}",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help=f"Root output directory. Default: {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.05,
        help="Voxel grid leaf size in meters (default: 0.05)",
    )
    parser.add_argument(
        "--normal_radius", type=float, default=0.3,
        help="Radius for normal estimation in meters (default: 0.3)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process files even if output already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print files that would be processed without doing anything.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Build the list of files to process
    all_files = TRAIN_FILES + VAL_FILES
    if args.file:
        if args.file not in all_files:
            # Accept absolute or relative paths too
            candidate = Path(args.file)
            if candidate.exists():
                files_to_process = [candidate]
            else:
                logger.error(
                    f"Unknown file: {args.file}. "
                    f"Expected one of: {all_files} or a valid path."
                )
                sys.exit(1)
        else:
            files_to_process = [input_dir / args.file]
    else:
        files_to_process = [input_dir / f for f in all_files]

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN — files that would be processed:")
        for f in files_to_process:
            out = output_dir / f.stem
            status = "EXISTS" if (out / "points.npy").exists() else "MISSING"
            logger.info(f"  {f.name:<20}  →  {out}  [{status}]")
        return

    # ── Run preprocessing ─────────────────────────────────────────────────
    from src.preprocessing.pipeline import preprocess_file
    from src.data.loader import CLASS_NAMES

    t_start = time.time()
    results = []

    for ply_path in files_to_process:
        if not ply_path.exists():
            logger.warning(f"File not found, skipping: {ply_path}")
            continue

        result = preprocess_file(
            input_path=ply_path,
            output_dir=output_dir,
            voxel_size=args.voxel_size,
            normal_radius=args.normal_radius,
            use_fast_downsample=True,
            overwrite=args.overwrite,
        )
        result["file"] = ply_path.name
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    processed = [r for r in results if not r.get("skipped", False)]

    logger.info("═" * 60)
    logger.info(f"SUMMARY — {len(results)} files, {len(processed)} processed")
    logger.info(f"Total time: {total_elapsed/60:.1f} min")
    logger.info("═" * 60)

    import numpy as np
    for r in results:
        status = "SKIPPED" if r.get("skipped") else f"{r['elapsed_total']:.0f}s"
        logger.info(f"  {r['file']:<20}  {r['n_points']:>10,} pts  [{status}]")

    # Aggregate class distribution across all processed files
    all_counts = [
        r["class_counts"] for r in processed
        if r.get("class_counts") is not None
    ]
    if all_counts:
        total_counts = np.sum(all_counts, axis=0)
        total_pts = total_counts.sum()
        logger.info("\nAggregate class distribution (training set):")
        for i, (name, count) in enumerate(zip(CLASS_NAMES, total_counts)):
            logger.info(f"  [{i}] {name:<20} {count:>12,}  ({count/total_pts*100:5.1f}%)")

    logger.info(f"\nOutput directory: {output_dir.resolve()}")
    logger.info("Ready for training. Next step: python scripts/train.py")


if __name__ == "__main__":
    main()

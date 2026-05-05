"""
scripts/download_data.py
────────────────────────
Helper to download and verify the Paris-Lille-3D dataset.

The dataset requires a registration form at http://npm3d.fr/paris-lille-3d
There is no public direct download link — this script guides the user and
validates files once manually placed in the target directory.

Usage:
    python scripts/download_data.py --check data/raw/
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Expected files in Paris-Lille-3D (training + test splits)
EXPECTED_FILES = [
    "Lille1_1.ply",
    "Lille1_2.ply",
    "Lille2.ply",
    "Paris.ply",
]

DOWNLOAD_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║         Paris-Lille-3D — Download Instructions                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: http://npm3d.fr/paris-lille-3d                        ║
║  2. Fill in the registration form (name, institution, email)     ║
║  3. Download the annotated PLY files (~3 GB total)               ║
║  4. Place the .ply files in: data/raw/                           ║
║                                                                  ║
║  Expected files:                                                 ║
║    - Lille1_1.ply  (~300M points, Lille scan part 1)             ║
║    - Lille1_2.ply  (~300M points, Lille scan part 2)             ║
║    - Lille2.ply    (~250M points, Lille scan 2)                  ║
║    - Paris.ply     (~350M points, Paris scan)                    ║
║                                                                  ║
║  Tip: start with Lille1_1.ply for development — it is the       ║
║  most commonly used split in published benchmarks.               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def check_files(data_dir: Path) -> bool:
    """Check which expected files are present in data_dir."""
    logger.info(f"Checking data directory: {data_dir.resolve()}")
    found = []
    missing = []

    for fname in EXPECTED_FILES:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            logger.info(f"  ✓ {fname} ({size_mb:.0f} MB)")
            found.append(fname)
        else:
            logger.warning(f"  ✗ {fname} — NOT FOUND")
            missing.append(fname)

    # Also list any unexpected .ply files
    other_ply = [
        f for f in data_dir.glob("*.ply")
        if f.name not in EXPECTED_FILES
    ]
    if other_ply:
        logger.info(f"  Other .ply files found: {[f.name for f in other_ply]}")

    logger.info(f"\n  {len(found)}/{len(EXPECTED_FILES)} expected files present.")
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Paris-Lille-3D dataset helper")
    parser.add_argument("--check", type=str, metavar="DIR",
                        help="Check which files are present in DIR")
    parser.add_argument("--instructions", action="store_true",
                        help="Print download instructions")
    args = parser.parse_args()

    if args.instructions or (not args.check):
        print(DOWNLOAD_INSTRUCTIONS)

    if args.check:
        data_dir = Path(args.check)
        data_dir.mkdir(parents=True, exist_ok=True)
        ok = check_files(data_dir)
        if not ok:
            print(DOWNLOAD_INSTRUCTIONS)
            sys.exit(1)
        logger.info("All files present. Ready to preprocess.")


if __name__ == "__main__":
    main()

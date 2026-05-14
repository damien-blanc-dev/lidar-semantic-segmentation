"""
src/utils/results_logger.py
────────────────────────────
Append one CSV row to outputs/results.csv after each training or inference run.

Schema is fixed so every experiment type maps to the same table — missing
fields are written as empty strings, not NaN, to keep the CSV readable.

All IoU / mIoU / OA values are expected in *percent* (0–100).
"""

import csv
from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path("outputs/results.csv")

COLUMNS = [
    "timestamp",
    "experiment",
    "variant",
    "model",
    "blocksize",
    "numpoints",
    "loss",
    "normal_radius",
    "mIoU",
    "OA",
    "pedestrian_iou",
    "bollard_iou",
    "polesign_iou",
    "trashcan_iou",
    "params",
    "train_time_s",
    "inference_time_s",
]


def log_result(row: dict, results_path: str | Path = RESULTS_PATH) -> None:
    """Append one row to the experiment results CSV.

    Unknown keys are silently dropped; missing keys are written as "".
    Creates the file (with header) if it does not exist yet.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not results_path.exists()
    row.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in COLUMNS})

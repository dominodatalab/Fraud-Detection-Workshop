"""TEMPLATE — Ground-truth generator: mature the hidden labels the load generator logged and
emit ground-truth batches so Domino model-quality metrics populate.

Model-agnostic scaffolding (reads the scoring log, matures rows older than a window, writes a
batch keyed on the row identifier). Implement `to_ground_truth(row)` for your label encoding.
Registering the batch with DMM needs an S3 datasource (DOMINO_PLATFORM_NOTES.md §5c).

Usage:  python scripts/ground_truth.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

WINDOW_H = config.get("traffic.maturation_window_hours", env="MATURATION_WINDOW_HOURS", default=6, cast=float)
LOG_DIR = config.scoring_log_dir()
LOG_CSV = os.path.join(LOG_DIR, "scoring_log.csv")
GT_DIR = os.path.join(LOG_DIR, "ground_truth_batches")


def to_ground_truth(row: dict):
    """Map a matured scoring-log row to its ground-truth label, in the SAME space as the
    prediction output DMM matches on (e.g. the decision the model made). Customize."""
    # e.g. return "decline" if int(row["true_label"]) == 1 else "approve"
    raise NotImplementedError


def main() -> int:
    if not os.path.exists(LOG_CSV):
        print("no scoring log yet"); return 0
    rows = list(csv.DictReader(open(LOG_CSV)))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_H)
    matured = []
    for r in rows:
        if str(r.get("matured")) == "1":
            continue
        try:
            ts = datetime.strptime(r["application_timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts <= cutoff and r.get("true_label") not in (None, ""):
            matured.append({"gt_uuid": r["row_identifier"], "y_gt": to_ground_truth(r)})
            r["matured"] = "1"
    if not matured:
        print(f"nothing older than {WINDOW_H}h to mature"); return 0

    os.makedirs(GT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(GT_DIR, f"ground_truth_batch_{stamp}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gt_uuid", "y_gt"]); w.writeheader(); w.writerows(matured)
    with open(LOG_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"matured {len(matured)} labels -> {GT_DIR}")
    print("Register with DMM: PUT /model/{id}/register-dataset/ground_truth "
          "(y_gt->ground_truth forPredictionOutput=<your prediction>, gt_uuid->row_identifier) — needs S3 datasource.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

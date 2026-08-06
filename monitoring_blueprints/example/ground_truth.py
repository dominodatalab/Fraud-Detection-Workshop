"""Filled instantiation of templates/ground_truth.py for the blueprint worked example."""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
# Self-locate this model's own config.yaml — see load_generator.py for why (avoid a
# project-wide config-path env var colliding with other models/jobs in the same project).
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
import config

WINDOW_H = config.get("traffic.maturation_window_hours", env="MATURATION_WINDOW_HOURS", default=6, cast=float)
LOG_DIR = config.scoring_log_dir()
LOG_CSV = os.path.join(LOG_DIR, "scoring_log.csv")
GT_DIR = os.path.join(LOG_DIR, "ground_truth_batches")


def to_ground_truth(row: dict):
    # decision-space label matching the model's "decision" prediction output
    return "yes" if int(row["true_label"]) == 1 else "no"


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create the Domino TrainingSet used as the Model Monitoring baseline — fully generic,
driven by config.yaml's `model.features`/`model.predictions` (no project-specific schema file).

CRITICAL (DOMINO_PLATFORM_NOTES.md §5a): the baseline must contain ONLY columns that are
actually present in the endpoint's prediction capture — the features + prediction outputs
(+ row identifier + timestamp). Adding the training target, a ground-truth column, or any
other uncaptured column causes DMM to fail every ingestion with
"[UNRESOLVED_COLUMN.WITH_SUGGESTION] <col> cannot be resolved".

This script therefore does NOT include a target column at all — a monitoring baseline needs
only the feature/prediction distributions, not a supervised target.

Usage:
  python scripts/create_baseline_trainingset.py --data path/to/training_data.csv \\
      --score-fn mymodule:score_dataframe
  (or import and call `build_baseline(df, score_fn)` from your own training script)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config

from domino_data.training_sets import client as ts_client, model as ts_model

TRAINING_SET_NAME = config.get("monitoring.baseline_trainingset_name",
                               env="BASELINE_TRAININGSET_NAME", default="model_baseline")


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """df must already contain: every feature name, every prediction name, row_identifier,
    timestamp. (Score your training data with your model BEFORE calling this — see
    example/create_baseline.py for a worked instantiation.)"""
    features = config.get("model.features", default=[])
    predictions = config.get("model.predictions", default=[])
    row_id = config.get("model.row_identifier", default="row_identifier")
    ts_col = config.get("model.timestamp", default="application_timestamp")

    cols = [f["name"] for f in features] + [p["name"] for p in predictions] + [row_id, ts_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: baseline dataframe is missing columns: {missing}")
    baseline = df[cols].copy()

    categorical_cols = [f["name"] for f in features if f["valueType"] == "categorical"]
    categorical_cols += [p["name"] for p in predictions if p["valueType"] == "categorical"]

    monitoring_meta = ts_model.MonitoringMeta(
        timestamp_columns=[ts_col],
        categorical_columns=categorical_cols,
    )
    version = ts_client.create_training_set_version(
        training_set_name=TRAINING_SET_NAME,
        df=baseline,
        description=f"Monitoring baseline (capture-aligned) for {config.get('model.registered_model_name')}.",
        key_columns=[row_id],
        target_columns=[],  # no target — this is a monitoring baseline, not a training set
        monitoring_meta=monitoring_meta,
        meta={"purpose": "monitoring_baseline"},
    )
    print(f"Created TrainingSet '{TRAINING_SET_NAME}' version {getattr(version, 'number', '?')} "
          f"with {len(baseline)} rows, columns: {cols}")
    return baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV already containing features + predictions")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    build_baseline(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

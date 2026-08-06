"""Build the capture-aligned monitoring baseline TrainingSet for CreditFraudModel.

Scores the project's transformed training data through the registered pyfunc, attaches
the prediction outputs + row id + timestamp, drops the `Class` target (must NOT be in a
monitoring baseline — causes UNRESOLVED_COLUMN), and registers it via the blueprint's
generic build_baseline(). Config (feature/prediction schema, trainingset name) comes from
monitoring_blueprints/config.yaml.
"""
from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import pandas as pd
import mlflow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
import config
from create_baseline_trainingset import build_baseline

TRANSFORMED = "/mnt/data/fraud-detection-workshop/transformed_cc_transactions.csv"
SAMPLE_ROWS = int(os.getenv("BASELINE_SAMPLE_ROWS", "8000"))


def main() -> int:
    feat_names = [f["name"] for f in config.get("model.features", default=[])]
    row_id = config.get("model.row_identifier", default="event_id")
    ts_col = config.get("model.timestamp", default="application_timestamp")

    df = pd.read_csv(TRANSFORMED)
    if SAMPLE_ROWS and len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    missing = [c for c in feat_names if c not in df.columns]
    if missing:
        raise SystemExit(f"training data missing model features: {missing}")

    model = mlflow.pyfunc.load_model("models:/CreditFraudModel/2")
    preds = np.asarray(model.predict(df[feat_names])).ravel().astype(int)

    out = df[feat_names].copy()
    out["prediction"] = preds
    out["decision"] = np.where(preds == 1, "FRAUD", "LEGIT")
    out[row_id] = [str(uuid.uuid4()) for _ in range(len(out))]
    # spread timestamps across the last 30 days so the baseline has a valid time column
    base = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30)
    out[ts_col] = (base + pd.to_timedelta(
        np.random.RandomState(42).randint(0, 30 * 24 * 3600, len(out)), unit="s")
    ).astype("datetime64[us]").astype(str)

    print(f"scored {len(out)} rows | FRAUD={int((preds==1).sum())} LEGIT={int((preds==0).sum())}")
    print(f"baseline columns ({len(out.columns)}):", list(out.columns))
    build_baseline(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

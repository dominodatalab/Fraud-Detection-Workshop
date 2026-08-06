"""Score the example's training data with the trained model and create the capture-aligned
baseline TrainingSet via scripts/create_baseline_trainingset.py.

Usage:  python monitoring_blueprints/example/create_baseline.py
"""
import os
import sys
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, HERE)
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))

from create_baseline_trainingset import build_baseline
from model_pyfunc import CAT_ENCODING


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "training_data.csv"))
    model = joblib.load(os.path.join(HERE, "model.joblib"))

    X = pd.DataFrame({"feature_a": df["feature_a"], "feature_b": df["feature_b"],
                      "feature_c": df["feature_c"].map(CAT_ENCODING)})
    p = model.predict_proba(X)[:, 1]

    baseline_df = df[["feature_a", "feature_b", "feature_c"]].copy()
    baseline_df["prediction"] = p.round(6)
    baseline_df["decision"] = ["yes" if v >= 0.5 else "no" for v in p]
    baseline_df["row_identifier"] = [str(uuid.uuid4()) for _ in range(len(df))]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    baseline_df["application_timestamp"] = now

    build_baseline(baseline_df)


if __name__ == "__main__":
    main()

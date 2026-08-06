"""Register a capture-instrumented version of CreditFraudModel.

Loads the already-fitted sklearn model from CreditFraudModel v2 (no retrain — identical
predictions), wraps it in MonitoredModel (Domino prediction capture baked in), and logs it as a
new registered version so it can be deployed as a MONITORED endpoint that actually captures
prediction data. See DOMINO_PLATFORM_NOTES.md §6.
"""
from __future__ import annotations

import os
import sys

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from creditfraud_model_pyfunc import MonitoredModel, FEATURE_NAMES

REGISTERED_MODEL_NAME = "CreditFraudModel"
EXPERIMENT_NAME = "CreditFraudModel"
SOURCE_VERSION = 2  # reuse the fitted sklearn estimator from this version


def main() -> int:
    # 1) pull the already-fitted sklearn estimator and persist it as an artifact
    sk = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{SOURCE_VERSION}")
    model_path = os.path.join(HERE, "creditfraud_model.joblib")
    joblib.dump(sk, model_path)

    # 2) build a signature that includes row_identifier so it flows through to capture
    pool = pd.read_csv(os.path.join(HERE, "data", "creditfraud_pool.csv"))
    ex = pool[FEATURE_NAMES].head(2).astype(float).copy()
    ex["row_identifier"] = ["ex-1", "ex-2"]
    out = pd.DataFrame({"row_identifier": ["ex-1", "ex-2"], "prediction": [0, 1],
                        "decision": ["LEGIT", "FRAUD"]})
    signature = infer_signature(ex, out)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="creditfraud_capture_wrap"):
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MonitoredModel(),
            artifacts={"model": model_path},
            code_paths=[os.path.join(HERE, "creditfraud_model_pyfunc.py")],
            signature=signature,
            input_example=ex,
            pip_requirements=["mlflow==3.3.2", "scikit-learn==1.7.2",
                              "pandas==2.3.2", "numpy==2.4.6", "joblib"],
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    print("logged:", info.model_uri)

    # 3) sanity: load back as pyfunc and predict (capture is a no-op off-endpoint)
    m = mlflow.pyfunc.load_model(info.model_uri)
    res = m.predict(ex)
    print("local predict:\n", res.to_string(index=False))
    from mlflow.tracking import MlflowClient
    latest = max(int(v.version) for v in MlflowClient().search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"))
    print("NEW REGISTERED VERSION:", latest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

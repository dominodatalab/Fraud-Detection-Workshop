"""Train + register the blueprint worked-example model as an MLflow pyfunc with capture.

Usage:  python monitoring_blueprints/example/train.py
"""
import os
import sys

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
import config
from model_pyfunc import MonitoredModel, FEATURE_NAMES, CAT_ENCODING

REGISTERED_MODEL_NAME = config.get("model.registered_model_name", default="blueprint_test_model")
EXPERIMENT_NAME = config.get("model.experiment_name", default=REGISTERED_MODEL_NAME)


def main():
    df = pd.read_csv(os.path.join(HERE, "data", "training_data.csv"))
    X = pd.DataFrame({
        "feature_a": df["feature_a"], "feature_b": df["feature_b"],
        "feature_c": df["feature_c"].map(CAT_ENCODING),
    })
    y = df["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression().fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"holdout AUC: {auc:.3f}")

    model_path = os.path.join(HERE, "model.joblib")
    joblib.dump(model, model_path)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="blueprint_example_train") as run:
        mlflow.log_metric("auc", auc)
        example = pd.DataFrame({
            "feature_a": [50.0, 80.0], "feature_b": [20.0, 20.0], "feature_c": ["cat_a", "cat_c"],
            "row_identifier": ["ex-1", "ex-2"],
        })
        out = pd.DataFrame({"row_identifier": ["ex-1", "ex-2"],
                            "prediction": [0.1, 0.6], "decision": ["no", "yes"]})
        from mlflow.models import infer_signature
        signature = infer_signature(example, out)
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MonitoredModel(),
            artifacts={"model": model_path},
            code_paths=[os.path.join(HERE, "model_pyfunc.py")],
            signature=signature,
            input_example=example,
            pip_requirements=["mlflow==3.2.0", "scikit-learn==1.5.1", "pandas==2.2.1", "numpy==1.26.4"],
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        print(f"Logged pyfunc: {info.model_uri}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()

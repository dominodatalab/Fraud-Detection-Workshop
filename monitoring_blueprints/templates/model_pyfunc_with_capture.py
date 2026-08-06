"""TEMPLATE — MLflow pyfunc with Domino prediction capture baked in.

This is the capture contract that makes a registry-deployed endpoint monitorable.
Copy into your model repo, fill in `_score`, log with mlflow.pyfunc.log_model + an explicit
signature (NOT sklearn/xgboost flavors — they break Domino endpoint deploy).
See DOMINO_PLATFORM_NOTES.md §2, §6.

Key rules:
  - predict_names MUST include "ground_truth" with a sentinel, or DMM ingestion fails with
    UNRESOLVED_COLUMN: ground_truth cannot be resolved.
  - event_id = your row identifier = the ground-truth join key.
  - capture on EVERY request; never let a capture error break scoring.
"""
import json
import os
import uuid

import mlflow.pyfunc
import pandas as pd

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c"]     # <-- your captured features
PREDICT_NAMES = ["prediction", "decision", "ground_truth"]  # include ground_truth (sentinel)
GT_SENTINEL = "PENDING"                                      # or -1.0 for numeric targets


class MonitoredModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # self.model = joblib.load(context.artifacts["model"])   # <-- load your artifact
        self.capture = None
        try:
            from domino_data_capture.data_capture_client import DataCaptureClient
            self.capture = DataCaptureClient(feature_names=FEATURE_NAMES, predict_names=PREDICT_NAMES)
            print(f"[capture] ready. is_dev_mode={getattr(self.capture,'is_dev_mode','?')}")  # False == real endpoint
        except Exception as e:
            print(f"[capture] disabled: {e}")

    def _score(self, row: dict):
        # <-- YOUR MODEL. Return (prediction, decision).
        raise NotImplementedError

    def predict(self, context, model_input):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(
            [model_input] if isinstance(model_input, dict) else model_input)
        out = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            rid = str(row.get("row_identifier") or uuid.uuid4())
            prediction, decision = self._score(row)
            if self.capture is not None:
                try:
                    fvals = [row[f] for f in FEATURE_NAMES]
                    self.capture.capturePrediction(fvals, [prediction, decision, GT_SENTINEL], event_id=rid)
                except Exception as e:
                    print(f"[capture] capturePrediction failed: {e}")
            out.append({"row_identifier": rid, "prediction": prediction, "decision": decision})
        return pd.DataFrame(out)

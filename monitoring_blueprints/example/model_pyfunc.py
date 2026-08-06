"""Concrete instantiation of templates/model_pyfunc_with_capture.py for the worked example.
A tiny sklearn LogisticRegression over feature_a/feature_b/feature_c(encoded), with Domino
prediction capture baked into predict(). See DOMINO_PLATFORM_NOTES.md §2c, §6.
"""
import json
import os
import uuid

import joblib
import mlflow.pyfunc
import pandas as pd

FEATURE_NAMES = ["feature_a", "feature_b", "feature_c"]
PREDICT_NAMES = ["prediction", "decision", "ground_truth"]  # ground_truth sentinel required
GT_SENTINEL = "PENDING"
CAT_ENCODING = {"cat_a": 0, "cat_b": 1, "cat_c": 2}


class MonitoredModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.capture = None
        try:
            from domino_data_capture.data_capture_client import DataCaptureClient
            self.capture = DataCaptureClient(feature_names=FEATURE_NAMES, predict_names=PREDICT_NAMES)
            print(f"[capture] ready. is_dev_mode={getattr(self.capture,'is_dev_mode','?')}")
        except Exception as e:
            print(f"[capture] disabled: {e}")

    def _score(self, row: dict):
        c_enc = CAT_ENCODING.get(row["feature_c"], 0)
        x = pd.DataFrame([{"feature_a": float(row["feature_a"]), "feature_b": float(row["feature_b"]),
                          "feature_c": float(c_enc)}])
        prob = float(self.model.predict_proba(x)[0][1])
        decision = "yes" if prob >= 0.5 else "no"
        return round(prob, 6), decision

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

"""Capture-instrumented MLflow pyfunc for CreditFraudModel.

Wraps the fitted sklearn model (logged as CreditFraudModel v2) and bakes in Domino prediction
capture: on every request it calls DataCaptureClient.capturePrediction(features, [prediction,
decision, ground_truth], event_id) so the endpoint writes the DMM capture parquet. Without this,
a registry-deployed model captures NOTHING (DOMINO_PLATFORM_NOTES.md §6).

prediction = fraud class label (0/1), decision = FRAUD/LEGIT, ground_truth = sentinel (joined
later on event_id). Feature/predict names match the DMM baseline (creditfraud_baseline).
"""
import uuid

import joblib
import mlflow.pyfunc
import pandas as pd

FEATURE_NAMES = ["num__Time", "num__Amount", "num__Age", "num__Tenure", "num__MerchantRisk", "num__DeviceTrust", "num__Txn24h", "num__Avg30d", "num__IPReputation", "num__Latitude", "num__Longitude", "num__DistFromHome", "num__Hour", "num__CardPresent", "num__amount_vs_avg30d_ratio", "num__risk_score", "num__trust_score", "cat__TxType_payment", "cat__TxType_purchase", "cat__TxType_transfer", "cat__TxType_withdrawal", "cat__DeviceType_ATM", "cat__DeviceType_POS", "cat__DeviceType_desktop", "cat__DeviceType_mobile", "cat__DeviceType_web", "cat__MerchantCat_clothing", "cat__MerchantCat_electronics", "cat__MerchantCat_entertainment", "cat__MerchantCat_gas", "cat__MerchantCat_grocery", "cat__MerchantCat_restaurant", "cat__MerchantCat_travel", "cat__MerchantCat_utilities", "cat__Channel_chip", "cat__Channel_contactless", "cat__Channel_in-store", "cat__Channel_online", "cat__generation_Baby Boomer", "cat__generation_Generation X", "cat__generation_Generation Z", "cat__generation_Millennial"]
PREDICT_NAMES = ["prediction", "decision", "ground_truth"]
GT_SENTINEL = "PENDING"


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

    def predict(self, context, model_input):
        df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(
            [model_input] if isinstance(model_input, dict) else model_input)
        X = df[FEATURE_NAMES].astype(float)
        preds = self.model.predict(X)
        out = []
        for i in range(len(df)):
            row = df.iloc[i]
            rid = str(row["row_identifier"]) if "row_identifier" in df.columns and pd.notna(row.get("row_identifier")) else str(uuid.uuid4())
            prediction = int(preds[i])
            decision = "FRAUD" if prediction == 1 else "LEGIT"
            if self.capture is not None:
                try:
                    fvals = [float(row[f]) for f in FEATURE_NAMES]
                    self.capture.capturePrediction(fvals, [prediction, decision, GT_SENTINEL], event_id=rid)
                except Exception as e:
                    print(f"[capture] capturePrediction failed: {e}")
            out.append({"row_identifier": rid, "prediction": prediction, "decision": decision})
        return pd.DataFrame(out)

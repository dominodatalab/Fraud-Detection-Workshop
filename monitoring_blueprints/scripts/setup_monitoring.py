"""Configure Domino Model Monitoring (DMM) for a model — via the DMM REST API.

The DMM service is NOT on the workspace API proxy and is absent from the public/`/v4` specs.
It lives at `{DOMINO_DOMAIN}/model-monitor/v2/api/` and authenticates with the Domino API key
in the `X-Domino-Api-Key` header. Endpoints + payload shapes were confirmed against the DMM
frontend's own API client bundle and validated live. See DOMINO_PLATFORM_NOTES.md §5.

Endpoints used:
  GET  /model-monitor/v2/api/models?pageNumber&pageSize&numberOfLastChecksToFetch   list
  GET  /model-monitor/v2/api/model?model_id=<id>                                     get one
  PUT  /model-monitor/v2/api/model                                                   register model
  PUT  /model-monitor/v2/api/model/{model_id}/register-dataset/{dataset_type}        register dataset
  POST /model-monitor/v2/api/model/{model_id}/analyse-drift                          on-demand drift
  POST /model-monitor/v2/api/model/{model_id}/save-scheduler-query                   save drift config
  POST /model-monitor/api/scheduler/create_drift_schedule_check                      activate schedule

Usage:
  python scripts/setup_monitoring.py --list                    # live: list monitored models
  python scripts/setup_monitoring.py --get <dmm_model_id>       # live: show one model
  python scripts/setup_monitoring.py                            # DRY RUN: print register payloads
  python scripts/setup_monitoring.py --execute                  # actually register (writes!)
  python scripts/setup_monitoring.py --configure-drift <id>     # save PSI config on drift_features
  python scripts/setup_monitoring.py --schedule-drift <id>      # activate the daily drift schedule

The feature/prediction schema is read from config.yaml (`model.features`) — NOT a project-specific
file — so this script is portable across models with only config.yaml changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import config

# All settings resolved centrally (env > config.yaml > default) — no instance/model hard-codes.
DOMAIN = config.domain()
DMM = f"{DOMAIN}/model-monitor/v2/api"
API_KEY = os.getenv("DOMINO_USER_API_KEY", "")
HEADERS = {"X-Domino-Api-Key": API_KEY, "Content-Type": "application/json"}

# Workbench (Model API) identifiers this monitored model attaches to (set after registry deploy).
WORKBENCH_MODEL_ID = config.get("endpoint.model_id", env="MODEL_ID", default="")
WORKBENCH_MODEL_VERSION_ID = config.get("endpoint.model_version_id", env="MODEL_VERSION_ID", default="")
MODEL_NAME = config.get("monitoring.monitor_model_name", env="MONITOR_MODEL_NAME",
                        default=config.get("model.registered_model_name", default="monitored_model"))

# DMM datasources are cloud object stores (s3/azure). Point these at a DMM datasource + path
# where the training/prediction/ground-truth CSVs live (per instance).
DATASOURCE_NAME = config.get("monitoring.datasource_name", env="MONITORING_DATASOURCE_NAME", default="")
DATASOURCE_TYPE = config.get("monitoring.datasource_type", env="MONITORING_DATASOURCE_TYPE", default="s3")
PRED_PATH = config.get("monitoring.prediction_path", env="PRED_PATH", default="monitoring/prediction_data/")
GT_PATH = config.get("monitoring.ground_truth_path", env="GT_PATH", default="monitoring/ground_truth_batches/")

# Features that carry the injected/expected drift — set PSI alerts on these. Empty by default:
# must be set in config.yaml (monitoring.drift_features) — no model-specific fallback here.
DRIFT_FEATURES = config.get("monitoring.drift_features", default=[])
DRIFT_ALGORITHM = config.get("monitoring.drift_algorithm", default="POPULATION STABILITY INDEX")
PSI_THRESHOLD = config.get("monitoring.psi_threshold", default=0.2, cast=float)


def _valtype(valueType: str) -> str:
    # DMM value types: "numerical" | "categorical" | "string"
    return {"numerical": "numerical", "categorical": "categorical", "string": "string"}.get(valueType, valueType)


def build_variables() -> list[dict]:
    """DMM variableConfig for a **domino_workbench** model = FEATURES ONLY, read from
    config.yaml's `model.features` (NOT a project-specific schema file).

    Hard-won lesson (verified live): for workbench-sourced models DMM derives the prediction
    outputs from the endpoint's own capture schema — passing a prediction variable here is
    rejected ("Variable <p> not allowed for registration"). And you must NOT declare the
    training target or any uncaptured column: DMM would try to resolve it against the
    prediction-capture parquet and fail with "[UNRESOLVED_COLUMN] <col> cannot be resolved".
    Ground truth is wired separately via register-dataset.
    """
    features = config.get("model.features", default=[])
    if not features:
        raise SystemExit("ERROR: model.features is empty in config.yaml — set the feature schema.")
    return [{"name": f["name"], "variableType": "feature", "valueType": _valtype(f["valueType"])}
            for f in features]


def model_registration_payload() -> dict:
    """modelRegistrationConfigRequest — shape verified live (PUT /model -> 200).
    Top level: modelMetadata + variables + datasetDetails (training baseline) + run_cohort_analysis."""
    return {
        "modelMetadata": {
            "name": MODEL_NAME,
            "modelType": "classification",
            "version": "1",
            "description": f"{MODEL_NAME} — registered via DMM API (monitoring_blueprints).",
            "sourceType": "domino_workbench",
            "sourceDetails": {
                "workbenchModelId": WORKBENCH_MODEL_ID,
                "workbenchModelVersionId": WORKBENCH_MODEL_VERSION_ID,
            },
        },
        "variables": build_variables(),
        "run_cohort_analysis": False,
        # Training/baseline dataset (a file in a DMM S3 datasource). The Domino workbench flow
        # stages the TrainingSet into DMM storage when you set it as Training Data in the UI;
        # this datasetDetails block is a placeholder satisfying the schema for a direct API
        # registration attempt (see DOMINO_PLATFORM_NOTES.md §5a for the UI-staging reality).
        "datasetDetails": {
            "name": f"{MODEL_NAME}_baseline", "datasetType": "file",
            "datasetConfig": {"path": os.getenv("TRAINING_PATH", "training_data.csv"), "fileFormat": "csv"},
            "datasourceName": DATASOURCE_NAME, "datasourceType": DATASOURCE_TYPE,
        },
    }


def dataset_payload(dataset_type: str) -> dict:
    """dataset_type ∈ {'prediction','ground_truth'}."""
    if dataset_type == "ground_truth":
        return {
            "variables": [
                {"name": "y_gt", "variableType": "ground_truth", "valueType": "categorical",
                 "forPredictionOutput": "decision"},
                {"name": "gt_uuid", "variableType": "row_identifier", "valueType": "string"},
            ],
            "datasetDetails": {
                "name": f"{MODEL_NAME}_ground_truth", "datasetType": "file",
                "datasetConfig": {"path": GT_PATH, "fileFormat": "csv"},
                "datasourceName": DATASOURCE_NAME, "datasourceType": DATASOURCE_TYPE,
            },
        }
    # prediction dataset: features (prediction outputs are derived from the endpoint capture)
    return {
        "variables": build_variables(),
        "datasetDetails": {
            "name": f"{MODEL_NAME}_predictions", "datasetType": "file",
            "datasetConfig": {"path": PRED_PATH, "fileFormat": "csv"},
            "datasourceName": DATASOURCE_NAME, "datasourceType": DATASOURCE_TYPE,
        },
    }


def list_models():
    r = requests.get(f"{DMM}/models",
                     params={"pageNumber": 0, "pageSize": 50, "numberOfLastChecksToFetch": 1},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    for m in r.json().get("modelDashboardItems", []):
        print(f"  {m['id']}  {m['name']!r:40s} {m['modelType']:14s} status={m.get('modelStatus')}")


def get_model(mid: str):
    r = requests.get(f"{DMM}/model", params={"model_id": mid}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def register(execute: bool):
    reg = model_registration_payload()
    pred = dataset_payload("prediction")
    gt = dataset_payload("ground_truth")
    if not execute:
        print("DRY RUN — payloads that WOULD be sent (use --execute to send):\n")
        print("PUT /model\n", json.dumps(reg, indent=2))
        print("\nPUT /model/{id}/register-dataset/prediction\n", json.dumps(pred, indent=2))
        print("\nPUT /model/{id}/register-dataset/ground_truth\n", json.dumps(gt, indent=2))
        print(f"\nThen set PSI>{PSI_THRESHOLD} drift alerts on {DRIFT_FEATURES} and schedule daily checks.")
        return
    r = requests.put(f"{DMM}/model", json=reg, headers=HEADERS, timeout=60)
    print("PUT /model ->", r.status_code, r.text[:300])
    r.raise_for_status()
    mid = r.json().get("id") or r.json().get("modelId")
    for dtype, payload in (("prediction", pred), ("ground_truth", gt)):
        rr = requests.put(f"{DMM}/model/{mid}/register-dataset/{dtype}", json=payload,
                          headers=HEADERS, timeout=60)
        print(f"register-dataset/{dtype} ->", rr.status_code, rr.text[:200])
    print(f"\nRegistered DMM model {mid}. Configure PSI>{PSI_THRESHOLD} on {DRIFT_FEATURES} + schedule checks.")


def find_monitor_for_endpoint(model_id: str) -> str | None:
    """Find the DMM monitor whose sourceDetails.workbenchModelId matches this endpoint."""
    r = requests.get(f"{DMM}/models", params={"pageNumber": 0, "pageSize": 100, "numberOfLastChecksToFetch": 1},
                     headers=HEADERS, timeout=30)
    for m in r.json().get("modelDashboardItems", []):
        det = requests.get(f"{DMM}/model", params={"model_id": m["id"]}, headers=HEADERS, timeout=30).json()
        if det.get("sourceDetails", {}).get("workbenchModelId") == model_id:
            return m["id"]
    return None


def configure_drift(monitor_id: str):
    """Save PSI (or configured algorithm) on DRIFT_FEATURES, keeping only variables that are
    actually on the monitor (avoids UNRESOLVED_COLUMN from stale/extra baseline columns)."""
    if not DRIFT_FEATURES:
        raise SystemExit("ERROR: monitoring.drift_features is empty in config.yaml.")
    cfg = requests.get(f"{DMM}/model/{monitor_id}/drift-scheduler-query", headers=HEADERS, timeout=30).json()
    feats = cfg.get("features", [])
    if not feats:
        raise SystemExit(f"ERROR: monitor {monitor_id} has no variables yet — "
                          "has the baseline Training Data been set on the endpoint version?")
    for f in feats:
        if f["name"] in DRIFT_FEATURES:
            f["algorithm"] = DRIFT_ALGORITHM
            f["condition"] = {"operator": "<", "lower": PSI_THRESHOLD, "upper": PSI_THRESHOLD}
            f["alertable"] = True
    r = requests.post(f"{DMM}/model/{monitor_id}/save-scheduler-query", headers=HEADERS,
                      json={"config": feats}, timeout=30)
    print("save-scheduler-query ->", r.status_code, r.text[:150])
    print(f"  configured {DRIFT_ALGORITHM} > {PSI_THRESHOLD} on: {[f for f in DRIFT_FEATURES if f in {x['name'] for x in feats}]}")


def schedule_drift(monitor_id: str, cron: str = "0 6 * * *", data_since_days: int = 30,
                   timezone: str = "America/New_York"):
    """Activate the daily scheduled drift check. data_since is a LOOK-BACK COUNT IN DAYS,
    NOT an epoch (DOMINO_PLATFORM_NOTES.md §5b). Requires prediction data already ingested on
    this monitor (400 'No prediction data registered' otherwise — wait for the first daily
    ingestion after traffic starts flowing)."""
    body = {"modelId": monitor_id, "name": f"{MODEL_NAME} drift check (daily)", "timezone": timezone,
            "cronExpression": cron, "calendarType": "D", "dataSinceLastCheck": False,
            "data_since": data_since_days, "dataSince": data_since_days}
    r = requests.post(f"{DOMAIN}/model-monitor/api/scheduler/create_drift_schedule_check",
                      headers=HEADERS, json=body, timeout=60)
    print("create_drift_schedule_check ->", r.status_code, r.text[:250])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", metavar="DMM_MODEL_ID")
    ap.add_argument("--execute", action="store_true", help="actually register (writes to DMM)")
    ap.add_argument("--configure-drift", metavar="DMM_MODEL_ID",
                    help="save PSI drift config on monitoring.drift_features")
    ap.add_argument("--schedule-drift", metavar="DMM_MODEL_ID",
                    help="activate the daily scheduled drift check")
    ap.add_argument("--find-monitor", action="store_true",
                    help="find the DMM monitor id for endpoint.model_id in config.yaml")
    args = ap.parse_args()
    if not API_KEY:
        print("ERROR: DOMINO_USER_API_KEY not set"); return 2
    if args.list:
        list_models()
    elif args.get:
        get_model(args.get)
    elif args.find_monitor:
        mid = find_monitor_for_endpoint(WORKBENCH_MODEL_ID)
        print(mid or "not found")
    elif args.configure_drift:
        configure_drift(args.configure_drift)
    elif args.schedule_drift:
        schedule_drift(args.schedule_drift)
    else:
        register(args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

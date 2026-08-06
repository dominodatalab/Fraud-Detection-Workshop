"""Filled load generator for CreditFraudModel (fraud-detection-workshop).

Puts sustained, business-hours traffic on the live endpoint so Domino captures predictions
for monitoring, with a baseline->drift shift on the configured drift_features. Samples real
rows from a training-data pool (preserves the 42-column one-hot structure), then shifts the
drift features upward once drift kicks in. See DOMINO_PLATFORM_NOTES.md §2c (pyfunc contract),
§6 (capture).

Runtime deps: config.yaml (endpoint URL+token, drift_features, traffic.*). Needs the endpoint
token in the CREDITFRAUD_ENDPOINT_AUTH_TOKEN project env var (set_project_token.py).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
import config

ENDPOINT_URL = config.endpoint_url()
TOKEN = config.endpoint_token()
REQUESTS_PER_RUN = config.get("traffic.requests_per_run", env="REQUESTS_PER_RUN", default=100, cast=int)
DRIFT_SWITCH_AFTER_RUN = config.get("traffic.drift_switch_after_run", env="DRIFT_SWITCH_AFTER_RUN", default=8, cast=int)
DRIFT_FEATURES = config.get("monitoring.drift_features", default=[])
FEATURES = [f["name"] for f in config.get("model.features", default=[])]
LOG_DIR = config.scoring_log_dir()
STATE = os.path.join(LOG_DIR, "generator_state.json")
LOG_CSV = os.path.join(LOG_DIR, "scoring_log.csv")

# Source rows to replay as traffic. Prefer a small committed pool if present; otherwise sample
# from the project's mounted transformed dataset (datasets are mounted in scheduled jobs). The
# pool CSV is gitignored (*.csv), so in a committed job this falls back to the dataset.
_POOL_LOCAL = os.path.join(HERE, "data", "creditfraud_pool.csv")
_POOL_DATASET = os.path.join(config.datasets_dir(), config.project_name(),
                             "transformed_cc_transactions.csv")
if os.path.exists(_POOL_LOCAL):
    POOL = pd.read_csv(_POOL_LOCAL)
elif os.path.exists(_POOL_DATASET):
    POOL = pd.read_csv(_POOL_DATASET).sample(n=2000, random_state=7).reset_index(drop=True)
else:
    raise SystemExit(f"no traffic source: neither {_POOL_LOCAL} nor {_POOL_DATASET} exists")
DRIFT_SHIFT = float(os.getenv("DRIFT_SHIFT_STD", "2.5"))  # std-multiples to push drift features

HOURLY = {h: m for h, m in enumerate(
    [.15, .1, .1, .1, .1, .15, .3, .5, .75, 1, 1.1, 1.15, 1.1, 1.1, 1.15, 1.15, 1.1, 1.05, .95, .85, .7, .55, .4, .25])}


def build_request(drift_state: str, rng) -> dict:
    """Sample a real transformed row (42 features); on drift, push the drift_features up."""
    row = POOL.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    req = {c: float(row[c]) for c in FEATURES}
    if drift_state == "drifted":
        for c in DRIFT_FEATURES:
            if c in req:
                sd = float(POOL[c].std()) or 1.0
                req[c] = req[c] + DRIFT_SHIFT * sd
    req["_true_label"] = int(row["Class"])  # carried out-of-band for ground truth (not sent)
    return req


def hidden_label(request: dict, drift_state: str, rng):
    """True fraud outcome for this row (the training label), matured later for model quality."""
    return int(request.get("_true_label", 0))


def call_endpoint(payload: dict) -> dict:
    r = requests.post(ENDPOINT_URL, json={"data": payload},
                      headers={"Content-Type": "application/json"},
                      auth=(TOKEN, TOKEN) if TOKEN else None, timeout=30)
    r.raise_for_status()
    result = r.json().get("result", {})
    if isinstance(result, list):
        return {"result_row": result[0] if result else None}
    return result


def main() -> int:
    if not ENDPOINT_URL:
        print("ERROR: endpoint URL not resolved (set endpoint.model_id in config)"); return 2
    if not TOKEN or TOKEN == os.getenv("DOMINO_USER_API_KEY"):
        print("WARNING: no dedicated endpoint token resolved — calls will 401 (see CREDITFRAUD_ENDPOINT_AUTH_TOKEN).")
    os.makedirs(LOG_DIR, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"run_number": 0}
    run = state["run_number"] + 1
    drift_state = "drifted" if run > DRIFT_SWITCH_AFTER_RUN else "baseline"
    rng = np.random.default_rng(1000 + run)
    now = datetime.now()
    volume = max(1, int(round(REQUESTS_PER_RUN * HOURLY.get(now.hour, 1.0))))

    ok = fail = 0
    rows = []
    for _ in range(volume):
        req = build_request(drift_state, rng)
        true_label = hidden_label(req, drift_state, rng)
        payload = {k: v for k, v in req.items() if not k.startswith("_")}  # strip out-of-band fields
        rid = str(uuid.uuid4())
        payload["row_identifier"] = rid
        payload["application_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec = {"row_identifier": rid, "application_timestamp": payload["application_timestamp"],
               "drift_state": drift_state, "true_label": true_label, "matured": 0}
        try:
            call_endpoint(payload); ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3:
                print("  request failed:", e)
        rows.append(rec)

    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    json.dump({"run_number": run, "drift_state": drift_state}, open(STATE, "w"))
    print(f"[load] run={run} drift={drift_state} volume={volume} ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Filled instantiation of templates/load_generator.py for the blueprint worked example
(blueprint_test_model — feature_a/feature_b/feature_c, baseline->drifted shift)."""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
# Self-locate this model's own config.yaml instead of relying on a project-wide MONITORING_CONFIG
# env var — that would affect every other job/workspace in the project too (a config-path
# override is as collision-prone as a secret when a project hosts more than one model).
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
import config

ENDPOINT_URL = config.endpoint_url()
TOKEN = config.endpoint_token()
REQUESTS_PER_RUN = config.get("traffic.requests_per_run", env="REQUESTS_PER_RUN", default=100, cast=int)
DRIFT_SWITCH_AFTER_RUN = config.get("traffic.drift_switch_after_run", env="DRIFT_SWITCH_AFTER_RUN", default=8, cast=int)
LOG_DIR = config.scoring_log_dir()
STATE = os.path.join(LOG_DIR, "generator_state.json")
LOG_CSV = os.path.join(LOG_DIR, "scoring_log.csv")

BASELINE = json.load(open(os.path.join(HERE, "data", "population_baseline.json")))
DRIFTED = json.load(open(os.path.join(HERE, "data", "population_drifted.json")))
CAT_ENCODING = {"cat_a": 0, "cat_b": 1, "cat_c": 2}

HOURLY = {h: m for h, m in enumerate(
    [.15, .1, .1, .1, .1, .15, .3, .5, .75, 1, 1.1, 1.15, 1.1, 1.1, 1.15, 1.15, 1.1, 1.05, .95, .85, .7, .55, .4, .25])}


def build_request(drift_state: str, rng) -> dict:
    profile = DRIFTED if drift_state == "drifted" else BASELINE
    feature_a = float(rng.normal(profile["feature_a_mean"], profile["feature_a_std"]))
    feature_b = float(rng.normal(profile["feature_b_mean"], profile["feature_b_std"]))
    cats = list(profile["cat_probs"].keys())
    probs = list(profile["cat_probs"].values())
    feature_c = str(rng.choice(cats, p=probs))
    return {"feature_a": feature_a, "feature_b": feature_b, "feature_c": feature_c}


def hidden_label(request: dict, drift_state: str, rng):
    enc = CAT_ENCODING.get(request["feature_c"], 0)
    logit = 0.04 * (request["feature_a"] - 50) + 0.08 * (request["feature_b"] - 20) - 0.5 * enc
    prob = 1 / (1 + np.exp(-logit))
    return int(rng.uniform(0, 1) < prob)


def call_endpoint(payload: dict) -> dict:
    r = requests.post(ENDPOINT_URL, json={"data": payload},
                      headers={"Content-Type": "application/json"},
                      auth=(TOKEN, TOKEN) if TOKEN else None, timeout=30)
    r.raise_for_status()
    result = r.json().get("result", {})
    if isinstance(result, list):  # registry/pyfunc returns list-of-rows
        return {"result_row": result[0] if result else None}
    return result


def main() -> int:
    if not ENDPOINT_URL:
        print("ERROR: endpoint URL not resolved (set endpoint.model_id in config or ENDPOINT_URL)"); return 2
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
        rid = str(uuid.uuid4())
        req["row_identifier"] = rid
        req["application_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec = {**req, "drift_state": drift_state, "true_label": hidden_label(req, drift_state, rng), "matured": 0}
        try:
            call_endpoint(req); ok += 1
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

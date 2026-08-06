"""TEMPLATE — Load generator: put sustained, business-hours traffic on the live endpoint so
Domino captures predictions for monitoring, with an injected baseline->drift shift.

Model-agnostic scaffolding (business-hours volume curve, drift-state counter, live-endpoint
call + normalized response). Implement the two model-specific hooks:
    build_request(drift_state, rng) -> dict     # one synthetic request payload
    hidden_label(request, drift_state, rng) -> Any  # true outcome, logged for ground truth later
Config comes from config.yaml / env (endpoint URL+token, requests_per_run, drift_switch_after_run).
See DOMINO_PLATFORM_NOTES.md §2c (pyfunc invocation contract), §6 (capture).
"""
from __future__ import annotations

import csv
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ENDPOINT_URL = config.endpoint_url()
TOKEN = config.endpoint_token()
REQUESTS_PER_RUN = config.get("traffic.requests_per_run", env="REQUESTS_PER_RUN", default=100, cast=int)
DRIFT_SWITCH_AFTER_RUN = config.get("traffic.drift_switch_after_run", env="DRIFT_SWITCH_AFTER_RUN", default=8, cast=int)
LOG_DIR = config.scoring_log_dir()
STATE = os.path.join(LOG_DIR, "generator_state.json")
LOG_CSV = os.path.join(LOG_DIR, "scoring_log.csv")

HOURLY = {h: m for h, m in enumerate(
    [.15,.1,.1,.1,.1,.15,.3,.5,.75,1,1.1,1.15,1.1,1.1,1.15,1.15,1.1,1.05,.95,.85,.7,.55,.4,.25])}


# ── implement these two for your model ───────────────────────────────────────
def build_request(drift_state: str, rng) -> dict:
    """Return ONE request payload dict (feature values). Sample from a 'drifted'
    distribution when drift_state == 'drifted'. Send bool flags as ints (pyfunc)."""
    raise NotImplementedError


def hidden_label(request: dict, drift_state: str, rng):
    """The true outcome for this request (not sent to the endpoint) — logged for the
    ground-truth job to reveal after a maturation delay."""
    raise NotImplementedError
# ─────────────────────────────────────────────────────────────────────────────


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
    import json
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

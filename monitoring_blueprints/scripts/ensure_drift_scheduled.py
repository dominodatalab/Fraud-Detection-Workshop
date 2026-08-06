"""Self-cleaning: activate the DMM drift schedule as soon as the platform allows it, then
delete the scheduled job that ran this script — no lingering polling job once the work is done.

Scheduling a drift check requires the monitor to have already been through Domino's daily prediction
ingestion (no on-demand trigger — see DOMINO_PLATFORM_NOTES.md §5b, §6). On a brand-new monitor this
fails for the first day or more with 400 "No prediction data registered". Rather than requiring a human
to remember to retry (or leaving a job polling forever), run THIS script as a **frequent** (e.g. every
10 min) scheduled job created by `create_scheduled_jobs.py`: it no-ops/retries harmlessly while
scheduling isn't possible yet, and the MOMENT it succeeds, it deletes its own scheduled job — closing
the loop with zero manual follow-up and zero lingering compute afterward.

Usage:  python scripts/ensure_drift_scheduled.py [monitor_id]
        (omit monitor_id to auto-resolve via config.yaml's endpoint.model_id)
"""
from __future__ import annotations

import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config
import domino_api as api
from setup_monitoring import configure_drift, schedule_drift, find_monitor_for_endpoint, WORKBENCH_MODEL_ID, MODEL_NAME

DOMAIN = config.domain()
HEADERS = {"X-Domino-Api-Key": os.getenv("DOMINO_USER_API_KEY", "")}
SELF_JOB_TITLE = f"{MODEL_NAME} - Ensure Drift Scheduled"


def _delete_self_scheduled_job() -> None:
    """Find the scheduled job that runs this script (by title) and delete it — its job is done."""
    pid = config.project_id()
    jobs = api.get(f"/v4/projects/{pid}/scheduledjobs").json()
    matches = [j for j in jobs if j.get("title") == SELF_JOB_TITLE]
    if not matches:
        print(f"[ensure_drift_scheduled] no scheduled job titled '{SELF_JOB_TITLE}' found to clean up "
              "(already removed, or created under a different title).")
        return
    for j in matches:
        r = api.request("DELETE", f"/v4/projects/{pid}/scheduledjobs/{j['id']}")
        print(f"[ensure_drift_scheduled] removed scheduled job '{SELF_JOB_TITLE}' ({j['id']}) -> {r.status_code}")


def main():
    monitor_id = sys.argv[1] if len(sys.argv) > 1 else find_monitor_for_endpoint(WORKBENCH_MODEL_ID)
    if not monitor_id:
        print("ERROR: could not resolve a DMM monitor for this endpoint (has the baseline been set in the UI?)")
        return 1

    def is_scheduled() -> bool:
        s = requests.get(f"{DOMAIN}/model-monitor/v2/api/get_model_summary",
                         params={"model_id": monitor_id}, headers=HEADERS, timeout=30).json()
        return bool(s.get("isDataDriftCheckScheduled"))

    if is_scheduled():
        print(f"[ensure_drift_scheduled] {monitor_id}: already scheduled — cleaning up this job.")
        _delete_self_scheduled_job()
        return 0

    print(f"[ensure_drift_scheduled] {monitor_id}: not yet scheduled — (re)configuring + attempting activation…")
    configure_drift(monitor_id)  # idempotent: re-saves the same PSI config every time, harmless
    schedule_drift(monitor_id)   # 400 "No prediction data registered" is EXPECTED pre-ingestion — not an error

    if is_scheduled():
        print(f"[ensure_drift_scheduled] {monitor_id}: ✅ now scheduled — cleaning up this job.")
        _delete_self_scheduled_job()
    else:
        print(f"[ensure_drift_scheduled] {monitor_id}: still pending (will retry on next scheduled run).")
    return 0  # never fail the job — "not yet possible" is a normal, expected interim state


if __name__ == "__main__":
    raise SystemExit(main())

"""Create the monitoring scheduled jobs from config.yaml:
  1. Load generator — puts sustained business-hours traffic on the endpoint (feeds capture).
  2. Ground-truth generator — matures labels so model-quality metrics populate.
  3. Ensure Drift Scheduled — frequent, SELF-DELETING poll that activates the DMM drift schedule
     the moment the platform allows it (blocked until the first daily ingestion), then removes
     itself. Default every 10 min — frequent because it's nearly free (fast no-op once scheduled)
     and short-lived (deletes itself), unlike the other two jobs which run forever.

Model-agnostic: commands/crons come from config `jobs.*`. Encodes the hard-won best practices:
  - clean `python scripts/x.py` command (NO inline VAR= — Domino tokenizes it away),
  - config supplied via project env vars,
  - mainRepoGitRef pinned so jobs run the intended committed code (not stale main).
See DOMINO_PLATFORM_NOTES.md §3, §3a.

Usage:  python scripts/create_scheduled_jobs.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import domino_api as api

DSE = os.getenv("DOMINO_ENVIRONMENT_ID", "")
HW = os.getenv("DOMINO_HARDWARE_TIER_ID", "small-k8s")


def _user_object_id() -> str:
    r = api.get("/v4/users/self")
    return r.json().get("id", "") if r.status_code == 200 else ""


def _job(title, command, cron, git_ref, tz, user_id, paused=False):
    return {
        "title": title, "command": command,
        "schedule": {"cronString": cron, "isCustom": True},
        "timezoneId": tz, "isPaused": paused,
        "allowConcurrentExecution": False, "hardwareTierIdentifier": HW,
        "overrideEnvironmentId": DSE, "environmentRevisionSpec": "ActiveRevision",
        "scheduledByUserId": user_id, "notifyOnCompleteEmailAddresses": [],
        # ⚠️ type MUST be "branches" (plural) — "branch" (singular) is silently ACCEPTED at
        # creation (200) but every actual run then fails with a 500 IllegalStateException
        # "Unknown reference type" (only visible in the job's own logs, not the API response).
        "mainRepoGitRef": {"type": "branches", "value": git_ref},
    }


def main():
    pid = config.project_id()
    tz = config.get("jobs.timezone", default="America/New_York")
    git_ref = config.get("jobs.git_ref", default="main")
    user_id = _user_object_id()
    if not user_id:
        print("ERROR: could not resolve user ObjectId (GET /v4/users/self)"); return 2

    # Namespace titles by model — a project may monitor more than one model with this blueprint.
    model_name = config.get("monitoring.monitor_model_name",
                            default=config.get("model.registered_model_name", default="model"))
    jobs = [
        _job(f"{model_name} - Load Generator",
             config.get("jobs.load_generator_command", default="python scripts/load_generator.py"),
             config.get("jobs.load_generator_cron", default="0 0/15 * * * ?"), git_ref, tz, user_id),
        _job(f"{model_name} - Ground Truth Generator",
             config.get("jobs.ground_truth_command", default="python scripts/ground_truth_generator.py"),
             config.get("jobs.ground_truth_cron", default="0 0 0/2 * * ?"), git_ref, tz, user_id),
        _job(f"{model_name} - Ensure Drift Scheduled",
             config.get("jobs.ensure_drift_scheduled_command", default="python scripts/ensure_drift_scheduled.py"),
             config.get("jobs.ensure_drift_scheduled_cron", default="0 0/10 * * * ?"), git_ref, tz, user_id),
    ]
    for j in jobs:
        r = api.post(f"/v4/projects/{pid}/scheduledjobs",
                     headers={"Content-Type": "application/json"}, json=j)
        ok = r.status_code < 300
        print(f"{j['title']:38s} -> {r.status_code} {'id='+r.json().get('id','') if ok else r.text[:150]}")
    token_env = config.get("endpoint.token_env_var", default="ENDPOINT_AUTH_TOKEN")
    print(f"\nReminder: run scripts/set_project_token.py to set '{token_env}' as a PROJECT env var "
          "(the URL comes from config.yaml, no env var needed). The load job authenticates with the token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deploy a registered MLflow model as a MONITORED Domino Model API (registry deploy).

Model-agnostic: reads names/ids from config.yaml. Registry deploy (not file-based) is the
only path that gives Domino Model Monitoring a training-data baseline. See
DOMINO_PLATFORM_NOTES.md §2, §2a, §5a.

Usage:
    python scripts/deploy_endpoint.py            # deploy latest registered version
    python scripts/deploy_endpoint.py --version 2
    python scripts/deploy_endpoint.py --new-version   # add a new version to the existing endpoint
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import domino_api as api

DSE = os.getenv("DOMINO_ENVIRONMENT_ID", "")
HW = os.getenv("DOMINO_HARDWARE_TIER_ID", "small-k8s")


def _version_body(reg_name, reg_version, pred_dataset_resource_id=None):
    body = {
        "projectId": config.project_id(),
        "source": {"type": "Registry", "registeredModelName": reg_name,
                   "registeredModelVersion": int(reg_version)},
        "environmentId": DSE,
        "logHttpRequestResponse": False,
        "monitoringEnabled": True,
        "recordInvocation": True,
        "shouldDeploy": True,
    }
    if pred_dataset_resource_id:
        body["predictionDatasetResourceId"] = pred_dataset_resource_id
    return body


def poll(model_api_id, version_id=None, timeout_min=40):
    """Poll a version's build+deploy to running (or the endpoint's activeStatus)."""
    for _ in range(timeout_min * 3):
        if version_id:
            ds = api.get(f"/v4/models/{model_api_id}/{version_id}/getModelDeploymentStatus")
            s = ds.json().get("status") if ds.status_code == 200 else "?"
            print(f"  deploy={s}", flush=True)
            if s == "running":
                return True
            if s in ("failed", "error"):
                return False
        else:
            st = api.get(f"/models/{model_api_id}/activeStatus")
            if st.status_code == 200:
                j = st.json()
                print(f"  {j.get('status')} / {j.get('lastOperation',{}).get('longStateDescription')}", flush=True)
                if j.get("status") == "Running":
                    return True
                if j.get("lastOperation", {}).get("isFailure"):
                    return False
        time.sleep(20)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1, help="registered model version to deploy")
    ap.add_argument("--new-version", action="store_true",
                    help="add a new version to the existing endpoint (config endpoint.model_id)")
    args = ap.parse_args()

    reg_name = config.get("model.registered_model_name")
    name = config.get("monitoring.monitor_model_name", default=reg_name)

    if args.new_version:
        mid = config.get("endpoint.model_id", env="MODEL_ID")
        if not mid:
            print("ERROR: endpoint.model_id not set — deploy a first version without --new-version.")
            return 2
        vers = api.get(f"/api/modelServing/v1/modelApis/{mid}/versions").json()
        pdr = (vers[0].get("predictionDatasetResourceId") if isinstance(vers, list) and vers else None)
        r = api.post(f"/api/modelServing/v1/modelApis/{mid}/versions",
                     headers={"Content-Type": "application/json"},
                     json=_version_body(reg_name, args.version, pdr))
        print("new version ->", r.status_code, r.text[:200])
        return 0 if r.status_code < 300 else 1

    req = {
        "name": name, "description": f"{name} (registry-deployed, monitored)",
        "environmentId": DSE, "isAsync": False, "strictNodeAntiAffinity": False,
        "environmentVariables": [], "hardwareTierId": HW, "replicas": 1,
        "version": _version_body(reg_name, args.version),
    }
    r = api.post("/api/modelServing/v1/modelApis", headers={"Content-Type": "application/json"}, json=req)
    print("create endpoint ->", r.status_code, r.text[:200])
    if r.status_code >= 300:
        return 1
    mid = r.json().get("id")
    print(f"endpoint id: {mid}  → set endpoint.model_id in config.yaml. Polling to Running…")
    ok = poll(mid)
    print("Running ✅" if ok else "did not reach Running ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

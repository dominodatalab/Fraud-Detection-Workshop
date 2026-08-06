"""Smoke test a deployed Model API — fully generic, built from config.yaml's `model.features`
(no model-specific field names hard-coded here).

Auth: Domino Model APIs use Basic auth with the access token as user AND password.
Falls back to DOMINO_USER_API_KEY (works only if the endpoint is public or the caller has
direct model access) if no explicit ENDPOINT_AUTH_TOKEN/token file is provided.

Usage:  python scripts/smoke_test_endpoint.py
"""
import os
import sys
import json
import uuid
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

URL = config.endpoint_url()
TOKEN = config.endpoint_token()
FEATURES = config.get("model.features", default=[])
ROW_ID_FIELD = config.get("model.row_identifier", default="row_identifier")
TS_FIELD = config.get("model.timestamp", default="application_timestamp")


def base_payload() -> dict:
    """One valid request built from each feature's `sample` value in config.yaml."""
    payload = {}
    for f in FEATURES:
        if "sample" not in f:
            raise SystemExit(f"ERROR: model.features['{f['name']}'] has no `sample` in config.yaml.")
        payload[f["name"]] = f["sample"]
    return payload


def perturbed(payload: dict, factor: float) -> dict:
    """Vary numerical fields by `factor` for a bit of request diversity (categoricals unchanged)."""
    out = dict(payload)
    numeric_names = {f["name"] for f in FEATURES if f["valueType"] == "numerical"}
    for k in numeric_names:
        v = out[k]
        out[k] = type(v)(v * factor) if isinstance(v, float) else v * factor
    return out


def call(payload: dict):
    return requests.post(
        URL, json={"data": payload},
        headers={"Content-Type": "application/json"},
        auth=(TOKEN, TOKEN) if TOKEN else None, timeout=30,
    )


def main():
    if not FEATURES:
        print("ERROR: model.features is empty in config.yaml"); return 2
    print(f"URL   : {URL}")
    print(f"auth  : {'token set (%d chars)' % len(TOKEN) if TOKEN else 'NONE'}")

    base = base_payload()
    cases = [("BASE", base)]
    for i, factor in enumerate((0.5, 1.0, 1.5), start=1):
        cases.append((f"VAR{i}", perturbed(base, factor)))

    failures = 0
    for label, payload in cases:
        payload = dict(payload)
        payload[ROW_ID_FIELD] = f"smoke-{label.lower()}-{uuid.uuid4().hex[:8]}"
        payload[TS_FIELD] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            r = call(payload)
            if r.status_code != 200:
                print(f"[{label}] HTTP {r.status_code}: {r.text[:250]}")
                failures += 1
                continue
            body = r.json()
            result = body.get("result", body)
            print(f"[{label}] {json.dumps(result)}")
        except Exception as e:
            print(f"[{label}] ERROR: {e}")
            failures += 1
    print(f"\n{len(cases)-failures}/{len(cases)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

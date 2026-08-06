"""
Thin authenticated Domino REST client for use *inside* a Domino workspace.

Routes through the local API proxy (http://localhost:8899) and fetches a fresh
short-lived bearer token from /access-token on every call -- the same mechanism
the Domino MCP server uses. Import and reuse across the build scripts.
"""
from __future__ import annotations

import os
import requests

PROXY = os.getenv("DOMINO_API_PROXY", "http://localhost:8899")

PROJECT_ID = os.getenv("DOMINO_PROJECT_ID", "")
PROJECT_NAME = os.getenv("DOMINO_PROJECT_NAME", "")
PROJECT_OWNER = os.getenv("DOMINO_PROJECT_OWNER", "")
USER_API_KEY = os.getenv("DOMINO_USER_API_KEY", "")


def auth_headers() -> dict:
    try:
        r = requests.get(f"{PROXY}/access-token", timeout=10)
        r.raise_for_status()
        tok = r.text.strip()
        if not tok.startswith("Bearer "):
            tok = f"Bearer {tok}"
        return {"Authorization": tok}
    except Exception:
        # fall back to the classic API key header
        return {"X-Domino-Api-Key": USER_API_KEY}


def request(method: str, path: str, **kw):
    url = path if path.startswith("http") else f"{PROXY}{path}"
    headers = {**auth_headers(), **kw.pop("headers", {})}
    return requests.request(method, url, headers=headers, timeout=kw.pop("timeout", 60), **kw)


def get(path, **kw):
    return request("GET", path, **kw)


def post(path, **kw):
    return request("POST", path, **kw)


if __name__ == "__main__":
    print("PROJECT_ID  :", PROJECT_ID)
    print("PROJECT     :", f"{PROJECT_OWNER}/{PROJECT_NAME}")
    r = get("/access-token")
    print("token status:", r.status_code, "(len", len(r.text.strip()), ")")

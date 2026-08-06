"""One-shot: write the model access token to this project's Domino environment variables,
under the name configured by `endpoint.token_env_var` in config.yaml (default
ENDPOINT_AUTH_TOKEN — override this if the project monitors more than one model, since
scheduled jobs share PROJECT-wide env vars with no per-job override; see config.template.yaml).

Run it yourself from the workspace terminal (or via the `!` prefix in Claude Code):
    python scripts/set_project_token.py [token_filename]

Reads the token from the gitignored token file (config `endpoint.token_file`, or an override
passed as argv[1]) and POSTs it to /v4/projects/{projectId}/environmentVariables. The load
generator scheduled job picks it up on its next run. (This is separated out so the credential
write is an explicit user action, not the agent's.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import domino_api as api

PID = os.getenv("DOMINO_PROJECT_ID")
TOKEN_ENV_NAME = config.get("endpoint.token_env_var", default="ENDPOINT_AUTH_TOKEN")

_fname = sys.argv[1] if len(sys.argv) > 1 else config.get("endpoint.token_file", default=".endpoint_token")
# Relative filenames resolve next to config.yaml itself (it may live in a model subdirectory).
token_path = _fname if os.path.isabs(_fname) else os.path.join(config._CONFIG_DIR, _fname)

if not os.path.exists(token_path):
    print(f"ERROR: {token_path} not found. Put the model access token there first.")
    raise SystemExit(1)

token = open(token_path).read().strip()
r = api.post(
    f"/v4/projects/{PID}/environmentVariables",
    headers={"Content-Type": "application/json"},
    json={"name": TOKEN_ENV_NAME, "value": token},
)
print(f"POST {TOKEN_ENV_NAME} -> {r.status_code}")
if r.status_code == 200:
    print(f"Done. {TOKEN_ENV_NAME} is now a project environment variable.")
    print("Next: (re)enable the load-generator scheduled job so it can authenticate.")
else:
    print(r.text[:300])
    raise SystemExit(1)

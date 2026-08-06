"""Central configuration loader for all model-monitoring blueprint scripts.

Single source of truth is ``config.yaml`` at the repo root. Precedence for any value:
    environment variable  >  config.yaml  >  built-in default
so scheduled jobs / CI can override anything with an env var, and there are NO
instance-specific values hard-coded in the Python.

Instance runtime basics (project id, datasets dir, domain) come from the DOMINO_*
env vars that Domino injects, exposed here as helper functions.

Usage:
    from config import cfg
    cfg.get("monitoring.psi_threshold", cast=float)         # 0.2
    cfg.get("endpoint.model_id", env="MODEL_ID")            # env wins, else config, else None
    cfg.endpoint_url(); cfg.endpoint_token(); cfg.scoring_log_dir()
"""
from __future__ import annotations

import functools
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CONFIG_PATH = os.getenv("MONITORING_CONFIG", os.path.join(_ROOT, "config.yaml"))
# Relative paths INSIDE config.yaml (token_file, etc.) resolve relative to config.yaml's own
# directory, not this blueprint's root — config.yaml may live anywhere (e.g. a model's own
# subdirectory via MONITORING_CONFIG), so paths must travel with it.
_CONFIG_DIR = os.path.dirname(os.path.abspath(_CONFIG_PATH))


@functools.lru_cache(maxsize=1)
def _data() -> dict:
    try:
        with open(_CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def get(dotted: str, env: str | None = None, default=None, cast=None):
    """Resolve a config value: env var (if set) > config.yaml (dotted path) > default."""
    val = None
    if env and os.getenv(env) not in (None, ""):
        val = os.getenv(env)
    else:
        node = _data()
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        val = node
    if val in (None, ""):
        val = default
    if cast is not None and val is not None:
        val = cast(val)
    return val


# ── instance runtime (from Domino-injected env) ──────────────────────────────
def domain() -> str:
    return os.getenv("DOMINO_DOMAIN", "").rstrip("/")


def project_id() -> str:
    return os.getenv("DOMINO_PROJECT_ID", "")


def project_name() -> str:
    return os.getenv("DOMINO_PROJECT_NAME", "")


def datasets_dir() -> str:
    return os.getenv("DOMINO_DATASETS_DIR", "/mnt/data")


# ── derived convenience ──────────────────────────────────────────────────────
# ⚠️ Domino scheduled jobs have NO per-job environment variables — only PROJECT-wide ones
# (confirmed: the scheduledjobs API has no env-var field). If a project monitors more than
# one model, generic names like ENDPOINT_URL/ENDPOINT_AUTH_TOKEN will COLLIDE across models'
# scheduled jobs (whichever was set last wins for everyone). Set `endpoint.token_env_var` (and
# `endpoint.url_env_var` if you really need a URL override) to a name namespaced per model,
# e.g. `MY_MODEL_ENDPOINT_AUTH_TOKEN`, whenever more than one monitored model shares a project.

def endpoint_url() -> str:
    """Explicit config `endpoint.url` (or its configurable env override), else derived from
    domain + model_id. URL is not secret, so config.yaml alone is normally sufficient —
    prefer that over an env var when multiple models share a project (see note above)."""
    url_env = get("endpoint.url_env_var", default=None)
    url = get("endpoint.url", env=url_env) if url_env else get("endpoint.url")
    if url:
        return url
    mid = get("endpoint.model_id", env="MODEL_ID")
    return f"{domain()}/models/{mid}/latest/model" if (domain() and mid) else ""


def endpoint_token() -> str:
    """Model access token: the env var named by `endpoint.token_env_var` (default
    ENDPOINT_AUTH_TOKEN), else the gitignored token_file, else the Domino user API key
    (works only if the caller already has direct model access)."""
    token_env = get("endpoint.token_env_var", default="ENDPOINT_AUTH_TOKEN")
    tok = os.getenv(token_env)
    if tok:
        return tok
    token_file = get("endpoint.token_file", default=".endpoint_token")
    path = token_file if os.path.isabs(token_file) else os.path.join(_CONFIG_DIR, token_file)
    if os.path.exists(path):
        return open(path).read().strip()
    return os.getenv("DOMINO_USER_API_KEY", "")


def scoring_log_dir() -> str:
    """Full SCORING_LOG_DIR env override, else <datasets>/<project>/<subdir>."""
    override = os.getenv("SCORING_LOG_DIR")
    if override:
        return override
    subdir = get("traffic.scoring_log_subdir", default="monitoring")
    return os.path.join(datasets_dir(), project_name(), subdir)

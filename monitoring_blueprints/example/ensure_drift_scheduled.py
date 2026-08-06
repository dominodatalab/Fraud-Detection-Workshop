"""Self-locating wrapper for scripts/ensure_drift_scheduled.py — see that file for what this does
and why it's meant to run as a daily scheduled job (self-healing drift-schedule activation)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MONITORING_CONFIG", os.path.join(HERE, "config.yaml"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from ensure_drift_scheduled import main

if __name__ == "__main__":
    raise SystemExit(main())

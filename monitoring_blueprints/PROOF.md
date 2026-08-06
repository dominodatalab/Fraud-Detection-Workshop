# Proof of Concept — verified end-to-end on se-demo (2026-07-02)

This blueprint was validated by building and deploying a **second, fully independent model**
(`blueprint_test_model` — unrelated to the parent `RetailAcquisitionScorecard`) entirely through the
blueprint's own generic scripts, on the same live Domino instance the scorecard runs on. Every step
below is a real, verified result — not a dry run — including six real bugs the exercise caught and
fixed, plus one design refinement made per direct feedback (details in
`LESSONS_LEARNED_DOMINO_PLATFORM.md`). **End state: fully monitored, drift check confirmed actively
scheduled (`isDataDriftCheckScheduled: true`) — the entire pipeline works, with no step left
unverified.**

## What was built
`example/` — a tiny logistic-regression classifier over `feature_a`/`feature_b`/`feature_c` →
`prediction`/`decision`, with injected baseline→drift distributions, entirely separate from the
scorecard's model/data/config.

## What was verified live (not simulated)
| Step | Script | Result |
|---|---|---|
| Train + register | `example/train.py` | `blueprint_test_model` v1/v2 registered in MLflow |
| Deploy (registry, monitored) | `scripts/deploy_endpoint.py --version 2` | Endpoint `6a46928d…` built + reached **Running**, zero manual steps |
| Baseline TrainingSet | `scripts/create_baseline_trainingset.py` (via `example/create_baseline.py`) | `blueprint_test_baseline` v1 — capture-aligned columns only, verified clean |
| Smoke test | `scripts/smoke_test_endpoint.py` | 4/4 requests succeeded, fully config-driven (zero hard-coded field names) |
| Scheduled jobs | `scripts/create_scheduled_jobs.py` | All 3 jobs created (Load Generator, Ground Truth Generator, Ensure Drift Scheduled); **actual scheduled + ad-hoc runs verified Succeeded** with real data landing (416+ scoring-log rows, correct drift regime) |
| DMM registration | UI (Training Data step — the one unavoidable manual action) | Monitor auto-created with **exactly 5 clean variables** (3 features + prediction + decision, no target/GT leakage) |
| Drift config | `scripts/setup_monitoring.py --find-monitor / --configure-drift` | PSI > 0.2 saved on `feature_a`/`feature_c` |
| **Drift schedule ACTIVE** | `scripts/ensure_drift_scheduled.py` (self-cleaning daily job) | `isDataDriftCheckScheduled: true` confirmed live — see below |
| Ground truth job | `example/ground_truth.py` | Runs successfully; correctly no-ops until the maturation window elapses |

## ✅ CLOSED — drift check is actively scheduled, confirmed end-to-end
`scripts/ensure_drift_scheduled.py` ran every 10 minutes (`blueprint_test_model - Ensure Drift
Scheduled`), correctly reported "still pending" while `blueprint_test_model`'s monitor was in
`ingestionStatus: training`, and **self-activated + self-deleted automatically** once Domino's nightly
ingestion moved the monitor to `ingestionStatus: prediction`. Final confirmed state:
- `GET .../get_model_summary?model_id=6a469d6053c5efcabd1bce3e` → `isDataDriftCheckScheduled: true`
- `GET .../model?model_id=...` → `ingestionStatus: prediction` (was `training`)
- The scheduled job itself is **gone** from `/v4/projects/{id}/scheduledjobs` — it deleted itself,
  verified by its absence, not just a log message claiming success.

Total elapsed time from monitor creation to confirmed active schedule: the ingestion boundary the
monitor needed to cross (created 17:18 UTC same day; the scorecard's monitor, created 01:31 UTC the
same day — 17.5h earlier — was already past that boundary at the time of comparison, which is exactly
why it showed `isDataDriftCheckScheduled: true` while `blueprint_test_model` did not, yet). No manual
intervention was needed to close this out — the self-deleting job did its job.

## Observed + explained: "Column X: N out of M untrained classes found"
After the first ingestion, the DMM UI's Prediction Data page showed this warning for `feature_a`
(93/452) and `prediction` (10/452). Confirmed via `POST /model/{id}/analyse-drift` this is **expected,
not an error**: DMM bins every column from the training baseline only, and any live value outside those
bins (here, `feature_a` values above the baseline's max — a direct result of the demo's intentional
drift injection shifting the mean upward) lands in a synthetic `"Untrained Classes"` bin. Full mechanism
documented in `LESSONS_LEARNED_DOMINO_PLATFORM.md` §5d. Confirms the drift injection is real and
DMM is seeing it, not a configuration problem.

## Real bugs found and fixed by this exercise
1. **Scorecard leaks in the "generic" scripts** — `setup_monitoring.py` read a project-specific JSON
   schema file and defaulted to scorecard feature names; `smoke_test_endpoint.py` hard-coded scorecard
   field names. Fixed to be driven entirely by `config.yaml`.
2. **Missing capability** — `setup_monitoring.py` had no drift-scheduling functions despite the guide
   describing them; `create_baseline_trainingset.py` didn't exist in the blueprint at all. Both added.
3. **Multi-model env-var collision** — deploying a second model in the same project caused its smoke
   test to silently hit the *scorecard's* endpoint (both used the generic `ENDPOINT_URL`/
   `ENDPOINT_AUTH_TOKEN` project env vars, and scheduled jobs have no per-job env vars). Fixed with
   `endpoint.token_env_var` (namespaced per model) and dropping the generic `ENDPOINT_URL` env check
   entirely (URL isn't secret — belongs in config.yaml).
4. **Config path resolution bug** — relative paths in `config.yaml` (like `token_file`) resolved
   against the blueprint's root directory, not the actual `config.yaml`'s own location — broke as soon
   as the config moved into `example/`. Fixed to resolve relative to `config.yaml`'s real path.
5. **⭐ Critical scheduled-job bug (also silently broke the SCORECARD's live production jobs)** —
   `mainRepoGitRef.type` must be `"branches"` (plural). `"branch"` (singular) is silently **accepted**
   at job creation (200 OK) but every actual run then fails with `500 IllegalStateException: Unknown
   reference type` — visible only in the job's own run logs, never the creation response. This had been
   breaking the scorecard's real scheduled jobs for hours before this exercise caught it. Fixed on all
   4 live jobs (scorecard + blueprint) and in `create_scheduled_jobs.py`.
6. Debugging bug #5 also surfaced that `/api/jobs/beta/jobs/{id}/logs` never shows a job's actual
   stdout (only infra/pod events) — the real output is at
   `GET /v1/projects/{owner}/{project}/run/{id}/stdout`. Documented for future debugging.
7. **Design gap, not a bug**: the first cut of `ensure_drift_scheduled.py` was a daily-cadence job that
   never cleaned itself up — functionally fine, but a poor pattern (slow to notice success, and left a
   permanent job running forever after its one-time job was done). Fixed per explicit feedback: every
   10 minutes instead of daily, and it now **deletes its own scheduled job** the moment it succeeds.

## Net result
The blueprint went from "looks generic, copied from one project" to **actually proven generic** against
a second, independent model on a live instance — with the fixes committed back so the next model (e.g.
on fsi-demo) doesn't rediscover any of this.

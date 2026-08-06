# Setup Guide — Domino Model Monitoring for a New Model

Ordered, **model-agnostic** steps to stand up any model as a **monitored** Domino Model API on a fresh
instance, with the load + ground-truth best practices. The *why* behind every step is in
[`DOMINO_PLATFORM_NOTES.md`](DOMINO_PLATFORM_NOTES.md) (§ refs). One-shot prompt: [`PORTING_PROMPT.md`](PORTING_PROMPT.md).

## What you provide (model-specific)
- A trained model logged to the **MLflow registry** as an `mlflow.pyfunc` **with prediction capture baked
  in** and an explicit signature — use [`templates/model_pyfunc_with_capture.py`](templates/model_pyfunc_with_capture.py). (§2, §6)
- A **capture-aligned baseline TrainingSet**: it must contain **only the columns the endpoint captures**
  (features + prediction outputs). No target/ground-truth/uncaptured columns, or ingestion throws
  `UNRESOLVED_COLUMN`. (§5a)
- Optional, for load + quality: fill the two hooks in [`templates/load_generator.py`](templates/load_generator.py)
  and [`templates/ground_truth.py`](templates/ground_truth.py).

## What the blueprint gives you (reusable, config-driven)
`scripts/config.py` (loader), `domino_api.py` (auth), `deploy_endpoint.py`, `setup_monitoring.py`,
`create_scheduled_jobs.py`, `smoke_test_endpoint.py`, `set_project_token.py`, and `config.template.yaml`.

---

## Steps

**0. Config.** Copy `config.template.yaml` → **`config.yaml` in the same directory**
(`monitoring_blueprints/config.yaml`, right next to the template) — that's where `scripts/config.py`
looks by default (it resolves relative to itself: root = `monitoring_blueprints/`, **not** your
project's outer repo root — verified by testing). Fill `model.*` (names + feature/prediction schema),
`monitoring.drift_features`, `jobs.*`. Instance basics come from the `DOMINO_*` env. Precedence: env
var > config.yaml > default. (§7)
> 💡 **Monitoring more than one model, or want config.yaml elsewhere** (e.g. a model-specific
> subdirectory like `example/config.yaml`)? Set `MONITORING_CONFIG=<full path>` before running any
> script, or have a thin per-model wrapper script self-locate it
> (`os.environ.setdefault("MONITORING_CONFIG", ...)`) — see `example/load_generator.py` for the pattern.
> ⚠️ **If this Domino project already monitors another model**, set `endpoint.token_env_var` to a
> name namespaced for this model (e.g. `MYMODEL_ENDPOINT_AUTH_TOKEN`). Scheduled jobs have **no
> per-job env vars** — only project-wide ones — so two models sharing the generic
> `ENDPOINT_AUTH_TOKEN` name will silently clobber each other's authentication. (§3b)

**1. Register the model.** Train + `mlflow.pyfunc.log_model` with capture (template above) and a
signature. Confirm it's in the registry. (§2)

**2. Baseline TrainingSet.** Create it with **only captured columns** (features + prediction outputs),
target in `categorical_columns` if you keep one — but do NOT add uncaptured columns. (§4, §5a)

**3. Deploy — registry, monitored.** `python scripts/deploy_endpoint.py` → `POST /api/modelServing/v1/modelApis`
(source `Registry`, `monitoringEnabled:true`), polls to Running. Record the printed `model_id` in
`config.yaml`. **File-based deploy can't be monitored — always registry.** (§2, §2a, §5a)
> ⚠️ **Prerequisite, not automated:** `deploy_endpoint.py` deploys using `$DOMINO_ENVIRONMENT_ID` — the
> environment of **whatever workspace you're currently running in**. That environment must have
> `uwsgi` (any Domino Standard Environment does). Run this from a DSE workspace, or set
> `DOMINO_ENVIRONMENT_ID` to one explicitly, or the build will fail. (§2)

**4. Token + smoke test.** Grab the model access token from the UI (Overview / "Call your model");
`echo '<token>' > .endpoint_token`; `python scripts/smoke_test_endpoint.py`. Registry/pyfunc endpoints
enforce the signature — send bool flags as ints, floats as floats; response is a list-of-rows. (§2c)

**5. Set the baseline on the endpoint version (UI).** Endpoint → Monitoring → **Training Data** → select
your baseline TrainingSet. This stages it (UI-only) and creates the monitor with variables derived from
the baseline schema — hence step 2 must be clean. The baseline is **immutable per version**; to change it
later, publish a new version and re-set. (§5a)

**6. Scheduled jobs.** Set `ENDPOINT_URL` + `ENDPOINT_AUTH_TOKEN` as **project env vars**
(`python scripts/set_project_token.py`), then `python scripts/create_scheduled_jobs.py` — creates
**three** jobs from config.yaml's `jobs.*`:
  - **Load Generator** (every 15 min) — feeds live traffic/capture. Runs forever.
  - **Ground Truth Generator** (every 2 h) — matures labels for quality metrics. Runs forever.
  - **Ensure Drift Scheduled** (every 10 min, `scripts/ensure_drift_scheduled.py`) — see step 7.

Jobs run committed code from `jobs.git_ref` — **merge your branch first.** (§3, §3a)

**7. The drift check must end up ACTIVELY SCHEDULED — this is not optional.** Saving a check's
*configuration* is not the same as it *running*: a monitor with a saved PSI config but no active
schedule never actually analyzes anything. Do not consider setup complete until you see
`isDataDriftCheckScheduled: true` on `GET .../get_model_summary?model_id=<monitor_id>`.

The problem: activating the schedule requires the monitor's **first daily ingestion**
(no on-demand trigger — §6), so it's **guaranteed to fail on day one** with
`400 "No prediction data registered with the model"`. Rather than making this a manual "remember to
retry tomorrow" TODO, the **Ensure Drift Scheduled** job created in step 6 handles it automatically:
`scripts/ensure_drift_scheduled.py` runs every 10 min, no-ops/retries harmlessly while activation isn't
possible yet, and **the moment it succeeds, deletes its own scheduled job** — so there's no lingering
poller once the work is done and nothing for a human to remember. (§5b, §6)

💡 **The self-deleting-scheduled-job pattern is reusable for any "wait until platform condition X,
then stop polling" need** — a job can `DELETE /v4/projects/{id}/scheduledjobs/{its-own-id}` (looked up
by its own known title) once its goal is met.

**8. Model quality (optional).** Needs ground truth registered — `register-dataset/ground_truth` requires
an **S3 datasource**. If your GT is only in a Domino Dataset with no writable S3, quality can't be wired
headlessly; use a dataset-based external monitor. Drift works without it. (§5c)

---

## Best practices baked in
- **Registry deploy + monitoring** from the start (never file-based for a monitored model).
- **Capture-aligned baseline** — the #1 cause of `UNRESOLVED_COLUMN`.
- **Sustained load** via a scheduled job (business-hours curve) so capture + ingestion have data.
- **Ground-truth job** on a lag so model-quality populates.
- **Config centralized** in `config.yaml`; **no instance values hard-coded** in scripts.
- **Jobs pinned to a git ref** (they run committed code, not your workspace edits).
- Expect **daily ingestion** — no same-day live output; backdated dataset-based monitor for instant demos.

## Fixing a bad baseline later
Baseline is immutable per version: fix the TrainingSet → **new endpoint version** (`deploy_endpoint.py
--new-version`) → set the corrected TrainingSet as its Training Data → **delete the old monitor**. (§5a, §2b)

# Operating Domino Headlessly — Field Notes (Domino-Skills candidate)

**General, project-agnostic** knowledge for deploying and monitoring *any* model on Domino via API.
Written to feed **Domino Skills** — nothing here is specific to a particular model/dataset. For the
retail-scorecard specifics that motivated these notes, see `LESSONS_LEARNED.md`.

Legend: ✅ works · ❌ tried, doesn't work · ⚠️ gotcha · 💡 technique.

---

## 1. API discovery (how to find "UI-only" endpoints)
- **Curated public API:** `GET {DOMAIN}/assets/public-api.json`.
- **Full internal API:** `GET {DOMAIN}/assets/swagger.json` — server base `/v4`, "Domino Data Lab API v4".
  Many features absent from the public spec live here (e.g. scheduled jobs).
- **A service's own UI client** is the authoritative payload source when docs are thin: fetch the SPA's
  JS bundle (e.g. `{DOMAIN}/model-monitor/static/js/main.<hash>.js`) and grep for endpoint paths and the
  request-builder functions.
- 💡 A `400` with a specific validation message is the server teaching you the schema — iterate on the
  body, don't abandon the endpoint.
- ⚠️ Don't guess-storm undocumented routes to fish for tokens/credentials — safety tooling will (rightly)
  block it. Use the spec/JS-bundle route instead.
- Auth from inside a workspace: short-lived bearer from `http://localhost:8899/access-token` for nucleus
  (`/v4`, `/v1`, `/api/...`) calls.

## 2. Model API endpoints
- **Two deploy styles:** *file-based* (`Domino.model_publish(file, function, environment_id, name, desc)`
  in python-domino) and *registry* (deploy a registered MLflow model). Poll
  `GET /models/{modelApiId}/activeStatus` until `Running` (`Building → Ready to run → Starting → Running`).
- ⚠️ **Model APIs need `uwsgi`** in the environment — use the project **default env** or a DSE (they have it).
- ⚠️ **Endpoints are private by default.** Calls use the **model access token** as Basic auth *username
  AND password* (`token:token`). A user API key returns **401**. The token value isn't retrievable via API
  (only metadata) — copy it from the model's "Call your model"/Overview UI tab.
- ⚠️ **Build failure at image pull with a registry-credential error is infrastructure** (rotated pull
  secret), not your code — escalate; don't rabbit-hole on `uwsgi`/repocloner.
- ✅ **Stop/start a version:** `POST /v4/models/{modelId}/{modelVersionId}/{stop|start}ModelDeployment`;
  status via `.../getModelDeploymentStatus` / `.../getBuildStatus`.
- ⚠️ python-domino `endpoint_state`/`endpoint_publish` are for git-based **Endpoints**, a different feature
  from classic **Model APIs**.

### 2a. Registry deploy via the model-serving API (recommended for monitored models)
`POST /api/modelServing/v1/modelApis` — required: `name, description, environmentId, isAsync,
strictNodeAntiAffinity, environmentVariables, version`. `version` requires `projectId, source,
logHttpRequestResponse, monitoringEnabled`; `source.type ∈ {File, Registry}` (Registry: +
`registeredModelName`, `registeredModelVersion`). ⚠️ A **monitored** version also needs
`predictionDatasetResourceId` (reuse the endpoint's existing one from
`GET /api/modelServing/v1/modelApis/{id}/versions[0]`) or it 500s.

### 2b. New VERSION vs new ENDPOINT (Domino docs guidance)
Default to a **new version of the existing Model API** (same URL/token/consumers/history) when you
retrain, change code/config, or **switch source File↔Registry**:
`POST /api/modelServing/v1/modelApis/{modelApiId}/versions`. Create a **new endpoint** only for a
fundamentally different API. A file-first endpoint is still versionable in the serving API
(`GET /api/modelServing/v1/modelApis/{id}` → 200).
⚠️ Switching a version File→Registry changes the **invocation contract** — re-verify callers.

### 2c. pyfunc (registry) invocation contract
mlflow pyfunc endpoints enforce the logged **signature**: booleans must be sent as **ints** (`1`/`0`),
and numeric **types** must match (int where a float is expected → `"Failed to enforce schema"`). Response
is a **list-of-rows** (`[[...]]`), not a dict. Envelope is `{"data": {...}}`; auth is Basic `token:token`.

## 3. Scheduled jobs (headless)
❌ Not in the public API; python-domino has no scheduled-job method (only ad-hoc `runs_start`).
✅ Internal: `POST /v4/projects/{projectId}/scheduledjobs`. Required: `title, command,
schedule{cronString,isCustom}, timezoneId, isPaused, allowConcurrentExecution, hardwareTierIdentifier,
environmentRevisionSpec ("ActiveRevision"|"LatestRevision"|{revisionId}), scheduledByUserId,
notifyOnCompleteEmailAddresses[]`. `GET|PUT|DELETE .../scheduledjobs/{key}` to list/update/pause/delete.
- ⚠️ `scheduledByUserId` = the Mongo **ObjectId** (`GET /v4/users/self` → `.id`), NOT numeric `DOMINO_USER_ID`.
- ⚠️⚠️ **Domino tokenizes the command** and re-quotes each token, so inline `VAR=value python x.py` is
  silently ignored. Use a clean `python x.py` and supply config via **project env vars**
  (`POST /v4/projects/{id}/environmentVariables {name,value}`).
- Cron is Quartz-style with seconds: every 15 min `0 0/15 * * * ?`, every 2 h `0 0 0/2 * * ?`.
- 💡 **To debug a failed job, get its actual stdout, not `/api/jobs/beta/jobs/{id}/logs`.** That
  endpoint only returns infrastructure/pod-scheduling events (image pulls, "Successfully assigned…") —
  never your script's `print()` output or tracebacks, and it reports `isComplete: true` even though it's
  silent about the real failure. The real command output is at
  `GET /v1/projects/{ownerUsername}/{projectName}/run/{jobId}/stdout` → `{"stdout": "..."}` (also what
  python-domino's `runs_stdout` and the Domino MCP server's `check_domino_job_run_results` use). Look
  for your own script's markers or a Python traceback in there, not in the beta logs endpoint.

### 3a. ⚠️⚠️ Jobs run COMMITTED code at a git ref — not your workspace edits
A job with `mainRepoGitRef = null` runs the **project default branch (main)**. Fixes on a feature branch
won't be seen until merged or the ref is set. **Merge to the default branch before relying on scheduled
jobs.**

⚠️⚠️⚠️ **The scheduled-job `type` value is `"branches"` (plural) — `"branch"` (singular) is silently
ACCEPTED at creation (`PUT`/`POST` returns 200) but every actual scheduled run then fails**:
`500 IllegalStateException: Unknown reference type` (`ReferenceDTO$.get`), visible only in the job's
own run logs, never in the API response. **This bit us for real**: both a production job and a brand
new one ran silently broken for hours because the creation call "succeeded." 💡 **Creating a scheduled
job successfully (200) is NOT proof it runs — always verify at least one actual execution (check its
run logs / `executionStatus`), not just the creation response.** Verified-correct schema:
`{type:"branches", value:"<branch-name>"}`. The **ad-hoc** jobs API (`/api/jobs/v1/jobs`) uses a
different field name but the same value set: `{refType:"branches", value:"<b>"}`
(`refType ∈ head|commitId|tags|branches`).

### 3b. ⚠️⚠️ Scheduled jobs have NO per-job environment variables — only project-wide ones
Confirmed against the scheduledjobs schema: there is no env-var field on the job object, only
`overrideEnvironmentId`/`environmentRevisionSpec` (which env *image* revision to use, not variables).
Config/secrets can only reach a job via **project-level** env vars
(`POST /v4/projects/{id}/environmentVariables`), which are shared by **every** job and workspace in
the project. ⇒ **If a project monitors more than one model**, generic names (`ENDPOINT_URL`,
`ENDPOINT_AUTH_TOKEN`) **collide**: whichever model set them last silently wins for every other
model's scheduled job too (discovered live — model A's job authenticated against model B's endpoint
because both used the same project env var names). 💡 Fix: namespace the env var name per model (e.g.
`MYMODEL_ENDPOINT_AUTH_TOKEN`), and prefer putting non-secret values (like the endpoint URL) directly
in each model's own config file rather than a shared env var — only the true secret needs an env var
at all.

### 3c. 💡 Pattern: a SELF-DELETING scheduled job for "poll until platform condition X, then stop"
Domino has no built-in "run once when condition X becomes true" primitive, and several real
conditions here are ones the API can't force to happen sooner (e.g. §5b's ingestion-gated
schedule activation). Rather than either (a) a one-shot manual retry that requires a human to
remember to check back, or (b) a job that polls forever after its job is done, use a **frequent,
self-deleting** scheduled job:
1. Create a job (e.g. every 10 min) that checks the target condition and performs the action
   idempotently — safe to run repeatedly, harmless to retry, never fails just because the
   condition isn't met yet (return 0 either way).
2. The moment the condition is met, have the job **delete its own scheduled-job definition**:
   look itself up by its known, deterministic title —
   `GET /v4/projects/{projectId}/scheduledjobs`, find the entry whose `title` matches, then
   `DELETE /v4/projects/{projectId}/scheduledjobs/{scheduledJobKey}`.
3. Verify closure by the job's **absence** from the scheduled-jobs list, not just a log line
   claiming success (consistent with §3a: creation/log success ≠ actual state).
This closes the loop with zero manual follow-up and zero lingering compute once the one-time work
is done — reusable for any "wait for an async platform process, then stop checking" need (ingestion
delays, build completion, external approval, etc.), not just drift-schedule activation.

## 4. TrainingSets
Register with `domino_data.training_sets`. Version IDs are in the container path
`/trainingset/{projectId}/{featureSetId}/{featureSetVersionId}`.

## 5. Domino Model Monitor (DMM) API
⚠️ **Separate service** — not on the workspace proxy, absent from public/`/v4` specs. Base
`{DOMAIN}/model-monitor/v2/api`, auth header `X-Domino-Api-Key: <DOMINO_USER_API_KEY>`. (Grep the DMM SPA
JS bundle for exact payloads.)
- `GET /models?pageNumber=&pageSize=&numberOfLastChecksToFetch=` — list (all 3 params required).
- `GET /model?model_id=<id>` — get one (⚠️ **snake_case** `model_id`; `modelId` → 500).
- `PUT /model` — register: `{modelMetadata{name,modelType,version,sourceType,sourceDetails{workbenchModelId,
  workbenchModelVersionId}}, variables:[{name,valueType,variableType}], run_cohort_analysis,
  datasetDetails{name,datasetType:"file",datasetConfig{path,fileFormat},datasourceName,datasourceType}}`.
- `PUT /model/{id}/register-dataset/{prediction|ground_truth}`; `POST /model/{id}/analyse-drift`;
  `GET /model/{id}/variables/summary`; `GET /get_model_summary?model_id=`; `DELETE /model?model_id=`.
- `GET /datasources`, `PUT /datasource` — ⚠️ DMM datasources are **cloud object stores only**
  (`s3`, `generic_s3`, `azure_blob`, `azure_data_lake`).

### 5a. ⭐ The baseline is bound at endpoint-version publish time, and is immutable
For a **workbench** (Domino-endpoint) monitor, DMM aligns the monitored variable set to the baseline's
columns and analyzes **every** variable against the prediction-capture data. Consequences:
- ⚠️⚠️ The **baseline must contain only columns present in prediction capture** (features + prediction
  outputs). Any extra column (target, ground-truth label, uncaptured field) → on ingestion:
  `[UNRESOLVED_COLUMN.WITH_SUGGESTION] <col> cannot be resolved`.
- ⚠️⚠️ Register a workbench model with **features only** — DMM derives prediction outputs from the
  endpoint's capture schema; declaring a prediction variable → `"Variable <p> not allowed for
  registration"`. The target/ground-truth are wired separately.
- ⚠️⚠️ **Training data (and prediction data) are UI-controlled and immutable per version** — greyed out
  with *"publish a new model version of this API to change these settings."* To FIX/CHANGE the baseline:
  (1) fix the TrainingSet → new version; (2) **publish a new endpoint version**; (3) set the corrected
  TrainingSet as that version's Training Data in the UI → fresh monitor with clean variables;
  (4) **delete the old monitor** (`DELETE /model?model_id=`) — it lingers and keeps erroring.
- ❌ **No clean API to stage a Domino TrainingSet/Dataset into DMM storage** — that happens in the UI when
  you pick a TrainingSet. `datasetDetails` always needs a `datasourceName/datasetConfig` (S3), even with
  `featureSetId`. File-based endpoints have no training-data lineage at all → baseline greyed out → **use
  registry deploy for anything you intend to monitor.**
- ❌ `PATCH /model` is metadata-only — **no API to edit a registered monitor's variables.**
- ⚠️ **Can't delete a monitor while its endpoint version is Running** → 409. Stop the endpoint, delete,
  restart (§2).

### 5b. Scheduling checks (two steps)
1. `POST /model/{id}/save-scheduler-query {config:[{id,name,type,value,algorithm,
   condition{operator,lower,upper},alertable}]}` — saves *what* to check (returns `true`, does NOT schedule).
   The `features` list re-reflects the monitor's variables, so prune any you don't want checked.
2. `POST /model-monitor/api/scheduler/create_drift_schedule_check` (note `/api/`, not `/v2/api/`):
   `{modelId, name, timezone, cronExpression, calendarType, dataSinceLastCheck, data_since}` — activates it.
   - ⚠️ `data_since` is a **look-back COUNT in calendar periods** (UI default `1`), NOT an epoch.
   - ⚠️ `calendarType ∈ {D,M,W}` — **Day is the finest cadence; no hourly.**
   - ⚠️ Requires **prediction data already ingested** on the monitor (400 `"No prediction data
     registered"` on a fresh monitor/version — wait for the first ingestion). 💡 Check readiness via
     `GET /model?model_id=<id>` → `ingestionStatus`: `"training"` (baseline only, no predictions
     ingested yet) → `"prediction"` (has ingested live predictions — scheduling now possible). This
     field flips automatically the moment the nightly batch completes; poll it instead of guessing.
   - Edit: `POST .../scheduler/update_scheduler_config` with `scheduler_job_id`. Remove:
     `POST .../scheduler/{schedulerJobId}/unschedule_scheduler_job` (id in path, body `{}`).
   - Model **quality** scheduling → 400 until ground truth is registered.

### 5c. Ground truth for a workbench monitor
Only path is `register-dataset/ground_truth` (needs an **S3 datasource**). There is **no label-ingestion
API** (`traffic/ground-truth` is read-only; the capture client only does `capturePrediction`). If GT lives
in a Domino Dataset and there's no writable S3, quality can't be wired headlessly — use an S3 datasource or
a dataset-based **external** monitor.

### 5d. 💡 "N out of M untrained classes found" — expected on real drift, not a bug
DMM builds per-column **bins from the training/baseline distribution only**: categorical columns get one
bin per distinct baseline category; numerical columns get ~20 range bins spanning the baseline's observed
min–max. Every column (this applies to **numerical columns too**, e.g. a feature or even a numeric
`prediction` output — not just categoricals) gets a synthetic **`"Untrained Classes"`** catch-all bin for
any live/prediction value that falls **outside** every trained bin (an unseen category, or a number below
the baseline min / above the baseline max). The UI's "N out of M untrained classes found" warning on the
Prediction Data page is exactly this: N live rows landed in that catch-all across M total rows analyzed.
- **This is a companion signal to drift, not a defect.** If you intentionally shift a feature's
  distribution upward/downward (or introduce new categories) to demo drift, seeing a rising
  untrained-classes count on exactly that column is expected and confirms the shift is real —
  values are landing somewhere the baseline never saw. Verified directly: `analyse-drift`'s response
  includes per-column `bins`/`counts` arrays ending in `["Untrained Classes","Invalid"]`; the raw counts
  are inspectable via `POST /model/{id}/analyse-drift` (needs `pagination{sortOrder:<int>,
  sortOn:<one of column_name|divergence|model_column_id|feature_importance|created_on|created_at|
  updated_at>}` — the schema validation errors along the way name the missing/wrong fields, iterate on them).
- For a column you did **not** expect to drift, a nonzero/rising count is a genuine early-warning signal
  worth investigating (upstream data quality change, new customer segment, schema drift), complementary
  to the PSI/KL divergence metric.

## 6. Prediction capture & ingestion
- **Capture is explicit — a registry-deployed model does NOT auto-capture.** Wire
  `domino_data_capture.DataCaptureClient(feature_names, predict_names)` into the pyfunc `predict()` and
  call `capturePrediction(feature_vals, predict_vals, event_id=...)` on every request; `event_id` = the
  ground-truth join key. Capture columns = features + predict_names + `event_id` + `$$date$$`/`$$hour$$`.
- ⚠️⚠️ **Include `ground_truth` in `predict_names` with a sentinel**, e.g.
  `predict_names=["prediction", "ground_truth"]` and pass a placeholder (`"PENDING"` or `-1.0`) when the
  live label is unknown. If the target column is absent from the capture parquet, ingestion fails with
  `UNRESOLVED_COLUMN: ground_truth cannot be resolved`. Matured labels are joined later on `event_id`.
- ⚠️ Capture writes to `$PREDICTION_DATA_DIRECTORY/<HOSTNAME>.log`; if that env var is unset it silently
  writes to `/tmp/dummy.log` and the data is lost. Confirm the live path by printing
  `DataCaptureClient.is_dev_mode` in the model logs — `False` = a real endpoint (workspace = `True`).
- ⚠️⚠️ **Ingestion is a daily batch; there is no on-demand trigger** (`dataset-jobs` is a *query*). So
  live-endpoint drift/quality output appears **after the next daily ingestion**, not same-day. For an
  instant demo, use a **dataset-based (external) monitor** with **backdated** prediction/GT timestamps in
  S3 (bypasses the live endpoint).

## 7. Secrets & config hygiene
- Safety tooling blocks an agent from writing credentials into shared config or brute-forcing token routes.
  Have the **user** run the credential write (or add a permission rule); keep tokens in a gitignored file /
  personal env, not shared project config unless intended.
- 💡 **Centralize all instance config in one file** (env var > file > default precedence); read instance
  runtime basics from the injected `DOMINO_*` env. Then porting to a new instance = edit the config file,
  never the code.
- ⚠️⚠️ **A 200/201 on creation is not proof a Domino job/schedule actually works** — several failure
  modes (bad `mainRepoGitRef.type`, bad command syntax, missing env var) only surface at **execution**
  time, in the run's own logs, never in the creation response. Always trigger/wait for at least one real
  run and check its `executionStatus` + logs before considering a job "done."
- ⚠️⚠️ **Project-wide config-selector env vars (e.g. a `*_CONFIG` path override) are as collision-prone
  as secrets** when a project hosts more than one model/pipeline — they affect every job and workspace
  session project-wide, not just the one you're testing. Prefer a script explicitly locating its own
  config file (relative to itself) over a project-level env var for anything non-secret; reserve project
  env vars for the one thing that must vary per model and can't live in a committed file (the secret token).

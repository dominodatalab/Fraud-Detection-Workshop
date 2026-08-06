# Agent Prompt — set up a NEW model in Domino Model Monitoring

Paste the block below into **Claude Code** in a Domino workspace (git repo connected) on the target
instance, after copying this `monitoring_blueprints/` directory into the repo.

---

```
Set up THIS model as a monitored Domino Model API end-to-end, following the blueprint in
monitoring_blueprints/. Goal: registry-deployed endpoint with prediction capture + Domino
Model Monitoring (PSI drift, and model quality if ground truth is available), plus scheduled
jobs that put the endpoint under load, mature ground truth, and keep the drift schedule active.

READ FIRST and follow:
- monitoring_blueprints/SETUP_GUIDE.md        the ordered steps
- monitoring_blueprints/DOMINO_PLATFORM_NOTES.md  the gotchas/solutions — obey them
- monitoring_blueprints/config.template.yaml  the config contract
- monitoring_blueprints/PROOF.md              a fully worked, verified reference run

Do this:
1. Copy config.template.yaml to config.yaml IN THE SAME DIRECTORY
   (monitoring_blueprints/config.yaml, next to the template) — that is the DEFAULT location
   scripts/config.py looks for (it resolves relative to itself: root = monitoring_blueprints/,
   NOT the outer project's repo root). Fill it for THIS model: model.registered_model_name,
   the feature/prediction schema (with `sample` values per feature), monitoring.drift_features,
   jobs.*. Instance basics (project id, domain, DSE env, hardware tier) come from DOMINO_* env.
   Do NOT hard-code instance values in scripts — everything reads from config.yaml.
   If this project will monitor MORE THAN ONE model, either put each model's config.yaml in
   its own subdirectory (see example/config.yaml) and set MONITORING_CONFIG=<path> per script
   invocation, and set endpoint.token_env_var to a name namespaced per model (scheduled jobs
   have no per-job env vars, only project-wide ones — generic names collide).
2. Ensure the model is logged to the MLflow registry as an mlflow.pyfunc WITH prediction
   capture baked in (use templates/model_pyfunc_with_capture.py) and an explicit signature —
   NOT sklearn/xgboost flavors. predict_names must include a ground_truth sentinel.
3. Create a CAPTURE-ALIGNED baseline TrainingSet with scripts/create_baseline_trainingset.py:
   ONLY the captured columns (features + prediction outputs). No target/ground-truth/uncaptured
   columns (they cause UNRESOLVED_COLUMN on every ingestion).
4. Deploy FROM THE REGISTRY with monitoring enabled: python scripts/deploy_endpoint.py.
   Record the printed endpoint model_id/model_version_id in config.yaml. (File-based deploy
   can't be monitored — no training-data lineage.)
5. Smoke test (python scripts/smoke_test_endpoint.py) — pyfunc enforces the signature
   (int flags, float types matching the schema; response is a list-of-rows, not a dict).
6. Implement the two hooks in templates/load_generator.py (build_request + hidden_label) and
   templates/ground_truth.py (to_ground_truth) for this model (see example/ for a filled
   reference); then create ALL THREE scheduled jobs: python scripts/create_scheduled_jobs.py
   (load generator + ground truth generator + the self-cleaning "Ensure Drift Scheduled" job).
   Merge your branch first — jobs run committed code from jobs.git_ref, and mainRepoGitRef.type
   MUST be "branches" (plural) or every run silently fails at execution (not creation).
7. Set the baseline Training Data on the endpoint version in the UI (see step below) — this
   auto-creates the DMM monitor. Then: python scripts/setup_monitoring.py --find-monitor,
   --configure-drift <id>. Scheduling the check itself is handled automatically by the
   "Ensure Drift Scheduled" job from step 6 (it polls every 10 min, retries harmlessly until
   the first daily ingestion allows activation, then deletes its own scheduled job on success —
   don't hand-schedule it and don't wait on it manually).

Work autonomously via the internal /v4 and /model-monitor/v2/api endpoints (recipes in the
platform notes). Stop and ask me ONLY for: (a) pasting the model access token from the UI, and
(b) setting the baseline Training Data on the endpoint version in the UI (the one non-API step).
Be honest that DMM ingestion is daily — live drift output appears after the next ingestion, not
immediately; and model quality needs ground truth in an S3 datasource.

Keep config.yaml updated with the endpoint/monitor IDs as you go, and commit your work
(feature branch + PR; don't push to main without asking).
```

---

### Tips
- To reproduce the exact current setup, also point Claude at `Domino_Instance.md` → **CURRENT ACTIVE
  STATE** for the reference values.
- If an endpoint build fails at image pull with a registry-credential error, that's infrastructure
  (rotated pull secret), not your code — escalate (DOMINO_PLATFORM_NOTES.md §2).
- Same-day drift output isn't possible on the live/endpoint path (daily ingestion, no trigger); use a
  dataset-based external monitor if you need an instant demo (DOMINO_PLATFORM_NOTES.md §6).
- Reference implementation of this exact blueprint (config layout, filled hooks, verified end-to-end
  including catching real bugs): `monitoring_blueprints/example/` + `monitoring_blueprints/PROOF.md`.
  The parent repo's `DEPLOY_MONITORING.md`/`LESSONS_LEARNED.md` are a second worked example, specific
  to the retail scorecard model.

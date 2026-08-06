# monitoring_blueprints — reusable kit for Domino Model Monitoring

Drop this directory into a new project to stand up **any** model as a **monitored** Domino Model API on
**any** instance, with the load + ground-truth best practices. Model-agnostic and config-driven — no
values are hard-coded; you fill `config.yaml` and implement two small hooks for your model.

## Requirements
Standard Domino DSE compute environments have everything needed pre-installed: `mlflow`, `pandas`,
`numpy`, `scikit-learn`, `requests`, `joblib`, `pyyaml`, and `dominodatalab-data` (for
`domino_data.training_sets`, used by `create_baseline_trainingset.py`). No `requirements.txt` is
shipped here since nothing beyond the DSE default is needed — if your environment is stripped down,
install those.

## Start here
1. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** — the ordered "these are the steps" procedure.
2. **[PORTING_PROMPT.md](PORTING_PROMPT.md)** — paste into Claude Code to have the Agent do it.
3. **[DOMINO_PLATFORM_NOTES.md](DOMINO_PLATFORM_NOTES.md)** — the *why*: every Domino gotcha + solution
   (deploy, DMM API, scheduled jobs, capture, ingestion, git refs). Candidate content for Domino Skills.
4. **[config.template.yaml](config.template.yaml)** — copy to **`config.yaml` in this same directory**
   (`monitoring_blueprints/config.yaml`) and fill in. That's the default `scripts/config.py` looks for —
   *not* your project's outer repo root. Use `MONITORING_CONFIG=<path>` to point elsewhere (e.g. a
   model-specific subdirectory) if monitoring more than one model in the same project.
5. **[PROOF.md](PROOF.md)** — this kit isn't just written to look generic, it's **verified** generic:
   a second, independent model was built and deployed end-to-end using only these scripts, catching
   and fixing 3 real bugs (including one that had been silently breaking production) along the way.
   `example/` is that worked reference.

## What's here
```
SETUP_GUIDE.md              ordered steps (model-agnostic)
PORTING_PROMPT.md           one-shot Agent prompt
DOMINO_PLATFORM_NOTES.md    general Domino platform knowledge (the gotcha bible)
config.template.yaml        config contract (model schema, endpoint, monitoring, jobs)
scripts/  (reusable, config-driven — use as-is)
  config.py                 config loader (env > yaml > default) + helpers
  domino_api.py             workspace-proxy auth helper
  set_project_token.py      write the model access token to a project env var
  deploy_endpoint.py        registry-deploy a MONITORED endpoint + poll to Running
  create_baseline_trainingset.py  build a capture-aligned baseline TrainingSet
  setup_monitoring.py       drive the DMM API (register, PSI drift config, schedule)
  ensure_drift_scheduled.py self-deleting job: activates the drift schedule once possible, then removes itself
  create_scheduled_jobs.py  create all 3 scheduled jobs (load, ground-truth, ensure-drift-scheduled)
  smoke_test_endpoint.py    exercise the deployed endpoint
templates/  (model-specific — copy + implement)
  model_pyfunc_with_capture.py   the pyfunc capture contract (fill _score)
  load_generator.py              synthetic load + drift (fill build_request/hidden_label)
  ground_truth.py                label maturation (fill to_ground_truth)
```

## The rules that cause the most pain (all in the platform notes)
1. **Deploy from the registry, not file-based** — file-based endpoints have no training-data lineage, so
   monitoring can't attach a baseline.
2. **The baseline TrainingSet must contain ONLY captured columns** — any extra column (target, ground
   truth, uncaptured field) makes DMM fail every ingestion with `UNRESOLVED_COLUMN`.
3. **Ingestion is daily** — no same-day live output; the endpoint's baseline is immutable per version
   (change it → new version).
4. **Verify a scheduled job actually RUNS, not just that creation returned 200** — a wrong
   `mainRepoGitRef.type` (must be `"branches"`, not `"branch"`) is accepted at creation and only fails
   at execution time, silently, in the job's own logs. Found this the hard way: it broke a live
   production job for hours before anyone noticed.
5. **Namespace secrets if a project hosts more than one monitored model** — scheduled jobs have no
   per-job env vars, only project-wide ones (`endpoint.token_env_var` in config.yaml).
6. **A saved drift-check config is not a running schedule** — activation needs the first daily
   ingestion, so it fails on day one (expected). Don't leave this as a manual "check back tomorrow";
   `ensure_drift_scheduled.py` runs every 10 min, retries harmlessly, and deletes its own scheduled job
   the moment it succeeds — self-healing AND self-cleaning.

> These scripts started as snapshots of the parent `RetailAcquisitionScorecard` repo, then were fixed
> and re-verified against a second, fully independent model (`example/`) deployed end-to-end on a live
> Domino instance — including catching and fixing the two real bugs above, which the first pass missed.

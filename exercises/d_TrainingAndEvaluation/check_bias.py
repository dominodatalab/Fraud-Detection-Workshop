import mlflow
import os
from pathlib import Path

mlflow.set_experiment("Governance Bundle Experiment")

with mlflow.start_run() as run:
    mlflow.log_param("roc_auc", 0.52777)
    mlflow.log_metric("precision", 0.94234)

    # Save an output file (e.g., model, result JSON)
    output_path = "outputs/result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write('{"status": "pass", "details": "all checks passed"}')

    mlflow.log_artifact(output_path)

    print("Logged run to MLflow:", run.info.run_id)

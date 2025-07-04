import os, time, yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Directories
DOMINO_WORKING_DIR = os.environ.get("DOMINO_WORKING_DIR", ".")
DATA_DIR = DOMINO_WORKING_DIR.replace("code", "data")
ARTIFACT_DIR = DOMINO_WORKING_DIR.replace("code", "artifacts")
PROJECT = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")


def split_data(clean_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 2018):
    """
    Load the cleaned DataFrame and split into train/validation sets.
    Returns: df, X_train, X_val, y_train, y_val, feature list
    """
    df = clean_df.dropna(subset=["Class"]).copy()

    TARGET = "Class"
    drop_cols = ["Time", TARGET]
    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return df, X_train, X_val, y_train, y_val, features


def train_and_log(
    model, name: str,
    df: pd.DataFrame,
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
    features: list, clean_filename: str
):
    """
    Train the model, log parameters, metrics, plots, and model artifact to MLflow.
    """
    # Ensure artifact directory
    Path(ARTIFACT_DIR).mkdir(exist_ok=True, parents=True)

    with mlflow.start_run(run_name=name):
        # Log model hyperparameters
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("clean_filename", clean_filename)
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("num_rows", len(df))

        # Save params YAML
        params_yaml = {
            "model_name": model.__class__.__name__,
            "clean_filename": clean_filename,
            "num_features": len(features),
            "num_rows": len(df),
            "features": features,
        }
        yaml_path = os.path.join(ARTIFACT_DIR, f"{name.lower().replace(' ', '_')}_params.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(params_yaml, f, default_flow_style=False)
        mlflow.log_artifact(yaml_path, artifact_path="params")

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        # Predictions & metrics
        proba = model.predict_proba(X_val)[:, 1]
        pred = model.predict(X_val)
        metrics = {
            "roc_auc": roc_auc_score(y_val, proba),
            "pr_auc": average_precision_score(y_val, proba),
            "accuracy": accuracy_score(y_val, pred),
            "precision_fraud": precision_score(y_val, pred, pos_label=1),
            "recall_fraud": recall_score(y_val, pred, pos_label=1),
            "f1_fraud": f1_score(y_val, pred, pos_label=1),
            "fit_time_sec": fit_time,
        }
        mlflow.log_metrics(metrics)

        # Save validation predictions and metrics as CSVs to the dataset directory
        domino_datasource_dir = DOMINO_WORKING_DIR.replace('code', 'data')
        domino_project_name = PROJECT
        os.makedirs(os.path.join(domino_datasource_dir, domino_project_name), exist_ok=True)
        
        # Save validation predictions
        val_pred_df = pd.DataFrame({
            'y_true': y_val,
            'y_pred': pred,
            'y_proba': proba
        })
        pred_csv_path = os.path.join(domino_datasource_dir, domino_project_name, f"{name.lower().replace(' ', '_')}_val_predictions.csv")
        val_pred_df.to_csv(pred_csv_path, index=False)

        # Save metrics as CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_csv_path = os.path.join(domino_datasource_dir, domino_project_name, f"{name.lower().replace(' ', '_')}_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Inference signature & model logging
        signature = infer_signature(X_val, proba)
        input_example = X_val.iloc[:5]
        mlflow.sklearn.log_model(
            model,
            artifact_path=f"{name.lower().replace(' ', '_')}_model",
            registered_model_name=f"CC Fraud {name} Classifier",
            signature=signature,
            input_example=input_example
        )
        mlflow.set_tag("pipeline", "classifier_training_no_pca")
        mlflow.set_tag("model", name)

        # Plotting helpers
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})")
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        roc_path = os.path.join(ARTIFACT_DIR, f"{name.lower().replace(' ', '_')}_roc.png")
        plt.tight_layout(); plt.savefig(roc_path); plt.close()
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Precision-Recall Curve
        rec, prec, _ = precision_recall_curve(y_val, proba)
        plt.figure()
        plt.plot(rec, prec, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        pr_path = os.path.join(ARTIFACT_DIR, f"{name.lower().replace(' ', '_')}_pr.png")
        plt.tight_layout(); plt.savefig(pr_path); plt.close()
        mlflow.log_artifact(pr_path, artifact_path="plots")

        # Confusion Matrix
        cm = confusion_matrix(y_val, pred, normalize='true')
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Normalized Confusion Matrix')
        cm_path = os.path.join(ARTIFACT_DIR, f"{name.lower().replace(' ', '_')}_cm.png")
        plt.tight_layout(); plt.savefig(cm_path); plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Feature importances (if available)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            plt.figure()
            plt.bar(range(len(idx)), imp[idx])
            plt.xticks(range(len(idx)), [features[i] for i in idx], rotation=45, ha='right')
            plt.title('Top Feature Importances')
            fi_path = os.path.join(ARTIFACT_DIR, f"{name.lower().replace(' ', '_')}_fi.png")
            plt.tight_layout(); plt.savefig(fi_path); plt.close()
            mlflow.log_artifact(fi_path, artifact_path="plots")

    mlflow.end_run()
    return {
        "model_name": name,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "accuracy": metrics["accuracy"],
        "precision_fraud": metrics["precision_fraud"],
        "recall_fraud": metrics["recall_fraud"],
        "f1_fraud": metrics["f1_fraud"],
        "fit_time_sec": metrics["fit_time_sec"],
    }

def train_fraud(model_obj, model_name, clean_df, experiment_name, clean_filepath, random_state=None):

    # Set up experiment
    mlflow.set_experiment(experiment_name)

    print(f'loading data clean path {clean_filepath}')

    # Load data once
    df, X_train, X_val, y_train, y_val, features = split_data(
        clean_df,
        random_state=random_state
    )

    print(f'training model {model_name}')
    res = train_and_log(
        model_obj, model_name,
        df, X_train, X_val, y_train, y_val,
        features, clean_filepath
    )
    return res
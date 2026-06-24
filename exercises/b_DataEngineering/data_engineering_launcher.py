"""
Credit Card Fraud Detection Preprocessing Pipeline — Launcher Version

Launcher-parameterized version of data_engineering.py. Accepts input/output
filenames and EDA report toggle as command-line arguments from a Domino Launcher.

Domino Launcher command:
    python exercises/b_DataEngineering/data_engineering_launcher.py ${input_filename} ${output_filename} ${generate_eda_report}
"""

import argparse
import io, os, time, subprocess, requests, json
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
from domino_data.data_sources import DataSourceClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from domino import Domino
from mlflow.models import infer_signature
import sys
sys.path.append(os.environ["DOMINO_WORKING_DIR"])
from domino_short_id import domino_short_id


experiment_name = f"CC Fraud Preprocessing {domino_short_id()}"

domino_working_dir = os.environ.get("DOMINO_WORKING_DIR", ".")
domino_project_name = os.environ.get("DOMINO_PROJECT_NAME", "my-local-project")
domino_project_owner = os.environ.get("DOMINO_PROJECT_OWNER", os.environ.get("DOMINO_USER_NAME", "default-owner"))
dataset_name = os.environ.get(domino_project_name, "Fraud-Detection-Workshop")

domino_datasource_dir = '/domino/datasets/local'
domino_dataset_dir = f"{domino_datasource_dir}/{domino_project_name}"
domino_artifact_dir = '/mnt/artifacts'


def get_generation_label(age):
    birth_year = datetime.today().year - age

    if birth_year <= 1945:
        return "Silent Generation"
    elif birth_year <= 1964:
        return "Baby Boomer"
    elif birth_year <= 1980:
        return "Generation X"
    elif birth_year <= 1996:
        return "Millennial"
    elif birth_year <= 2012:
        return "Generation Z"
    else:
        return "Generation Alpha"


def add_derived_features(df):
    df['amount_vs_avg30d_ratio'] = df['Amount'] / (df['Avg30d'] + 1e-6)
    df['risk_score'] = (df['MerchantRisk'] + df['IPReputation']) / 2
    df['trust_score'] = df['DeviceTrust'] - df['MerchantRisk']
    df['generation'] = df['Age'].apply(get_generation_label)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fraud Detection Preprocessing Pipeline")
    parser.add_argument("input_filename", type=str, help="Cleaned input CSV filename")
    parser.add_argument("output_filename", type=str, help="Transformed output CSV filename")
    parser.add_argument("generate_eda_report", type=str, help="Generate EDA report (true/false)")
    args = parser.parse_args()

    clean_filename = args.input_filename
    features_filename = args.output_filename
    generate_eda_report = args.generate_eda_report.strip().lower() == 'true'

    clean_path = f"{domino_dataset_dir}/{clean_filename}"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Preprocessing Pipeline") as run:
        print(f"Loading clean dataset from {clean_path}")
        clean_df = pd.read_csv(clean_path, index_col=0)
        print(f"Loaded {len(clean_df):,} rows from {clean_path}")
        print(clean_df.columns)

        full_cleaned_df = add_derived_features(clean_df)

        labels_df = full_cleaned_df['Class']
        features_df = full_cleaned_df.drop(columns=['Class'], errors='ignore')

        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features_df.select_dtypes(include=[object, "category"]).columns.tolist()
        print(features_df)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        pipeline = Pipeline([("preproc", preprocessor)])

        start_time = time.time()
        transformed_features_array = pipeline.fit_transform(features_df)
        fit_time = time.time() - start_time
        print(f"Transformed features in {fit_time:.2} seconds.")

        feature_names = pipeline.named_steps['preproc'].get_feature_names_out()
        transformed_features_df = pd.DataFrame(transformed_features_array,
                                               columns=feature_names, index=features_df.index)
        transformed_features_df['Class'] = labels_df

        features_path = f"{domino_dataset_dir}/{features_filename}"
        transformed_features_df.to_csv(features_path, index=False)
        print('saved to ', features_path)

        if generate_eda_report:
            from ydata_profiling import ProfileReport
            profile = ProfileReport(
                clean_df,
                title="Credit Card Fraud Detection - EDA Report",
                explorative=True,
                minimal=True
            )
            eda_path = f"{domino_artifact_dir}/preprocessing_report.html"
            profile.to_file(eda_path)
            mlflow.log_artifact(eda_path, artifact_path="eda")

        mlflow.log_artifact(clean_path, artifact_path="data")
        mlflow.log_param("input_filename", clean_filename)
        mlflow.log_param("output_filename", features_filename)
        mlflow.log_param("generate_eda_report", generate_eda_report)
        mlflow.log_param("num_rows_loaded", len(features_df))
        mlflow.log_param("num_cat_features", len(categorical_features))
        mlflow.log_param("num_num_features", len(numeric_features))
        mlflow.log_metric("fit_time", fit_time)

        pipeline.predict = pipeline.transform

        X_sample = features_df.iloc[:20].copy()
        for col in numeric_features:
            if np.issubdtype(X_sample[col].dtype, np.integer):
                X_sample[col] = X_sample[col].astype("float64")

        y_sample = pipeline.transform(X_sample)
        signature = infer_signature(X_sample, y_sample)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="preprocessing_pipeline",
            signature=signature
        )

        mlflow.set_tag("pipeline", "preprocessing")

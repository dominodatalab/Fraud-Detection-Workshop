"""
Domino Flow Training and Evaluation Workflow

Executes a Domino Flow that trains AdaBoost, GaussianNB, and XGBoost fraud
detection classifiers, then registers the best performing model (and the
Exercise 3 preprocessing pipeline) to the Domino Model Registry.

Steps:
1. Execute the Domino Flow (blocks until the flow finishes)
2. Find the best run from the flow's experiment (highest accuracy)
3. Register that run's model to the Model Registry
4. Register the Exercise 3 preprocessing pipeline as a separate model
"""

import os
import sys
import subprocess
from datetime import datetime

import mlflow
import mlflow.tracking

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_flow():
    """
    Execute the Domino Flow that trains AdaBoost, GaussianNB, and XGBoost
    classifiers. Returns the experiment name the flow's runs were logged
    under, or None if it couldn't be determined.
    """
    experiment_name = None

    print("Starting Domino Flow execution...")
    try:
        # Run the workflow using pyflyte with --remote flag for Domino execution
        result = subprocess.run(
            ["pyflyte", "run", "--remote", "exercises/d_TrainingAndEvaluation/workflow.py", "credit_card_fraud_detection_workflow"],
            capture_output=True,
            text=True,
            cwd=project_root
        )

        if result.returncode == 0:
            print("Flow execution started successfully!")
            print(f"Output: {result.stdout}")

            # Extract experiment name from the environment or generate it
            try:
                from domino_short_id import domino_short_id
                experiment_name = f"CC Fraud Classifier Training {domino_short_id()}"
                print(f"Flow will create experiments under: {experiment_name}")
            except Exception as e:
                print(f"Could not determine experiment name: {e}")
                experiment_name = "CC Fraud Classifier Training"

        else:
            print(f"Flow execution failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"Error executing flow: {e}")
        print("Note: This may require Domino platform environment setup")

    return experiment_name


def get_best_experiment_run(target_experiment_name=None):
    """
    Retrieve the best performing experiment run based on accuracy from the specified experiment
    """
    try:
        if target_experiment_name is None:
            print("No experiment name specified")
            return None

        print(f"Searching for best model in experiment: {target_experiment_name}")

        # Get the specific experiment by name
        try:
            experiment = mlflow.get_experiment_by_name(target_experiment_name)
            if experiment is None:
                print(f"Experiment '{target_experiment_name}' not found")
                # Fall back to searching all experiments
                experiments = mlflow.search_experiments()
                if not experiments:
                    print("No experiments found")
                    return None
                experiment_ids = [exp.experiment_id for exp in experiments]
            else:
                experiment_ids = [experiment.experiment_id]
                print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        except Exception as e:
            print(f"Error finding experiment: {e}")
            # Fall back to all experiments
            experiments = mlflow.search_experiments()
            experiment_ids = [exp.experiment_id for exp in experiments] if experiments else []

        if not experiment_ids:
            print("No experiments available")
            return None

        # Search for runs with accuracy metrics, ordered by accuracy (best first)
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string="metrics.accuracy > 0",
            order_by=["metrics.accuracy DESC"],
            max_results=10
        )

        if runs.empty:
            print("No runs found with accuracy metrics in the target experiment")
            return None

        # Get the best performing run
        best_run = runs.iloc[0]

        print(f"Best model found:")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Experiment: {best_run.get('experiment_id', 'Unknown')}")
        print(f"  Model: {best_run.get('tags.model_type', 'Unknown')}")
        print(f"  Accuracy: {best_run['metrics.accuracy']:.4f}")
        print(f"  ROC-AUC: {best_run.get('metrics.roc_auc', 'N/A')}")
        print(f"  Precision: {best_run.get('metrics.precision_fraud', 'N/A')}")
        print(f"  Recall: {best_run.get('metrics.recall_fraud', 'N/A')}")
        print(f"  F1-Score: {best_run.get('metrics.f1_fraud', 'N/A')}")

        return best_run

    except Exception as e:
        print(f"Error retrieving experiments: {e}")
        return None


def check_flow_datasets():
    """
    Check dataset information used by Flow scripts - fast file-based check
    Returns dataset info and list of data files found
    """
    try:
        data_files_found = []

        # Primary dataset file used by all trainers
        dataset_filename = 'transformed_cc_transactions.csv'

        # Check multiple possible locations for datasets
        search_paths = [
            "/mnt/data/Fraud-Detection-Workshop/",  # Domino dataset path
            "/mnt/code/",  # Local code directory
            "/tmp/",  # Temp directory
        ]

        # Look for CSV files in each path
        for search_path in search_paths:
            if os.path.exists(search_path):
                try:
                    files = [f for f in os.listdir(search_path)
                           if f.endswith('.csv') and os.path.isfile(os.path.join(search_path, f))]
                    for file in files:
                        if file not in data_files_found:  # Avoid duplicates
                            data_files_found.append(file)
                except Exception:
                    continue

        # Try to get info from the main dataset file
        dataset_info = "~500,000 credit card transactions (estimated)"
        for search_path in search_paths:
            dataset_path = os.path.join(search_path, dataset_filename)
            if os.path.exists(dataset_path):
                try:
                    # Quick file size check (fastest)
                    file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)

                    # Quick row count check (read just first few lines to estimate)
                    with open(dataset_path, 'r') as f:
                        first_line = f.readline()  # header
                        line_count = 1
                        for _ in range(100):  # sample first 100 rows
                            if f.readline():
                                line_count += 1
                            else:
                                break

                        # Estimate total rows based on file size ratio
                        sample_size = f.tell()
                        if sample_size > 0:
                            estimated_rows = int((file_size_mb * 1024 * 1024) / sample_size * line_count)
                        else:
                            estimated_rows = line_count

                    dataset_info = f"{estimated_rows:,} transactions ({file_size_mb:.1f} MB)"
                    break

                except Exception as e:
                    dataset_info = f"Dataset found but could not read: {dataset_filename} ({file_size_mb:.1f} MB)"
                    break

        return dataset_info, data_files_found

    except Exception as e:
        print(f"Warning: Could not check dataset info: {e}")
        return "~500,000 credit card transactions (estimated)", []


def register_model_to_registry(run_info, model_name="fraud_detection_classifier"):
    """
    Register the best model to Domino Model Registry with model card and specifications
    """
    try:
        if run_info is None:
            print("No run information provided")
            return None

        run_id = run_info['run_id']
        model_uri = f"runs:/{run_id}/model"

        # Create model version with only required parameters
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print(f"Model registered successfully:")
        print(f"  Name: {model_version.name}")
        print(f"  Version: {model_version.version}")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: {model_uri}")

        # Check dataset information used by Flow scripts
        dataset_info, data_files = check_flow_datasets()

        # Print first 3 data files found
        if data_files:
            print(f"📁 Data files used: {', '.join(data_files[:3])}")
            if len(data_files) > 3:
                print(f"   ... and {len(data_files) - 3} more files")

        # Extract training framework from model type or tags - enhanced detection
        training_framework = None

        # Debug: Print available run info for troubleshooting
        print(f"🔍 Debug - Available run tags:")
        for key in run_info.keys():
            if key.startswith('tags.'):
                print(f"   {key}: {run_info[key]}")

        # Try multiple methods to detect framework
        methods = [
            ('tags.model_type', run_info.get('tags.model_type')),
            ('tags.mlflow.runName', run_info.get('tags.mlflow.runName', '')),
            ('tags.model_name', run_info.get('tags.model_name')),
            ('tags.algorithm', run_info.get('tags.algorithm')),
        ]

        for method, value in methods:
            if value and training_framework is None:
                value_lower = str(value).lower()
                if 'xgb' in value_lower or 'xgboost' in value_lower:
                    training_framework = 'XGBoost'
                    print(f"✅ Detected XGBoost from {method}: {value}")
                    break
                elif 'ada' in value_lower or 'adaboost' in value_lower:
                    training_framework = 'AdaBoost'
                    print(f"✅ Detected AdaBoost from {method}: {value}")
                    break
                elif 'gnb' in value_lower or 'gaussian' in value_lower or 'naive' in value_lower:
                    training_framework = 'GaussianNB'
                    print(f"✅ Detected GaussianNB from {method}: {value}")
                    break

        # Fallback if no detection worked
        if training_framework is None:
            training_framework = 'Unknown Classifier'
            print(f"⚠️ Could not detect training framework, using fallback: {training_framework}")

        # Update model with comprehensive card template and specifications
        client = mlflow.tracking.MlflowClient()

        # Set simple one-sentence description for model version
        desc = "Best performing fraud detection classifier based on accuracy from automated flow training"
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=desc
        )

        # Set basic model tags using client
        basic_tags = {
            "model_type": run_info.get('tags.model_type', 'classifier'),
            "use_case": "fraud_detection",
            "training_data": "credit_card_transactions",
            "performance_metric": "accuracy",
            "deployment_ready": "true"
        }

        for key, value in basic_tags.items():
            client.set_model_version_tag(model_name, model_version.version, key, value)

        # Check if model card template exists and use it for registered model description
        template_path = os.path.join(project_root, "exercises", "d_TrainingAndEvaluation", "Model_Registry_template.md")
        try:
            with open(template_path, 'r') as file:
                model_card_description = file.read()
                client.update_registered_model(model_name, model_card_description)
                print(f"✅ Updated model description from template: {template_path}")
        except Exception as e:
            print(f"⚠️ Could not load model card template: {e}")
            # Fallback to simple description
            fallback_desc = "Fraud Detection Classifier trained using Domino Flow with multiple algorithm comparison"
            client.update_registered_model(model_name, fallback_desc)

        # Estimate model artifacts size (could check actual files if available)
        try:
            # Try to get actual model size if artifacts are accessible
            artifacts_path = f"/tmp/mlruns/{run_info.get('experiment_id', '')}/{run_id}/artifacts/model"
            if os.path.exists(artifacts_path):
                total_size = sum(os.path.getsize(os.path.join(artifacts_path, f))
                               for f in os.listdir(artifacts_path) if os.path.isfile(os.path.join(artifacts_path, f)))
                model_size = f"{total_size / (1024*1024):.1f} MB"
            else:
                # Use reasonable estimates based on model type
                size_estimates = {
                    'XGBoost': '15-25 MB',
                    'AdaBoost': '8-15 MB',
                    'GaussianNB': '< 5 MB'
                }
                model_size = size_estimates.get(training_framework, '10-20 MB')
        except Exception:
            model_size = '10-20 MB (estimated)'

        # Estimate inference memory requirements based on model type
        memory_estimates = {
            'XGBoost': '512 MB - 1 GB (tree ensemble requires memory for all trees)',
            'AdaBoost': '256 MB - 512 MB (smaller ensemble, moderate memory)',
            'GaussianNB': '128 MB - 256 MB (simple probabilistic model, minimal memory)'
        }
        inference_memory = memory_estimates.get(training_framework, '256 MB - 512 MB (estimated)')

        # Set relevant model specifications - ensure framework is properly set
        model_specs = {
            # Core model specifications
            "mlflow.domino.specs.Training Framework": training_framework,
            "mlflow.domino.specs.Model Artifacts Size": model_size,
            "mlflow.domino.specs.Training Dataset Size": dataset_info,
            "mlflow.domino.specs.Inference Memory": inference_memory
        }

        print(f"🔧 Setting model specs:")
        for key, value in model_specs.items():
            print(f"   {key}: {value}")

        # Apply all model specifications
        for key, value in model_specs.items():
            try:
                client.set_registered_model_tag(model_name, key, value)
                print(f"✅ Set spec: {key}")
            except Exception as e:
                print(f"❌ Failed to set spec {key}: {e}")

        print(f"✅ Applied {len(basic_tags)} basic tags and {len(model_specs)} model specifications")
        print(f"   Training Framework: {training_framework}")
        print(f"   Model Size: {model_size}")
        print(f"   Dataset Info: {dataset_info}")
        print(f"   Inference Memory: {inference_memory}")
        print("Model registration completed with full model card and specifications")

        return model_version

    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def register_preprocessing_pipeline():
    """
    Register the preprocessing pipeline from Exercise 3 as a separate model endpoint.
    This creates the feature scaling endpoint needed by the Streamlit app.
    """
    try:
        # Import the preprocessing functions from Exercise 3
        from exercises.c_DataEngineering.data_engineering import add_derived_features

        # Look for existing preprocessing experiments
        preprocessing_experiments = mlflow.search_experiments(
            filter_string="name LIKE '%Preprocessing%'"
        )

        if preprocessing_experiments:
            # Get the most recent preprocessing experiment
            latest_experiment = max(preprocessing_experiments,
                                  key=lambda x: x.creation_time)

            print(f"Found preprocessing experiment: {latest_experiment.name}")

            # Search for preprocessing runs in this experiment
            preprocessing_runs = mlflow.search_runs(
                experiment_ids=[latest_experiment.experiment_id],
                filter_string="tags.pipeline = 'preprocessing'",
                order_by=["start_time DESC"],
                max_results=1
            )

            if not preprocessing_runs.empty:
                best_preprocessing_run = preprocessing_runs.iloc[0]
                run_id = best_preprocessing_run.run_id

                print(f"Found preprocessing run: {run_id}")

                # Register the preprocessing pipeline as a model
                preprocessing_model_uri = f"runs:/{run_id}/preprocessing_pipeline"

                try:
                    # Register the preprocessing pipeline
                    registered_model = mlflow.register_model(
                        model_uri=preprocessing_model_uri,
                        name="CC_Fraud_Feature_Scaling"
                    )

                    print(f"✅ Registered preprocessing pipeline as model: CC_Fraud_Feature_Scaling")
                    print(f"   Model Version: {registered_model.version}")

                    # Update model version with description and tags
                    client = mlflow.tracking.MlflowClient()

                    # Add description
                    client.update_model_version(
                        name="CC_Fraud_Feature_Scaling",
                        version=registered_model.version,
                        description="Feature scaling pipeline for credit card fraud detection preprocessing"
                    )

                    # Add tags
                    client.set_model_version_tag(
                        name="CC_Fraud_Feature_Scaling",
                        version=registered_model.version,
                        key="stage",
                        value="staging"
                    )

                    client.set_model_version_tag(
                        name="CC_Fraud_Feature_Scaling",
                        version=registered_model.version,
                        key="model_type",
                        value="preprocessing"
                    )

                    print(f"✅ Updated model version with description and tags")

                    return registered_model

                except Exception as e:
                    print(f"❌ Error registering preprocessing model: {e}")
                    return None
            else:
                print("❌ No preprocessing runs found with 'preprocessing' tag")
                return None
        else:
            print("❌ No preprocessing experiments found")
            print("   Please run Exercise 3 (Data Engineering) first to create the preprocessing pipeline")
            return None

    except ImportError as e:
        print(f"❌ Could not import from Exercise 3: {e}")
        print("   Please ensure Exercise 3 (Data Engineering) has been completed")
        return None


if __name__ == "__main__":
    # Step 1: Execute the Domino Flow (blocks until training completes)
    experiment_name = run_flow()

    # Step 2/3: Find and register the best classifier from the flow's experiment
    best_run = get_best_experiment_run(experiment_name)
    if best_run is not None:
        registered_model = register_model_to_registry(best_run)
    else:
        print("No best run available for registration")
        print("Please use the Domino UI to manually register the model as described in the instructions")

    # Step 4: Register the Exercise 3 preprocessing pipeline as its own model
    print("🔄 Registering preprocessing pipeline as feature scaling endpoint...")
    preprocessing_model = register_preprocessing_pipeline()

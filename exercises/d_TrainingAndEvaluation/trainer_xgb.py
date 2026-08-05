# File: trainer_xgb.py
import os
import sys
from pathlib import Path
import json
from xgboost import XGBClassifier

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud

# Load DataFrame from dataset 
try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
    print('using workflow input: transformed_filename', transformed_df_filename)
except FileNotFoundError as e:
    print('file not found error', e)
    transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'XGBoost'
model_obj = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            max_bin=64,
            sketch_eps=0.2,
            use_label_encoder=False,
            eval_metric="auc",
        )

res = train_fraud(model_obj, model_name, transformed_df_filename)

DROP = {"threshold_scan", "curves"}
small = {k: v for k, v in res.items() if k not in DROP}

print(f"Training {model_name} completed successfully")
print(json.dumps({k: small.get(k) for k in ['roc_auc','f1_fraud','accuracy','log_loss']}, indent=2))

out_path = Path("/workflow/outputs/results")
if out_path.parent.exists():
    out_path.write_text(json.dumps(small))  # JSON, not str(dict)


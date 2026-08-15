# File: trainer_xgb.py
import os
import sys
import argparse
from pathlib import Path
import json
from xgboost import XGBClassifier

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud


def parse_args():
    parser = argparse.ArgumentParser(description="Train an XGBoost fraud detection classifier")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of boosting rounds (default: 100)")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Boosting learning rate / eta (default: 0.1)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum tree depth (default: 4)")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Subsample ratio of training rows per boosting round (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8,
                        help="Subsample ratio of columns per tree (default: 0.8)")
    parser.add_argument("--tree-method", type=str, default="hist",
                        help="Tree construction algorithm (default: hist)")
    parser.add_argument("--max-bin", type=int, default=64,
                        help="Max number of histogram bins for the hist tree method (default: 64)")
    parser.add_argument("--eval-metric", type=str, default="auc",
                        help="Evaluation metric used during training (default: auc)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Transformed features CSV filename. Overrides the "
                             "/workflow/inputs/transformed_filename Domino Flow input if set.")
    return parser.parse_args()


args = parse_args()

# Load DataFrame from dataset
if args.dataset:
    transformed_df_filename = args.dataset
    print('using --dataset arg: transformed_filename', transformed_df_filename)
else:
    try:
        transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
        print('using workflow input: transformed_filename', transformed_df_filename)
    except FileNotFoundError as e:
        print('file not found error', e)
        transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'XGBoost'
model_obj = XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            tree_method=args.tree_method,
            n_jobs=-1,
            max_bin=args.max_bin,
            sketch_eps=0.2,
            use_label_encoder=False,
            eval_metric=args.eval_metric,
        )

res = train_fraud(model_obj, model_name, transformed_df_filename, n_estimators=args.n_estimators)

DROP = {"threshold_scan", "curves"}
small = {k: v for k, v in res.items() if k not in DROP}

print(f"Training {model_name} completed successfully")
print(json.dumps({k: small.get(k) for k in ['roc_auc','f1_fraud','accuracy','log_loss']}, indent=2))

out_path = Path("/workflow/outputs/results")
if out_path.parent.exists():
    out_path.write_text(json.dumps(small))  # JSON, not str(dict)


# File: trainer_gnb.py
import os
import sys
import argparse
from pathlib import Path
import json
from sklearn.naive_bayes import GaussianNB

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exercises.d_TrainingAndEvaluation.generic_trainer import train_fraud


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GaussianNB fraud detection classifier")
    parser.add_argument("--var-smoothing", type=float, default=1e-9,
                        help="Portion of the largest variance added to variances for calculation stability (default: 1e-9)")
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

model_name = 'GaussianNB'
model_obj = GaussianNB(var_smoothing=args.var_smoothing)

res = train_fraud(model_obj, model_name, transformed_df_filename)

DROP = {"threshold_scan", "curves"}
small = {k: v for k, v in res.items() if k not in DROP}

print(f"Training {model_name} completed successfully")
print(json.dumps({k: small.get(k) for k in ['roc_auc','f1_fraud','accuracy','log_loss']}, indent=2))

out_path = Path("/workflow/outputs/results")
if out_path.parent.exists():
    out_path.write_text(json.dumps(small))  # JSON, not str(dict)


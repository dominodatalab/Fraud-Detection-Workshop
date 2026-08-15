# compare_training_results.py
import json
from pathlib import Path

# Inputs
def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

# Read the two result blobs
ada_blob = json.loads(read_input("ada_results"))
gnb_blob = json.loads(read_input("gnb_results"))
xgb_blob = json.loads(read_input("xgb_results"))

# Consolidate
consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob, "XGBoost": xgb_blob}
print('consolidated')

best_model, best_metric = '', 0
for name, blob in consolidated.items():
    if blob['roc_auc'] > best_metric:
        print('better model', name, blob['roc_auc'])
        best_model = name
        best_metric = blob['roc_auc']
print('best model', best_model)
print('bmetric',best_metric)

# Prepare output
OUT_DIR = Path("/workflow/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "consolidated"

OUT_FILE.write_text(f'model with highest AUC - {best_model}')

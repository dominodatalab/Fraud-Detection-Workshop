# compare_training_results.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR = Path("/workflow/outputs")
OUT_FILE = OUT_DIR / "comparison"

HIGHER = {
    "roc_auc", "pr_auc", "f1_fraud", "precision_fraud", 
    "recall_fraud", "accuracy", "ks"
}
LOWER = {
    "log_loss", "brier", "ece", "fit_time_sec", 
    "predict_time_sec", "inf_ms_row", "model_size_kb"
}

def read_input(name: str) -> str:
    p = Path(f"/workflow/inputs/{name}")
    return p.read_text().strip() if p.exists() else name

def to_dict(blob: str):
    try:
        # Try to parse as JSON first
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        # If that fails, try reading as file path
        try:
            p = Path(blob)
            if p.exists():
                return json.loads(p.read_text())
            else:
                raise FileNotFoundError(f"File not found: {blob}")
        except Exception as e:
            print(f"Error parsing {blob}: {e}")
            raise

ada_blob = to_dict(read_input("ada_results"))
gnb_blob = to_dict(read_input("gnb_results"))
consolidated = {"AdaBoost": ada_blob, "GaussianNB": gnb_blob}

df = pd.DataFrame.from_dict(consolidated, orient="index")
df.index.name = "model"

# Convert non-scalar values to NaN
for c in df.columns:
    df[c] = df[c].apply(lambda v: v if pd.api.types.is_scalar(v) else np.nan)

# Numeric coercion with try/except for newer pandas
for c in df.columns:
    if df[c].dtype == object:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass

# Rank numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
rank_cols = {}

for col in numeric_cols:
    if col in HIGHER:
        asc = False
    elif col in LOWER:
        asc = True
    else:
        asc = any(k in col.lower() for k in ("loss", "time", "err", "ms"))
    rank_cols[f"{col}_rank"] = df[col].rank(ascending=asc, method="min")

if rank_cols:
    df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Convert to records and handle NaN values
payload = df.reset_index().replace({np.nan: None}).to_dict(orient="records")

# Write output
OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[compare] wrote {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")
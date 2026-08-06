"""Generate synthetic training data + baseline/drifted population profiles for the
blueprint worked example. Deliberately unrelated to any real project's data.

Model: predict whether a synthetic "widget order" is high-value from feature_a (numeric),
feature_b (numeric), feature_c (categorical). Injected drift lives in feature_a's mean and
feature_c's category distribution.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

CAT_ENCODING = {"cat_a": 0, "cat_b": 1, "cat_c": 2}

BASELINE_PROFILE = {"feature_a_mean": 50.0, "feature_a_std": 10.0,
                    "feature_b_mean": 20.0, "feature_b_std": 5.0,
                    "cat_probs": {"cat_a": 0.5, "cat_b": 0.3, "cat_c": 0.2}}
DRIFTED_PROFILE = {"feature_a_mean": 80.0, "feature_a_std": 10.0,   # shifted mean -> PSI breach
                   "feature_b_mean": 20.0, "feature_b_std": 5.0,    # unchanged -> stable
                   "cat_probs": {"cat_a": 0.1, "cat_b": 0.3, "cat_c": 0.6}}  # shifted -> PSI breach


def sample(profile: dict, n: int, rng) -> pd.DataFrame:
    feature_a = rng.normal(profile["feature_a_mean"], profile["feature_a_std"], n)
    feature_b = rng.normal(profile["feature_b_mean"], profile["feature_b_std"], n)
    cats = list(profile["cat_probs"].keys())
    probs = list(profile["cat_probs"].values())
    feature_c = rng.choice(cats, size=n, p=probs)
    return pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b, "feature_c": feature_c})


def true_label(df: pd.DataFrame, rng) -> np.ndarray:
    """Ground-truth generating function (hidden from the model at inference)."""
    c_enc = df["feature_c"].map(CAT_ENCODING).astype(float)
    logit = 0.04 * (df["feature_a"] - 50) + 0.08 * (df["feature_b"] - 20) - 0.5 * c_enc
    prob = 1 / (1 + np.exp(-logit))
    return (rng.uniform(0, 1, len(df)) < prob).astype(int)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    df = sample(BASELINE_PROFILE, 5000, rng)
    df["target"] = true_label(df, rng)
    df.to_csv(os.path.join(DATA_DIR, "training_data.csv"), index=False)
    json.dump(BASELINE_PROFILE, open(os.path.join(DATA_DIR, "population_baseline.json"), "w"), indent=2)
    json.dump(DRIFTED_PROFILE, open(os.path.join(DATA_DIR, "population_drifted.json"), "w"), indent=2)
    print(f"wrote {len(df)} training rows, bad-rate={df['target'].mean():.3f}")


if __name__ == "__main__":
    main()

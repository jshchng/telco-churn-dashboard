import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from pathlib import Path
from src.features import create_features
from src.data import basic_clean

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"

def load_model(path: str = None):
    path = path or MODEL_PATH
    return joblib.load(path)

def score_dataframe(df: pd.DataFrame, model=None) -> pd.DataFrame:
    model = model or load_model()
    df = basic_clean(df)
    df = create_features(df)
    X = df.select_dtypes(include=["number"]).fillna(0)
    # Align columns to those used in training
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        X = X[feature_names]
    probs = model.predict_proba(X)[:,1]
    out = df.copy()
    out["pred_prob"] = probs
    out["risk_label"] = pd.cut(out["pred_prob"], bins=[-0.001, 0.2, 0.5, 1.0], labels=["low", "medium", "high"])
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV path to score")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    scored = score_dataframe(df)
    print(scored[["pred_prob", "risk_label"]].head())

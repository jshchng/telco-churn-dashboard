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

def score_dataframe(df: pd.DataFrame, model=None, allow_scaler_fallback: bool = False) -> pd.DataFrame:
    """Score a dataframe with the provided model.

    By default this function expects a scaler saved during training at models/scaler.joblib.
    If that scaler is missing the function will raise a ValueError unless
    `allow_scaler_fallback=True` is passed (not recommended for production).
    """
    model = model or load_model()
    df = basic_clean(df)
    df = create_features(df)
    X = df.select_dtypes(include=["number"]).fillna(0)
    # Align columns to those used in training (fill missing with zeros)
    scaler_path = MODEL_DIR / "scaler.joblib"
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        missing = [f for f in feature_names if f not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
        # Reorder columns to match training feature order BEFORE scaling
        X = X[feature_names]

    # Load scaler saved during training, otherwise fit a temporary scaler on input X
    from sklearn.preprocessing import StandardScaler
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    if scaler is None:
        if not allow_scaler_fallback:
            raise ValueError("scaler.joblib not found in models/. Score aborted. If you really want to fit a local scaler at prediction time pass allow_scaler_fallback=True (not recommended).")
        # Fallback: fit a local scaler on the scoring data (not recommended)
        try:
            scaler = StandardScaler()
            scaler.fit(X)
        except Exception:
            scaler = None

    if scaler is not None:
        # Scaler expects the same column order it was trained on; X is already ordered
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    probs = model.predict_proba(X)[:,1]
    out = df.copy()
    out["pred_prob"] = probs
    out["risk_label"] = pd.cut(out["pred_prob"], bins=[-0.001, 0.2, 0.5, 1.0], labels=["low", "medium", "high"])
    # Sanitize columns for downstream display (Streamlit/Arrow-friendly)
    def _sanitize_dataframe_for_display(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        # Ensure pred_prob is float
        if 'pred_prob' in df.columns:
            df['pred_prob'] = pd.to_numeric(df['pred_prob'], errors='coerce').astype(float)
        # Ensure risk_label is string
        if 'risk_label' in df.columns:
            df['risk_label'] = df['risk_label'].astype(str)

        for col in df.select_dtypes(include=['object']).columns:
            # Try convert to numeric if most values parse as numbers
            coerced = pd.to_numeric(df[col], errors='coerce')
            non_na_ratio = coerced.notna().mean()
            if non_na_ratio > 0.9:
                # mostly numeric-like, convert
                df[col] = coerced
            else:
                # fallback to safe string representation
                df[col] = df[col].astype(str).fillna('')

        # Convert categorical types to string
        for col in df.select_dtypes(include=['category']).columns:
            df[col] = df[col].astype(str)

        return df

    out = _sanitize_dataframe_for_display(out)
    return out


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed-type/object columns to types safe for pyarrow/Streamlit.
    - numeric-like object columns are coerced when >=90% parse as numeric
    - remaining object/category columns are stringified (missing -> empty string)
    - ensures 'pred_prob' is float and 'risk_label' is str
    """
    df = df.copy()
    if 'pred_prob' in df.columns:
        df['pred_prob'] = pd.to_numeric(df['pred_prob'], errors='coerce').astype(float)
    if 'risk_label' in df.columns:
        df['risk_label'] = df['risk_label'].astype(str)

    # Coerce object columns safely
    for col in df.select_dtypes(include=['object']).columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        non_na_ratio = coerced.notna().mean()
        if non_na_ratio >= 0.9:
            df[col] = coerced
        else:
            # Make sure missing values are empty strings to avoid 'nan' literal
            df[col] = df[col].astype(str).fillna('')

    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(str).fillna('')

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV path to score")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    scored = score_dataframe(df)
    print(scored[["pred_prob", "risk_label"]].head())

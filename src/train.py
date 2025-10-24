import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.data import load_raw, basic_clean, save_processed
from src.features import create_features

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def prepare_Xy(df: pd.DataFrame):
    """
    Using original dataset variable names:
    - target: 'Churn Label' (Yes/No) or 'Churn Value' (1/0)
    - numeric features: select numbers and engineered numeric features
    """
    df = create_features(df)

    # target handling
    if "Churn Value" in df.columns:
        y = df["Churn Value"]
    elif "Churn Label" in df.columns:
        # map Yes/No to 1/0; if already mapped in data.basic_clean it's numeric
        y = df["Churn Label"].map({"Yes": 1, "No": 0}) if df["Churn Label"].dtype == object else df["Churn Label"]
    elif "Customer Status" in df.columns:
        # Customer Status: Churned / Stayed / Joined
        y = df["Customer Status"].apply(lambda v: 1 if str(v).lower().strip() == "churned" else 0)
    else:
        raise ValueError("No churn target column found. Expect 'Churn Value' or 'Churn Label' or 'Customer Status'.")

    # drop identifier columns (best-effort) and any columns that look like the target
    drop_candidates = [c for c in df.columns if any(k in c.lower() for k in ("customer id", "customerid", "id"))]
    # also drop any column that mentions churn or status (case-insensitive)
    drop_candidates += [c for c in df.columns if any(k in c.lower() for k in ("churn", "customer status", "status"))]
    drop_candidates = list(set(drop_candidates))
    X = df.drop(columns=drop_candidates, errors="ignore")

    # select numeric features only for model (you can extend to dummies later)
    X_num = X.select_dtypes(include=["number"]).copy().fillna(0)

    return X_num, y


def detect_leakage(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95):
    """Compute univariate ROC-AUC for each numeric feature and report suspicious ones.
    Returns list of features with AUC >= threshold.
    """
    from sklearn.metrics import roc_auc_score
    suspicious = []
    results = []
    for col in X.columns:
        # need at least two unique values and not constant
        if X[col].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y, X[col])
        except Exception:
            # if values not appropriate, try ranking
            try:
                auc = roc_auc_score(y, pd.factorize(X[col])[0])
            except Exception:
                continue
        results.append((col, auc))
        if auc >= threshold or auc <= (1 - threshold):
            suspicious.append((col, auc))

    results = sorted(results, key=lambda x: abs(0.5 - x[1]), reverse=True)
    print("Top 10 features by univariate AUC distance from 0.5:")
    for col, auc in results[:10]:
        print(f"  {col}: AUC={auc:.4f}")
    if suspicious:
        print(f"\nWARNING: Suspicious features detected (univariate AUC >= {threshold} or <= {1-threshold}):")
        for col, auc in suspicious:
            print(f"  {col}: AUC={auc:.4f}")
    return suspicious

def train_and_select(X_train, y_train, X_val, y_val):
    # Baseline logistic
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict_proba(X_val)[:,1]
    auc_lr = roc_auc_score(y_val, pred_lr)
    print(f"Logistic AUC: {auc_lr:.4f}")

    # RandomForest with small tune
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100,200],
        "max_depth": [5,10,20,None],
        "min_samples_split": [2,5,10]
    }
    rsearch = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=6, scoring="roc_auc", cv=3, random_state=42, n_jobs=-1)
    rsearch.fit(X_train, y_train)
    best_rf = rsearch.best_estimator_
    pred_rf = best_rf.predict_proba(X_val)[:,1]
    auc_rf = roc_auc_score(y_val, pred_rf)
    print(f"RandomForest AUC: {auc_rf:.4f}")

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict_proba(X_val)[:,1]
    auc_xgb = roc_auc_score(y_val, pred_xgb)
    print(f"XGBoost AUC: {auc_xgb:.4f}")

    aucs = {
        "logistic": (auc_lr, lr),
        "rf": (auc_rf, best_rf),
        "xgb": (auc_xgb, xgb)
    }
    best_name = max(aucs.items(), key=lambda x: x[1][0])[0]
    best_model = aucs[best_name][1]
    print(f"Selected {best_name} as best model.")

    return best_model

def save_model(model, name="best_model.joblib"):
    out = MODEL_DIR / name
    joblib.dump(model, out)
    print(f"Saved model to {out}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to CSV (optional). Otherwise loads default raw file.")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = load_raw()

    df = basic_clean(df)
    save_processed(df)

    X, y = prepare_Xy(df)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Detect potential leakage via univariate AUCs on training set
    suspicious = detect_leakage(X_train, y_train, threshold=0.95)
    if suspicious:
        print("Suspicious features found. These will be dropped automatically before training:")
        cols_to_drop = [col for col, _ in suspicious]
        for col, auc in suspicious:
            print(f"  - {col}: AUC={auc:.4f}")
        # Drop from both train and validation sets
        X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_val = X_val.drop(columns=cols_to_drop, errors='ignore')
        print(f"Dropped {len(cols_to_drop)} suspicious features: {cols_to_drop}")

    # Fit scaler on training set only and transform both sets
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    # Persist scaler so prediction code can apply the same transform
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    model = train_and_select(X_train_scaled, y_train, X_val_scaled, y_val)
    save_model(model)

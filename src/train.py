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

    # drop identifier columns (best-effort)
    drop_cols = [c for c in ["Customer ID", "CustomerID", "CustomerID ", "CustomerID"] if c in df.columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [c for c in ["Churn Label", "Churn Value", "Customer Status"] if c in df.columns], errors="ignore")

    # select numeric features only for model (you can extend to dummies later)
    X_num = X.select_dtypes(include=["number"]).copy().fillna(0)

    return X_num, y

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
    X_train, X_val, y_train, y_val = X.loc[:int(len(X)*0.8)-1], X.loc[int(len(X)*0.8):], y.iloc[:int(len(y)*0.8)], y.iloc[int(len(y)*0.8):]
    # fallback to sklearn split if index slicing above misbehaves
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception:
        pass

    model = train_and_select(X_train, y_train, X_val, y_val)
    save_model(model)

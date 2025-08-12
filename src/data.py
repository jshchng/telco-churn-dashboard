from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILENAME = "telco_customer_churn.csv"

def load_raw(filename: str = DEFAULT_FILENAME) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning while preserving original column names.
    - trims whitespace from column names and string values
    - converts numeric-ish columns (Total Charges, Monthly Charge) to numeric
    - maps Yes/No columns to 1/0 **only** if column values are exactly 'Yes'/'No'
    - fills numeric NaNs with median
    """
    df = df.copy()

    # Trim column whitespace (keeps capitalization/spaces)
    df.columns = [c.strip() for c in df.columns]

    # Trim string values
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Convert key numeric columns if present
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"].replace({"": None, " ": None}), errors="coerce")
    if "Monthly Charge" in df.columns:
        df["Monthly Charge"] = pd.to_numeric(df["Monthly Charge"].replace({"": None, " ": None}), errors="coerce")

    # Map exact 'Yes'/'No' columns to 1/0
    for c in df.columns:
        unique = df[c].dropna().unique()
        if set(unique).issubset({"Yes", "No"}):
            df[c] = df[c].map({"Yes": 1, "No": 0})

    # Fill numeric NaNs with median
    for c in df.select_dtypes(include=["number"]).columns:
        if df[c].isna().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    return df

def save_processed(df: pd.DataFrame, filename: str = "clean.csv") -> Path:
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    return out

if __name__ == "__main__":
    df = load_raw()
    df = basic_clean(df)
    p = save_processed(df)
    print(f"Saved processed data to {p}")

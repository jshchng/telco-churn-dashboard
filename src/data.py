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

    # Normalize common empty tokens to <NA> so trimming doesn't create 'nan' strings
    df = df.replace({"": pd.NA, " ": pd.NA})

    # Trim string values safely (preserve non-strings and missing values)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert well-known numeric columns if present (conservative)
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    if "Monthly Charge" in df.columns:
        df["Monthly Charge"] = pd.to_numeric(df["Monthly Charge"], errors="coerce")

    # Heuristic: coerce object columns to numeric only when most values parse as numbers
    for c in df.select_dtypes(include=["object"]).columns:
        # try to coerce; only keep if >=90% of non-null values become numeric
        coerced = pd.to_numeric(df[c], errors="coerce")
        non_null_before = df[c].notna().sum()
        non_null_after = coerced.notna().sum()
        if non_null_before > 0 and (non_null_after / non_null_before) >= 0.9:
            df[c] = coerced

    # Map case-insensitive 'yes'/'no' to 1/0 for non-numeric columns that only contain those values
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        lowered = df[c].dropna().map(lambda x: x.strip().lower() if isinstance(x, str) else x).unique()
        if set(lowered).issubset({"yes", "no"}):
            df[c] = df[c].map(lambda x: 1 if isinstance(x, str) and x.strip().lower() == "yes" else (0 if isinstance(x, str) and x.strip().lower() == "no" else x))

    # Fill numeric NaNs with median (unchanged behavior)
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

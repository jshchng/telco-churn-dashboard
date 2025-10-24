import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features using the original column names.
    Adds columns:
      - 'tenure_bucket' (from 'Tenure in Months' if present)
      - 'avg_monthly' (Total Charges / Tenure in Months where possible)
      - 'n_services' (counts service-related boolean columns)
      - 'is_echeck' (from 'Payment Method' contains e-check)
      - 'monthly_x_tenure'
    """
    df = df.copy()

    # Defensive: drop columns that are likely to leak the target (e.g., satisfaction surveys)
    leak_candidates = [c for c in df.columns if 'satisfaction' in c.lower()]
    if leak_candidates:
        # remove from the feature pipeline
        df = df.drop(columns=leak_candidates, errors='ignore')

    # Tenure bucket
    if "Tenure in Months" in df.columns:
        try:
            df["tenure_bucket"] = pd.cut(df["Tenure in Months"],
                                         bins=[-1, 12, 24, 48, 1000],
                                         labels=["0-12", "13-24", "25-48", "49+"])
        except Exception:
            df["tenure_bucket"] = "unknown"
    else:
        df["tenure_bucket"] = "unknown"

    # avg_monthly
    if "Total Charges" in df.columns and "Tenure in Months" in df.columns:
        def safe_avg(r):
            try:
                t = float(r["Tenure in Months"])
                tot = float(r["Total Charges"])
                return tot / t if t > 0 else r.get("Monthly Charge", None)
            except Exception:
                return r.get("Monthly Charge", None)
        df["avg_monthly"] = df.apply(safe_avg, axis=1)
    elif "Monthly Charge" in df.columns:
        df["avg_monthly"] = df["Monthly Charge"]
    else:
        df["avg_monthly"] = None

    # Count number of service flags (look for common service words)
    service_candidates = [c for c in df.columns if any(k in c.lower() for k in ("phone", "internet", "online", "stream", "device", "backup", "security", "tech"))]
    if service_candidates:
        # convert to numeric if they are Yes/No or 1/0 already
        df["n_services"] = df[service_candidates].apply(lambda row: pd.to_numeric(row, errors="coerce").fillna(0).astype(int).sum(), axis=1)
    else:
        df["n_services"] = 0

    # is_echeck
    if "Payment Method" in df.columns:
        df["is_echeck"] = df["Payment Method"].astype(str).str.contains("e-check", case=False, na=False).astype(int)
    else:
        df["is_echeck"] = 0

    # monthly_x_tenure
    if "Monthly Charge" in df.columns and "Tenure in Months" in df.columns:
        df["monthly_x_tenure"] = df["Monthly Charge"] * df["Tenure in Months"]
    else:
        df["monthly_x_tenure"] = None

    return df

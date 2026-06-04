import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/loan_default_data.csv"
OUT = "data/processed/loan_default_processed.csv"

# 1) load raw
df = pd.read_csv(RAW)

# 2) preprocess (drops id, fixes TotalCharges, etc.)
df = preprocess_data(df, target_col="Default")

# 3) ensure target is 0/1 only if still object
if "Default" in df.columns and df["Default"].dtype == "object":
    df["Default"] = df["Default"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

# sanity checks
assert df["Default"].isna().sum() == 0, "Default has NaNs after preprocess"
assert set(df["Default"].unique()) <= {0, 1}, "Default not 0/1 after preprocess"

# 4) features
df_processed = build_features(df, target_col="Default")

# 5) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")

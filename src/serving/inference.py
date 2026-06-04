"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================
This module handles the core inference logic for the Loan Default application.
"""

import os
import glob
import pandas as pd
import mlflow

# === LOCAL IMPORTS ===
from src.features.build_features import build_features

# === MODEL LOADING CONFIGURATION ===
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"⚠️ Production container path not found, scanning local workspace...")
    try:
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from local run: {latest_model}")
        else:
            raise Exception("No model artifacts found in your local mlruns directory.")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA ALIGNMENT LOADING ===
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if not os.path.exists(feature_file):
        feature_file = "./artifacts/feature_columns.txt"
        
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns for structural model alignment")
except Exception as e:
    raise Exception(f"Failed to load feature alignment column array: {e}")


def predict(input_dict: dict) -> str:
    """
    Main prediction pipeline converting raw application dictionaries to risk statements.
    """
    # === STEP 1: Dict to DataFrame Conversion ===
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Process using training transformations ===
    df_transformed = build_features(df)
    
    # 💡 FIX: Force conversion of categorical/binary string objects to numeric categories for LightGBM
    # This prevents the single-row "pandas dtypes must be int, float or bool" crash
    categorical_mappings = {
        "Education": {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3},
        "EmploymentType": {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3},
        "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
        "LoanPurpose": {"Home": 0, "Auto": 1, "Business": 2, "Education": 3, "Other": 4},
        "HasMortgage": {"No": 0, "Yes": 1, "no": 0, "yes": 1},
        "HasDependents": {"No": 0, "Yes": 1, "no": 0, "yes": 1},
        "HasCoSigner": {"No": 0, "Yes": 1, "no": 0, "yes": 1}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_transformed.columns:
            # If the value is still a text string, map it to its corresponding number
            if df_transformed[col].dtype == 'object':
                df_transformed[col] = df_transformed[col].map(mapping).fillna(0).astype('int32')
            # If it's a boolean, make sure it behaves numeric
            elif df_transformed[col].dtype == 'bool':
                df_transformed[col] = df_transformed[col].astype('int32')

    # === STEP 3: Structural Schema Reindexing ===
    df_aligned = df_transformed.reindex(columns=FEATURE_COLS, fill_value=0)
    
    # 💡 DOUBLE-CHECK TYPE SAFETY FOR LIGHTGBM: Ensure nothing slipped through as object dtype
    for col in df_aligned.columns:
        if df_aligned[col].dtype == 'object':
            df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce').fillna(0).astype('float32')

    # === STEP 4: Generate Model Prediction ===
    try:
        preds = model.predict(df_aligned)
        
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"LightGBM Backend inference runner failed: {e}")
    
    # === STEP 5: Business Translation ===
    # Handles both regression probabilities (> 0.5) and binary classifications (== 1)
    if (isinstance(result, float) and result >= 0.5) or result == 1:
        return "⚠️ HIGH RISK: High probability of Loan Default"
    else:
        return "🟢 LOW RISK: Approved / Low Default Probability"
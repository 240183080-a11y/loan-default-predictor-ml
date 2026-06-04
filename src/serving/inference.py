"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module handles the core inference logic for the Loan Default application.
It imports your training-time feature engineering module directly to guarantee
absolute feature consistency and eliminate training/serving feature skew.
"""

import os
import glob
import pandas as pd
import mlflow

# === LOCAL IMPORTS ===
# Import your exact feature transformation logic from training
from src.features.build_features import build_features

# === MODEL LOADING CONFIGURATION ===
# /app/model is standard for the containerized Docker build.
# For local Windows development, we dynamically resolve the latest MLflow run.
MODEL_DIR = "/app/model"

try:
    # Load the trained LightGBM model in MLflow pyfunc format
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"⚠️ Production container path not found, scanning local workspace...")
    try:
        # Resolve to your project's local mlruns repository
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
# Load the exact column arrangement array saved by your run_pipeline.py run
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if not os.path.exists(feature_file):
        # Fallback tracking lookup for local dev workspace alignment
        feature_file = "./artifacts/feature_columns.txt"
        
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns for structural model alignment")
except Exception as e:
    raise Exception(f"Failed to load feature alignment column array: {e}")


def predict(input_dict: dict) -> str:
    """
    Main prediction pipeline converting raw application dictionaries to risk statements.

    Pipeline:
    1. Parse user input dict into a single-row Pandas DataFrame
    2. Process data using training-time build_features module
    3. Structural schema alignment (reindex features to match model brain)
    4. Model prediction evaluation
    """
    # === STEP 1: Dict to DataFrame Conversion ===
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Process using training transformations ===
    # Mutates and scales strings, booleans, and floats using your custom logic
    df_transformed = build_features(df)
    
    # === STEP 3: Structural Schema Reindexing ===
    # CRITICAL: Ensures feature columns match the exact sequence array from training.
    # New categories or missing flags are zero-filled automatically.
    df_aligned = df_transformed.reindex(columns=FEATURE_COLS, fill_value=0)
    
    # === STEP 4: Generate Model Prediction ===
    try:
        # Run inference against your trained LightGBM model
        preds = model.predict(df_aligned)
        
        # Format normalization
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
            
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"LightGBM Backend inference runner failed: {e}")
    
    # === STEP 5: Business Translation ===
    if result == 1:
        return "⚠️ HIGH RISK: High probability of Loan Default"
    else:
        return "🟢 LOW RISK: Approved / Low Default Probability"
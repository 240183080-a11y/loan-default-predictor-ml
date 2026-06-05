#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → LightGBM training
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.lightgbm  # 1. CHANGED: Import LightGBM logger instead of XGBoost/Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

# === Import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.data.load_data import load_data                    
from src.data.preprocess import preprocess_data            
from src.features.build_features import build_features     
from src.utils.validate_data import validate_loan_data     # 3. CHANGED: Custom verification for loan data

def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Configure MLflow path cleanly to avoid character escape errors
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    from pathlib import Path
    clean_uri = Path(project_root).joinpath("mlruns").as_uri()
    
    mlruns_path = args.mlflow_uri or clean_uri
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  

    # Start MLflow run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        mlflow.log_param("model", "lightgbm")           # 5. CHANGED: Identify as lightgbm
        mlflow.log_param("threshold", args.threshold)   
        mlflow.log_param("test_size", args.test_size)   

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # === Data Quality Validation ===
        print("🔍 Validating loan dataset quality...")
        is_valid, failed = validate_loan_data(df)       # 6. CHANGED: Call your custom loan validator
        mlflow.log_metric("data_quality_pass", int(is_valid))  

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Data quality check failed. Issues: {failed}")
        else:
            print("✅ Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("🔧 Preprocessing data...")
        df = preprocess_data(df)  

        processed_path = os.path.join(project_root, "data", "processed", "loan_default_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✅ Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering ===
        print("🛠️  Building features...")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        
        df_enc = build_features(df)  
        
        
        print(f"✅ Feature engineering completed: {df_enc.shape[1]} features")

        # === Save Feature Metadata ===
        import json, joblib
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)
        
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols,  
            "target": target                  
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"✅ Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("📊 Splitting data...")
        X = df_enc.drop(columns=[target])  
        y = df_enc[target]                 
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size,    
            stratify=y,                  
            random_state=42              
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        # === Handle Class Imbalance ===
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training (LightGBM Tuning) ===
        print("🤖 Training LightGBM model...")
        
        # 9. CHANGED: Using native optimized LightGBM hyperparameter structure
        model = LGBMClassifier(
            n_estimators=300,        
            learning_rate=0.03,       
            max_depth=7,            
            num_leaves=64,           # LightGBM parameter (2^max_depth is a good baseline)
            subsample=0.9,         
            colsample_bytree=0.9,  
            n_jobs=-1,              
            random_state=42,        
            scale_pos_weight=scale_pos_weight,
            verbosity=-1             # Disables noisy warning outputs
        )

        # Train Model and Track Training Time
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("📊 Evaluating model performance...")
        
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of default (class 1)
        
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  

        # Log Evaluation Metrics to MLflow
        precision = precision_score(y_test, y_pred)    
        recall = recall_score(y_test, y_pred)          
        f1 = f1_score(y_test, y_pred)                  
        roc_auc = roc_auc_score(y_test, proba)         
        
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        print(f"🎯 Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        mlflow.lightgbm.log_model(
            model, 
            artifact_path="model"  
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run loan default pipeline with LightGBM + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/loan_data.csv)")
    p.add_argument("--target", type=str, default="Default")
    p.add_argument("--threshold", type=float, default=0.40) # Slightly higher default threshold for risk
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Loan Default")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)
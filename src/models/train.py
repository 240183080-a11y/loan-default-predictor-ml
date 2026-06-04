import os
from pathlib import Path
import mlflow
import mlflow.lightgbm  # 1. CHANGED: Import LightGBM logger instead of XGBoost
import pandas as pd
from lightgbm import LGBMClassifier  # 2. CHANGED: Import LightGBM instead of XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains a LightGBM model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column (e.g., 'Default').
    """
    # 3. CRITICAL WINDOWS FIX: Add environment bypass and point to the project root mlruns
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    project_root = Path(__file__).resolve().parents[2] # Adjust if file depth changes
    mlflow.set_tracking_uri(project_root.joinpath("mlruns").as_uri())
    mlflow.set_experiment("Loan Default - LGBM")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. CHANGED: Swap model initialization to LightGBM
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=-1  # Keeps your terminal clean of warnings
    )

    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log params, metrics, and model
        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        
        # 5. CHANGED: Use mlflow.lightgbm to save the model brain
        mlflow.lightgbm.log_model(model, "model")

        # 🔑 Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
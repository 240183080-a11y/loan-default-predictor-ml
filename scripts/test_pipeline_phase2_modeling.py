import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import optuna

print("=== Phase 2: Modeling with LightGBM ===")

df = pd.read_csv("data/processed/loan_default_processed.csv")

# target must be numeric 0/1
if df["Default"].dtype == "object":
    df["Default"] = df["Default"].str.strip().map({"No": 0, "Yes": 1})

assert df["Default"].isna().sum() == 0, "Default has NaNs"
assert set(df["Default"].unique()) <= {0, 1}, "Default not 0/1"

X = df.drop(columns=["Default"])
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
THRESHOLD = 0.3


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "is_unbalance": True,
        "objective": "binary",
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_metric="logloss")
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return recall_score(y_test, y_pred, pos_label=1)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall:", study.best_value)

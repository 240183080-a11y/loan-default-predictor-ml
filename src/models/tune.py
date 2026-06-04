import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def tune_model(X, y):
    """
    Tunes a LightGBM model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),  # Equivalent to XGBoost's gamma
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "is_unbalance": True,  # Equivalent to scale_pos_weight for handling imbalanced data
        "objective": "binary",
        "verbose": -1
    }
    
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("Best Params:", study.best_params)
    return study.best_params
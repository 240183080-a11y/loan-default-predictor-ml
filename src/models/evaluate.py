from sklearn.metrics import classification_report, confusion_matrix

THRESHOLD = 0.3  # tuned in EDA (below 0.5 to improve recall on defaults)


def evaluate_model(model, X_test, y_test, threshold: float = THRESHOLD):
    """
    Evaluates a LightGBM model on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn import metrics


def evaluate_metrics(model, X_test, y_test):
    """
    Calculates main model metrics

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels

    Returns:
        dict: Dictionary with calculated metrics
    """
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1w = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize="true").tolist()
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_f1": float(f1w),
        "confusion_matrix": cm,
    }


def collect_performance_over_time(automl):
    """
    Collects performance data over time

    Args:
        automl: Trained AutoSklearn model

    Returns:
        dict: Temporal performance data
    """
    df = automl.performance_over_time_.copy()
    cols = [c for c in df.columns if c not in ("num_models_trained",)]
    result = {}
    for c in cols:
        if c == "Timestamp":
            # Convert timestamps to ISO string
            result[c] = [ts.isoformat() for ts in df[c]]
        else:
            result[c] = df[c].tolist()
    return result

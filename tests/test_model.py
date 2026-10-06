import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.train import evaluate_model, train_logistic_regression


def test_train_and_evaluate_accuracy_baseline():
    X_train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 10.0, 11.0],
            "f2": [0.1, 0.2, 0.9, 1.0],
        }
    )

    y_train = pd.Series([0, 0, 1, 1])

    X_test = pd.DataFrame(
        {
            "f1": [1.5, 10.5],
            "f2": [0.15, 0.95],
        }
    )

    y_test = pd.Series([0, 1])

    model = train_logistic_regression(X_train, y_train)

    assert isinstance(model, LogisticRegression)

    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] >= 0.5

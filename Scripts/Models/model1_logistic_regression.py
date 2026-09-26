"""
CAP182 SS2 Part B
Model 1 – Logistic Regression
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")

RANDOM_STATE = 42


def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_lr.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_lr.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["returned"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["returned"]

    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        "Model: Logistic Regression",
        "",
        "Configuration:",
        "C=1.0",
        "penalty=l2",
        "solver=liblinear",
        "max_iter=1000",
        "random_state=42",
        "",
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}",
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}",
        f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}",
        f"F1: {f1_score(y_test, predictions, zero_division=0):.4f}",
        f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}",
        "",
        "Confusion matrix:",
        str(confusion_matrix(y_test, predictions)),
        "",
        "Classification report:",
        classification_report(y_test, predictions, zero_division=0),]

    output_path = RESULTS_DIR / "logistic_regression_metrics.txt"
    output_path.write_text("\n".join(metrics), encoding="utf-8")

    joblib.dump(
        model,
        RESULTS_DIR / "logistic_regression_model.joblib")

    print("\n".join(metrics))
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()

"""
CAP182 SS2 Part B
Model 2 – Random Forest Classification
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_rf.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_rf.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["returned"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["returned"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        "Model: Random Forest Classification",
        "",
        "Configuration:",
        "n_estimators=300",
        "max_depth=15",
        "min_samples_split=5",
        "min_samples_leaf=2",
        "max_features=sqrt",
        "class_weight=balanced",
        "random_state=42",
        "n_jobs=-1",
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

    output_path = RESULTS_DIR / "random_forest_metrics.txt"
    output_path.write_text("\n".join(metrics), encoding="utf-8")

    joblib.dump(
        model,
        RESULTS_DIR / "random_forest_model.joblib")

    print("\n".join(metrics))
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()

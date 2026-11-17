
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.data_pipeline import (
    load_data,
    prepare_features,
    split_data
)


def main():

    df = load_data()
    X, y = prepare_features(df)

    (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test
    ) = split_data(X, y)

    model = joblib.load(
        "artifacts/churn_model.joblib"
    )

    with open("artifacts/metadata.json") as file:
        metadata = json.load(file)

    threshold = metadata["threshold"]

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(
            y_test, probabilities
        )
    }

    print("Final test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    pd.DataFrame([metrics]).to_csv(
        "outputs/test_metrics.csv",
        index=False
    )

    print("\nTest metrics saved!")


if __name__ == "__main__":
    main()

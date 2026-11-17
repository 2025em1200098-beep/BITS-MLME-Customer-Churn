
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from src.data_pipeline import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor
)


def main():

    # 1. Load data
    df = load_data()
    X, y = prepare_features(df)

    # 2. Split data
    (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test
    ) = split_data(X, y)

    print("Training records:", len(X_train))
    print("Validation records:", len(X_valid))
    print("Test records:", len(X_test))

    # 3. Baseline
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_valid)

    print(
        "Baseline F1:",
        f1_score(y_valid, baseline_predictions)
    )

    # 4. Build models
    model_configs = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    # 5. Train and compare
    for name, classifier in model_configs.items():

        preprocessor = build_preprocessor(X_train)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)
        probabilities = model.predict_proba(X_valid)[:, 1]

        metrics = {
            "Model": name,
            "Precision": precision_score(
                y_valid, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_valid, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_valid, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y_valid, probabilities
            ),
            "PR_AUC": average_precision_score(
                y_valid, probabilities
            )
        }

        results.append(metrics)
        trained_models[name] = model

    results_df = pd.DataFrame(results)

    print("\nValidation comparison:")
    print(results_df.to_string(index=False))

    # 6. Select model based on validation PR-AUC
    best_name = results_df.sort_values(
        "PR_AUC",
        ascending=False
    ).iloc[0]["Model"]

    best_model = trained_models[best_name]

    # 7. Select threshold using validation F1
    valid_probabilities = best_model.predict_proba(
        X_valid
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_valid,
        valid_probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    # 8. Save artifacts
    Path("artifacts").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    joblib.dump(
        best_model,
        "artifacts/churn_model.joblib"
    )

    metadata = {
        "model_name": best_name,
        "threshold": best_threshold,
        "features": X.columns.tolist()
    }

    with open("artifacts/metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    results_df.to_csv(
        "outputs/validation_results.csv",
        index=False
    )

    print("\nSelected model:", best_name)
    print("Selected threshold:", best_threshold)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()

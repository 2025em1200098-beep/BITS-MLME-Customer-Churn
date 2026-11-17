
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def prepare_input(input_df, required_features):

    # Remove identifier and target if supplied.
    input_df = input_df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    missing_columns = [
        col for col in required_features
        if col not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only the features expected by the model.
    input_df = input_df[required_features].copy()

    if "TotalCharges" in input_df.columns:
        input_df["TotalCharges"] = pd.to_numeric(
            input_df["TotalCharges"],
            errors="coerce"
        )

    return input_df


def predict_batch(input_df, model, threshold, required_features):

    clean_input = prepare_input(
        input_df,
        required_features
    )

    probabilities = model.predict_proba(
        clean_input
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = input_df.copy()

    results["churn_probability"] = probabilities
    results["churn_prediction"] = predictions
    results["churn_label"] = np.where(
        predictions == 1,
        "Churn",
        "No Churn"
    )

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Batch customer churn prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input customer CSV file"
    )

    parser.add_argument(
        "--output",
        default="outputs/churn_predictions.csv",
        help="Output prediction CSV file"
    )

    args = parser.parse_args()

    model = joblib.load(
        "artifacts/churn_model.joblib"
    )

    with open("artifacts/metadata.json") as file:
        metadata = json.load(file)

    input_df = pd.read_csv(args.input)

    results = predict_batch(
        input_df,
        model,
        metadata["threshold"],
        metadata["features"]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(output_path, index=False)

    print("Predictions saved to:", output_path)
    print("Customers processed:", len(results))


if __name__ == "__main__":
    main()

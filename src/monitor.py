
import argparse
import pandas as pd
import numpy as np


def calculate_psi(reference, current, bins=10):

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Reference or current data is empty.")

    if reference.nunique() <= bins:

        categories = sorted(reference.unique())

        ref_pct = np.array([
            (reference == value).mean()
            for value in categories
        ])

        cur_pct = np.array([
            (current == value).mean()
            for value in categories
        ])

    else:

        edges = np.unique(
            np.quantile(
                reference,
                np.linspace(0, 1, bins + 1)
            )
        )

        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(
            reference, bins=edges
        )[0]

        cur_counts = np.histogram(
            current, bins=edges
        )[0]

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

    epsilon = 1e-6

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    ))


def main():

    parser = argparse.ArgumentParser(
        description="Monitor numerical feature drift"
    )

    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)

    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)

    for frame in (reference, current):
        frame.drop(
            columns=["customerID", "Churn"],
            errors="ignore",
            inplace=True
        )

        if "TotalCharges" in frame.columns:
            frame["TotalCharges"] = pd.to_numeric(
                frame["TotalCharges"],
                errors="coerce"
            )

    numeric_features = reference.select_dtypes(
        include=["number"]
    ).columns

    results = []

    for feature in numeric_features:

        if feature not in current.columns:
            print("Skipping missing feature:", feature)
            continue

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        if psi < 0.10:
            status = "Low drift"
        elif psi < 0.25:
            status = "Moderate drift"
        else:
            status = "High drift"

        results.append({
            "Feature": feature,
            "PSI": psi,
            "Status": status
        })

    result_df = pd.DataFrame(results)

    print(result_df.to_string(index=False))

    result_df.to_csv(
        "outputs/drift_report.csv",
        index=False
    )

    print("\nDrift report saved.")


if __name__ == "__main__":
    main()

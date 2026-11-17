
from pathlib import Path

import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)


def load_data(data_path="data/raw/Telco-Customer-Churn.csv"):
    """Download the dataset if it is not already present."""

    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)

    df = pd.read_csv(path)

    # Convert blank TotalCharges entries into missing values.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


def prepare_features(df):
    """Separate input features from the churn target."""

    model_df = df.copy()

    model_df["TotalCharges"] = pd.to_numeric(
        model_df["TotalCharges"],
        errors="coerce"
    )

    X = model_df.drop(
        columns=["Churn", "customerID"]
    )

    y = model_df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    if y.isnull().any():
        raise ValueError("Target contains unknown or missing values.")

    return X, y.astype(int)


def split_data(X, y):
    """Create stratified 70/15/15 data splits."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test
    )


def build_preprocessor(X_train):
    """Build preprocessing using training columns."""

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

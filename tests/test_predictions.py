
import pandas as pd
import pytest

from src.monitor import calculate_psi
from src.predict_batch import prepare_input


def test_psi_same_distribution():

    reference = pd.Series([1, 2, 3, 4, 5, 6])
    current = pd.Series([1, 2, 3, 4, 5, 6])

    psi = calculate_psi(reference, current)

    assert psi < 0.0001


def test_psi_detects_distribution_change():

    reference = pd.Series([1, 2, 3, 4, 5, 6])
    current = pd.Series([20, 21, 22, 23, 24, 25])

    psi = calculate_psi(reference, current)

    assert psi > 0.10


def test_missing_feature_raises_error():

    input_df = pd.DataFrame({
        "tenure": [10],
        "MonthlyCharges": [50]
    })

    required_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    with pytest.raises(ValueError):
        prepare_input(input_df, required_features)

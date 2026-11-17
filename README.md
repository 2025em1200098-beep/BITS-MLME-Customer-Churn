
# Telecom Customer Churn Prediction

## 1. Project Overview

This project develops a mini production machine-learning system
to predict whether a telecom customer is likely to leave the
company.

The project uses supervised binary classification.

## 2. Business Objective

The objective is to identify customers who are likely to churn
so that the telecom company can take proactive retention
measures.

## 3. Dataset

The project uses the IBM Telco Customer Churn dataset.

The target variable is `Churn`:
- 0 = Customer does not churn
- 1 = Customer churns

CustomerID is excluded from model training.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Pytest

## 5. Models

The project compares:
- Dummy Classifier (baseline)
- Logistic Regression
- Random Forest

## 6. Evaluation Metrics

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## 7. Project Structure

- data/: dataset storage
- artifacts/: saved model and metadata
- outputs/: evaluation and prediction results
- notebooks/: exploratory analysis
- src/: reusable Python source code
- tests/: automated tests

## 8. How to Run

Install dependencies:

    pip install -r requirements.txt

Train the models:

    python -m src.train

Evaluate the selected model:

    python -m src.evaluate

Run batch predictions:

    python -m src.predict_batch --input data/raw/Telco-Customer-Churn.csv

Run automated tests:

    pytest -q

Run drift monitoring:

    python -m src.monitor --reference PATH_TO_REFERENCE.csv --current PATH_TO_CURRENT.csv

The dataset is downloaded automatically when training
if it is not already present.

## 9. Reproducibility

The project uses a fixed random seed of 42 for data splitting
and model training.

Preprocessing is fitted as part of the machine-learning
pipeline using training data only.

## 10. Limitations

The model is trained on a historical sample dataset.
Its predictions should be validated on current customer data
before being used for real business decisions.

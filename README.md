# Credit Scoring Model

## Overview
A machine-learning project that predicts good-credit and bad-credit
classes using the UCI German Credit dataset.

## Objectives
- Predict creditworthiness.
- Engineer features from financial history.
- Compare Logistic Regression, Random Forest, and Decision Tree.
- Evaluate accuracy, precision, recall, F1-score, and ROC-AUC.

## Feature Engineering
Five additional features:
- Estimated monthly repayment
- Credit amount relative to age
- Credit amount relative to existing credits
- High-credit-amount indicator
- Short-employment-history indicator

The repayment estimate excludes interest and fees. The dataset does
not include direct income data.

## Model Results

| Model | Accuracy | Bad-credit F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.770 | 0.681 | 0.821 |
| Random Forest | 0.750 | 0.597 | 0.789 |
| Decision Tree | 0.695 | 0.504 | 0.644 |

Logistic Regression performed best by bad-credit F1-score in this test.

## How to Run
Activate your virtual environment and run:

    python prepare_data.py
    python compare_models.py

## Outputs
The outputs/ directory contains trained models, evaluation results,
and confusion-matrix graphics.

## Limitations
These results are a baseline, not proof of suitability for real lending.
The dataset is limited, and the engineered ratios do not directly measure
a borrower's affordability.

## Technologies
Python, pandas, NumPy, scikit-learn, Matplotlib, and joblib.

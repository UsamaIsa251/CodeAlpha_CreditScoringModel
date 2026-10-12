import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create financial features without learning from the test set."""

    def fit(self, X, y=None):
        self.credit_amount_median_ = X["credit_amount"].median()
        return self

    def transform(self, X):
        X = X.copy()

        X["monthly_repayment_estimate"] = (
            X["credit_amount"] / X["duration"].replace(0, np.nan)
        )

        X["credit_amount_per_year_of_age"] = (
            X["credit_amount"] / X["age"].replace(0, np.nan)
        )

        X["credit_amount_per_existing_credit"] = (
            X["credit_amount"]
            / X["existing_credits"].replace(0, np.nan)
        )

        X["high_credit_amount"] = (
            X["credit_amount"] > self.credit_amount_median_
        ).astype(int)

        X["short_employment_history"] = (
            X["employment_since"].isin(["A71", "A72"])
        ).astype(int)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X

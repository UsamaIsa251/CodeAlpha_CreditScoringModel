import pandas as pd
import numpy as np

INPUT_FILE = "credit_data.csv"
OUTPUT_FILE = "credit_data_engineered.csv"

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "credit_amount",
    "duration",
    "age",
    "existing_credits",
    "employment_since",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 1. Estimated monthly principal repayment.
# This excludes interest, fees, and other financial obligations.
df["monthly_repayment_estimate"] = (
    df["credit_amount"] / df["duration"].replace(0, np.nan)
)

# 2. Loan amount relative to age.
df["credit_amount_per_year_of_age"] = (
    df["credit_amount"] / df["age"].replace(0, np.nan)
)

# 3. Loan amount relative to the number of existing credits.
df["credit_amount_per_existing_credit"] = (
    df["credit_amount"]
    / df["existing_credits"].replace(0, np.nan)
)

# 4. Flag relatively large loans using the dataset median.
# In proper evaluation, calculate this threshold on training data only.
amount_threshold = df["credit_amount"].median()
df["high_credit_amount"] = (
    df["credit_amount"] > amount_threshold
).astype(int)

# 5. Employment duration categories in the UCI dataset:
# A71 = unemployed, A72 = less than one year,
# A73 = 1–4 years, A74 = 4–7 years, A75 = 7 or more years.
# Flag unemployed or less-than-one-year employment.
df["short_employment_history"] = (
    df["employment_since"].isin(["A71", "A72"])
).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering completed.")
print(f"Original columns: {len(pd.read_csv(INPUT_FILE).columns)}")
print(f"Engineered columns: {len(df.columns)}")
print(f"Saved dataset: {OUTPUT_FILE}")
print("\nNew features:")
print(df[[
    "monthly_repayment_estimate",
    "credit_amount_per_year_of_age",
    "credit_amount_per_existing_credit",
    "high_credit_amount",
    "short_employment_history",
]].head())

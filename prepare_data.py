import pandas as pd

columns = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings_account",
    "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "credit_risk"
]

df = pd.read_csv(
    "german.data",
    sep=r"\s+",
    header=None,
    names=columns
)

# Original UCI labels: 1 = good credit, 2 = bad credit.
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

df.to_csv("credit_data.csv", index=False)

print("Dataset conversion successful.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nCredit risk distribution (0 = good, 1 = bad):")
print(df["credit_risk"].value_counts().sort_index())
print("\nFirst five records:")
print(df.head())

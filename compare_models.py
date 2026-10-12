
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from financial_features import FinancialFeatureEngineer


# 1. Load the original dataset.
# Feature engineering will be performed inside each model pipeline.
df = pd.read_csv("credit_data.csv")

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"].astype(int)

# 2. Create a stratified train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Define original and engineered columns.
numeric_features = [
    "duration",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "dependents",
    "monthly_repayment_estimate",
    "credit_amount_per_year_of_age",
    "credit_amount_per_existing_credit",
    "high_credit_amount",
    "short_employment_history",
]

categorical_features = [
    column for column in X.columns
    if column not in numeric_features
    and column not in [
        "monthly_repayment_estimate",
        "credit_amount_per_year_of_age",
        "credit_amount_per_existing_credit",
        "high_credit_amount",
        "short_employment_history",
    ]
]

# 4. Preprocess numeric and categorical features.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Compare three classification algorithms.
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
    ),
}

os.makedirs("outputs", exist_ok=True)

results = []
fitted_models = {}

for name, classifier in models.items():
    model = Pipeline([
        ("feature_engineering", FinancialFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Bad Credit Precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "Bad Credit Recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "Bad Credit F1": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)
    fitted_models[name] = model

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print(classification_report(
        y_test,
        predictions,
        target_names=["Good Credit", "Bad Credit"],
        zero_division=0,
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Good Credit", "Bad Credit"],
        cmap="Blues",
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.close()

# 6. Save comparison results.
results_df = pd.DataFrame(results).sort_values(
    by="Bad Credit F1", ascending=False
)
results_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

# 7. Save the best model based on bad-credit F1.
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

joblib.dump(best_model, "outputs/best_credit_scoring_model.joblib")

print(f"\nBest model by bad-credit F1-score: {best_name}")
print("Saved model: outputs/best_credit_scoring_model.joblib")
print("Results saved in outputs/model_comparison.csv")
print("Confusion matrices saved in outputs/")

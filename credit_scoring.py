
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

# 1. Project configuration
DATA_FILE = "credit_data.csv"
OUTPUT_DIR = "outputs"
MODEL_FILE = os.path.join(OUTPUT_DIR, "credit_scoring_model.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load and validate data
df = pd.read_csv(DATA_FILE)

if "credit_risk" not in df.columns:
    raise ValueError("The dataset must contain a credit_risk column.")

if df["credit_risk"].isna().any():
    raise ValueError("The target column contains missing values.")

X = df.drop(columns=["credit_risk"])
y = df["credit_risk"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected target labels 0 (good) and 1 (bad).")

print("=" * 55)
print("CREDIT SCORING MODEL")
print("=" * 55)
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print("\nTarget distribution (0 = good, 1 = bad):")
print(y.value_counts().sort_index())

# 3. Identify feature types
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

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

# 4. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL EVALUATION")
print("-" * 55)
print(f"Test accuracy: {accuracy:.3f}")
print(f"ROC-AUC:       {roc_auc:.3f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Good credit", "Bad credit"],
    zero_division=0,
))

# 7. Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Good credit", "Bad credit"],
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

ax.set_title("Credit Scoring - Confusion Matrix")
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=150,
)
plt.close(fig)

# 8. Plot the ROC curve
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=150,
)
plt.close(fig)

# 9. Plot the class distribution
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(
    x=y.map({0: "Good credit", 1: "Bad credit"}),
    hue=y.map({0: "Good credit", 1: "Bad credit"}),
    order=["Good credit", "Bad credit"],
    hue_order=["Good credit", "Bad credit"],
    legend=False,
    ax=ax,
)
ax.set_title("Credit Risk Class Distribution")
ax.set_xlabel("Credit risk category")
ax.set_ylabel("Number of records")
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "class_distribution.png"),
    dpi=150,
)
plt.close(fig)

# 10. Save the trained pipeline
joblib.dump(model, MODEL_FILE)

print("\nFILES SAVED")
print(f"Model: {MODEL_FILE}")
print(f"Charts: {OUTPUT_DIR}/")
print("Credit scoring model completed successfully.")



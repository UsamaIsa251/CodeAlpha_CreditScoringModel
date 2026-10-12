import pandas as pd
import matplotlib.pyplot as plt
import os

results = pd.read_csv("outputs/model_comparison.csv")

metrics = [
    "Accuracy",
    "Bad Credit Precision",
    "Bad Credit Recall",
    "Bad Credit F1",
    "ROC-AUC",
]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    width=0.8,
)

ax.set_title("Credit Scoring Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Classification Model")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.xticks(rotation=0)
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/model_comparison.png", dpi=300)
plt.close()

print("Updated outputs/model_comparison.png")
print(results.to_string(index=False))

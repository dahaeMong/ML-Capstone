import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# -------------------------------
# 1. Load data & models
# -------------------------------
DATA_PATH = "data/match_sample_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

feature_cols = [
    col for col in df.columns
    if col not in ["match_id", "player", "map", "agent", "win"]
]

X = df[feature_cols]
y = df["win"]

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}

# Output folder
os.makedirs("results", exist_ok=True)


# -------------------------------
# 2. Evaluation helper functions
# -------------------------------
def get_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y, pred)

    fpr, tpr, _ = roc_curve(y, prob) if prob is not None else (None, None, None)
    auc_score = auc(fpr, tpr) if prob is not None else None

    return cm, fpr, tpr, auc_score


# -------------------------------
# 3. Evaluate each model
# -------------------------------
results = []

for name, model in models.items():
    print(f"▶ Evaluating {name}...")

    cm, fpr, tpr, auc_score = get_metrics(model, X, y)

    # Save confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # Save ROC curve
    if auc_score is not None:
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{name} – ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{name.replace(' ', '_')}_roc.png")
        plt.close()

    # Save metrics
    results.append({
        "model": name,
        "auc": auc_score
    })

    # Feature importance (Tree models)
    if name in ["Random Forest", "XGBoost"]:
        importance = model.feature_importances_
        fi = pd.DataFrame({"feature": feature_cols, "importance": importance})
        fi = fi.sort_values(by="importance", ascending=False)

        plt.figure(figsize=(5, 4))
        sns.barplot(data=fi.head(10), x="importance", y="feature", palette="Greens_d")
        plt.title(f"{name} – Top 10 Feature Importance")
        plt.tight_layout()
        plt.savefig(f"results/{name.replace(' ', '_')}_feature_importance.png")
        plt.close()

# -------------------------------
# 4. Save metrics comparison table
# -------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_metrics.csv", index=False)

print("\n Evaluation completed!")
print("Results saved in /results folder")

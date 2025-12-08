import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -------------------------------
# 1. Load preprocessed data
# -------------------------------
DATA_PATH = "data/match_sample_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# 2. Features & target
# -------------------------------
feature_cols = [
    col for col in df.columns
    if col not in ["match_id", "player", "map", "agent", "win"]
]

X = df[feature_cols]
y = df["win"]

# -------------------------------
# 3. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Utility: Training + evaluation
# -------------------------------
def evaluate(model, X_test, y_test):
    """Return accuracy, F1, AUC"""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return acc, f1, auc


# -------------------------------
# 4. Train models
# -------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

trained_models = {}
metrics = {}

print("\n====================")
print("   Training Models")
print("====================\n")

for name, model in models.items():
    print(f"▶ Training {name}...")
    model.fit(X_train, y_train)

    acc, f1, auc = evaluate(model, X_test, y_test)
    metrics[name] = (acc, f1, auc)

    trained_models[name] = model
    print(f"   - Accuracy: {acc:.3f}")
    print(f"   - F1 Score: {f1:.3f}")
    if auc:
        print(f"   - AUC: {auc:.3f}")
    print()


# -------------------------------
# 5. Save models
# -------------------------------
os.makedirs("models", exist_ok=True)

for name, model in trained_models.items():
    save_path = f"models/{name}.pkl"
    joblib.dump(model, save_path)
    print(f" Saved {name} → {save_path}")

print("\n Training completed!")

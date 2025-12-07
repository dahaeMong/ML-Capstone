# ===============================
# XGBoost + Preprocessing Pipeline
# ===============================

import pandas as pd
import numpy as np
from scripts.preprocessing import load_and_preprocess
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
X_train, X_test, y_train, y_test, preprocessor, label_encoder = load_and_preprocess()

# -------------------------------
# 2. Train XGBoost
# -------------------------------
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -------------------------------
# 3. Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)

print(f"✅ XGBoost Accuracy: {acc:.3f}")
print(f"✅ XGBoost AUC: {auc:.3f}")
print(f"✅ XGBoost F1 Score: {f1:.3f}")


# -------------------------------
# 4. Recommendation Function
# -------------------------------
def recommend_agent_xgb(current_map, current_team_agents, df_raw):
    """
    Recommend best agent based on XGBoost + preprocessing.
    """
    all_agents = df_raw["agent"].unique()
    best_agent = None
    best_prob = -1

    for agent in all_agents:

        # -----------------------------
        # Construct 1-row input feature
        # -----------------------------
        row = {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "damage": 0,
            "headshot_pct": 0,
            "KDA": np.log1p(1.0),
            "map": current_map,
            "agent_encoded": label_encoder.transform([agent])[0],
        }

        # teammate binary columns
        for tm in current_team_agents:
            row[tm] = 1

        input_df = pd.DataFrame([row])

        # -----------------------------
        # Apply preprocessing (same as train)
        # -----------------------------
        input_processed = preprocessor.transform(input_df)

        # -----------------------------
        # Predict win probability
        # -----------------------------
        prob = model.predict_proba(input_processed)[0][1]

        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob


# -------------------------------
# 5. Test Recommendation
# -------------------------------
df_raw = pd.read_csv("data/match_sample_preprocessed.csv")

best_agent, win_prob = recommend_agent_xgb(
    "Ascent",
    ["Jett", "Sage", "Reyna", "Omen"],
    df_raw
)

print(f"\n🎯 Recommended Agent (XGBoost): {best_agent}, Predicted Win Probability: {win_prob*100:.2f}%")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
df = pd.read_csv("data/match_sample_preprocessed.csv")

# -------------------------------
# 2. Define features & target
# -------------------------------
feature_cols = [col for col in df.columns if col not in ["match_id", "player", "map", "agent", "win"]]
X = df[feature_cols]
y = df["win"]

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. XGBoost Model
# -------------------------------
model = XGBClassifier(
    n_estimators=200,       # Tree number
    learning_rate=0.05,     
    max_depth=5,            
    subsample=0.8,          
    colsample_bytree=0.8,   
    random_state=42,
    eval_metric="logloss"
)

# -------------------------------
# 5. Train model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate performance
# -------------------------------
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)

print(f"✅ XGBoost Accuracy: {acc:.3f}")
print(f"✅ XGBoost AUC: {auc:.3f}")
print(f"✅ XGBoost F1-Score: {f1:.3f}")

# -------------------------------
# 7. Recommendation function
# -------------------------------
def recommend_agent(current_map, current_team_agents, df, model):
    """
    Based on current map and team composition, XGBoost recommand highest win rate agent
    """
    # map info
    all_agents = df["agent"].unique()
    best_agent = None
    best_prob = -1

    for agent in all_agents:
        input_dict = {col: 0 for col in feature_cols}

        # map info
        if "map_encoded" in feature_cols:
            input_dict["map_encoded"] = df.loc[df["map"] == current_map, "map_encoded"].iloc[0]

        # team composition
        for t_agent in current_team_agents:
            if t_agent in feature_cols:
                input_dict[t_agent] = 1

        # current candidate agent
        if agent in feature_cols:
            input_dict[agent] = 1

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[:, 1][0]

        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob


# -------------------------------
# 8. Test recommendation
# -------------------------------
best_agent, win_prob = recommend_agent("Ascent", ["Jett", "Sage", "Reyna", "Omen"], df, model)
print(f"\n🎯 Recommended Agent (XGBoost): {best_agent}, Predicted Win Probability: {win_prob * 100:.2f} %")

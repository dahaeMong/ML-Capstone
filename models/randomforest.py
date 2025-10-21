# -------------------------------
# Randomforest model for recommendation model
# -------------------------------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
# 4. Random Forest training
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# 5. Score accuracy
# -------------------------------
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc:.2f}")

# -------------------------------
# 6. Recommendation function logic
# -------------------------------
def recommend_agent_rf(current_map, current_team_agents):
    """
    Based on current map and team composition
    """
    agents = df["agent"].unique()
    best_agent = None
    best_prob = -1

    for agent in agents:
        # generate input data
        input_dict = {col: 0 for col in feature_cols}

        # map info
        map_encoded = df.loc[df["map"] == current_map, "map_encoded"].iloc[0]
        input_dict["map_encoded"] = map_encoded

        # target agen encoding
        agent_encoded = df.loc[df["agent"] == agent, "agent_encoded"].iloc[0]
        input_dict["agent_encoded"] = agent_encoded

        # iterate through agents list to calculate win rate
        for t_agent in current_team_agents:
            if t_agent in df.columns:
                input_dict[t_agent] = 1

        input_df = pd.DataFrame([input_dict])
        prob = rf_model.predict_proba(input_df)[:, 1][0]  # win rate
        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob

# -------------------------------
# 7. Test run
# -------------------------------
recommended_agent, win_prob = recommend_agent_rf("Ascent", ["Jett", "Sage", "Reyna", "Omen"])
print(f"Recommended Agent: {recommended_agent}, Predicted Win Probability: {win_prob*100:.2f} %")

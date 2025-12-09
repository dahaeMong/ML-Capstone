# recommend.py
import pandas as pd
import joblib

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
DATA_PATH = "data/match_sample_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

# Features 
feature_cols = [col for col in df.columns if col not in ["match_id", "player", "map", "agent", "win"]]

# -------------------------------
# 2. Load trained models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl"),
}

# -------------------------------
# 3. Define recommendation function
# -------------------------------
def recommend_best_agent(current_map, current_team_agents, model, feature_cols, df):
    """
    Return best agent and predicted win probability
    """
    all_agents = df["agent"].unique()
    best_agent = None
    best_prob = -1

    for agent in all_agents:
        input_dict = {col: 0 for col in feature_cols}

        # map encoding
        if "map_encoded" in feature_cols:
            if current_map in df["map"].values:
                input_dict["map_encoded"] = df.loc[df["map"] == current_map, "map_encoded"].iloc[0]
            else:
                input_dict["map_encoded"] = 0  # default if map unknown

        # team composition
        for t_agent in current_team_agents:
            if t_agent in feature_cols:
                input_dict[t_agent] = 1

        # candidate agent
        if agent in feature_cols:
            input_dict[agent] = 1

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[:, 1][0] if hasattr(model, "predict_proba") else model.predict(input_df)[0]

        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob

# -------------------------------
# 4. User input
# -------------------------------
current_map = input("Map? ")
team_input = input("Team composition (comma separated, e.g., Jett,Sage,Reyna,Omen)? ")
current_team_agents = [x.strip() for x in team_input.split(",")]

# -------------------------------
# 5. Show recommendations
# -------------------------------
print("\Recommended Agents:")
for name, model in models.items():
    agent, prob = recommend_best_agent(current_map, current_team_agents, model, feature_cols, df)
    print(f"{name}: {agent} (Predicted Win Probability: {prob*100:.2f}%)")

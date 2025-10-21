# -------------------------------
# Baseline model for recommendation model
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
df = pd.read_csv("data/match_sample_preprocessed.csv")

# -------------------------------
# 2. Define features & target
# -------------------------------
feature_cols = [col for col in df.columns if col not in ["match_id","player","map","agent","agent_encoded","win"]]
X = df[feature_cols]
y = df["win"]

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# -------------------------------
# 4. Logistic Regression training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 5. Score accuracy
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Baseline Logistic Regression Accuracy: {acc:.2f}")

# -------------------------------
# 6. Recommendation function logic
# -------------------------------
def recommend_agent(current_map, current_team_agents, df, model, feature_cols):
    """
    Based on current map and team composition
    """
    all_agents = df["agent"].unique()
    best_agent = None
    best_prob = -1

    for agent in all_agents:
        # generate input data
        input_dict = {col: 0 for col in feature_cols}
        
        # map info
        if "map_encoded" in feature_cols:
            input_dict["map_encoded"] = df.loc[df["map"]==current_map, "map_encoded"].iloc[0]
        
        # Team composition
        for t_agent in current_team_agents:
            if t_agent in feature_cols:
                input_dict[t_agent] = 1

        # iterate through agents list to calculate win rate
        if agent in feature_cols:
            input_dict[agent] = 1

        input_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(input_df)[:, 1][0]  # win rate
        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob

# -------------------------------
# 7. Test run
# -------------------------------
recommended_agent, win_prob = recommend_agent(
    current_map="Ascent",
    current_team_agents=["Jett", "Sage", "Omen", "Yoru"],
    df=df,
    model=model,
    feature_cols=feature_cols
)

print(f"Recommended Agent: {recommended_agent}, Predicted Win Probability: {win_prob*100:.2f} %")
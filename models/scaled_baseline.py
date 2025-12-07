import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
df = pd.read_csv("data/match_sample_preprocessed.csv")

# -------------------------------
# 2. Define features & target
# -------------------------------
feature_cols = [col for col in df.columns if col not in ["match_id","player","map","agent","win"]]
X = df[feature_cols]
y = df["win"]

# -------------------------------
# 3. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Logistic Regression Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# 6. Score accuracy
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Baseline Logistic Regression Accuracy: {acc:.2f}")

# -------------------------------
# 7. Recommendation function logic
# -------------------------------
def recommend_agent(current_map_encoded, current_team_agents_onehot):
    """
    current_map_encoded: map_encoded int
    current_team_agents_onehot: team agent One-Hot vector (DataFrame row)
    """
    best_agent = None
    best_prob = -1

    for agent_col in [c for c in X.columns if c not in ["map_encoded", "agent_encoded"]]:
        input_vec = pd.DataFrame([0]*len(X.columns)).T
        input_vec.columns = X.columns

        # map info
        input_vec["map_encoded"] = current_map_encoded

        # Team composition
        for col in current_team_agents_onehot.index:
            input_vec[col] = current_team_agents_onehot[col]

        # Target agen
        input_vec[agent_col] = 1

        # Feature Scaling
        input_scaled = scaler.transform(input_vec)

        prob = model.predict_proba(input_scaled)[:,1][0]
        if prob > best_prob:
            best_prob = prob
            best_agent = agent_col

    return best_agent, best_prob

# -------------------------------
# 8. Test run
# -------------------------------
team_onehot = pd.Series({col:0 for col in X.columns if col not in ["map_encoded","agent_encoded"]})
for agent in ["Jett","Cypher","Reyna","Omen"]:
    if agent in team_onehot.index:
        team_onehot[agent] = 1

recommended_agent, win_prob = recommend_agent(1, team_onehot)
print(f"Recommended Agent: {recommended_agent}, Predicted Win Probability: {win_prob*100:.2f} %")

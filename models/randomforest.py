# -------------------------------
# RandomForest + Preprocessing
# -------------------------------

import pandas as pd
import numpy as np
from scripts.preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
X_train, X_test, y_train, y_test, preprocessor, label_encoder = load_and_preprocess()

# -------------------------------
# 2. Train Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# 3. Score accuracy
# -------------------------------
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {acc:.2f}")


# -------------------------------
# 4. Recommendation function
# -------------------------------
def recommend_agent_rf(current_map, current_team_agents, df_raw):
    """
    Recommend agent using preprocessed model pipeline
    """
    agents = df_raw["agent"].unique()

    best_agent = None
    best_prob = -1

    for agent in agents:
        # ---------------------------
        # Build single-row input dict
        # ---------------------------
        row = {
            "kills": 0,
            "deaths": 0,
            "assists": 0,
            "damage": 0,
            "headshot_pct": 0,
            "KDA": np.log1p(1.0),  # KDA baseline
            "map": current_map,
            "agent_encoded": label_encoder.transform([agent])[0],
        }

        # teammates (binary columns)
        for teammate in current_team_agents:
            if teammate in df_raw.columns:
                row[teammate] = 1
            else:
                row[teammate] = 0

        input_df = pd.DataFrame([row])

        # ----------------------------------
        # Apply preprocessing (same as train)
        -----------------------------------
        input_processed = preprocessor.transform(input_df)

        # ----------------------------------
        # Predict win probability
        # ----------------------------------
        prob = rf_model.predict_proba(input_processed)[0][1]

        if prob > best_prob:
            best_prob = prob
            best_agent = agent

    return best_agent, best_prob


# -------------------------------
# 5. Test run
# -------------------------------
df_raw = pd.read_csv("data/match_sample_preprocessed.csv")

recommended_agent, win_prob = recommend_agent_rf(
    "Ascent", ["Jett", "Cypher", "Reyna", "Omen"], df_raw
)

print(f"Recommended Agent: {recommended_agent}, Predicted Win Probability: {win_prob*100:.2f} %")

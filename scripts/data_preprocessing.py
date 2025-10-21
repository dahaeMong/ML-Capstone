import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# -------------------------------
# 1. Open JSON data file
# -------------------------------
with open("data/match_sample.json", "r", encoding="utf-8") as f:
    matches = json.load(f)

df = pd.DataFrame(matches)

# -------------------------------
# 2. Extract target stats
# -------------------------------
df["KDA"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)
df["headshot_pct"] = df["headshot%"]
df.drop(columns=["KDA", "headshot%"], inplace=True, errors='ignore') 
# -------------------------------
# 3. Feature Engineering
# -------------------------------

# Map & target agent encoding
le_map = LabelEncoder()
df["map_encoded"] = le_map.fit_transform(df["map"])

le_agent = LabelEncoder()
df["agent_encoded"] = le_agent.fit_transform(df["agent"])

# Team One-Hot Encoding
mlb = MultiLabelBinarizer()
team_encoded = mlb.fit_transform(df["team"])
team_encoded_df = pd.DataFrame(team_encoded, columns=mlb.classes_)
df = pd.concat([df, team_encoded_df], axis=1)

# -------------------------------
# 4. Label win or lose
# -------------------------------
df["win"] = df["result"].astype(int)
df.drop(columns=["result", "team"], inplace=True)

# -------------------------------
# 5. Check preprocessing
# -------------------------------
print(df.head())
print(df.info())

# -------------------------------
# 6. Save as CSV
# -------------------------------
df.to_csv("data/match_sample_preprocessed.csv", index=False)
print("💾 Preprocessed data saved as match_sample_preprocessed.csv")

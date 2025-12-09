# Valorant Agent Recommendation System

![Valorant](https://upload.wikimedia.org/wikipedia/en/9/99/Valorant_cover_art.jpg)

This project implements a **player-centric agent recommendation system** for Valorant. The system predicts the expected win probability for each agent based on the current map and team composition, helping players make strategic agent choices in competitive matches.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Usage](#usage)
- [Future Work](#future-work)

---

## Project Overview
- Goal: Recommend the best agent for a given map and team composition.
- Approach: Predict win probability for all available agents and select the one with the highest probability.
- Note: Currently uses **synthetic sample data** due to Riot API access limitations.

---

## Setup Instructions
git clone https://github.com/dahaeMong/ML-Capstone
cd ML_CAPSTONE
pip install -r requirements.txt
(optional) python scripts/generate_sample.py - only if you want to regenerate synthetic data
python scripts/preprocessing.py
python scripts/train.py
python scripts/evaluation.py
python scripts/recommend.py

## Example Run

Map? bind
Team composition (comma separated, e.g., Jett,Sage,Reyna,Omen)? jett,sage,reyna,omen

Recommended Agents:
Logistic Regression: Phoenix (Predicted Win Probability: 53.54%)
Random Forest: Yoru (Predicted Win Probability: 53.96%)
XGBoost: Phoenix (Predicted Win Probability: 57.74%)


You can input any map and team combination that exists in the dataset.
The system outputs recommended agent and predicted win probability for each trained model.

## Data

Dataset is synthetic, Riot API not used

| Feature | Description |
|---------|-------------|
| `match_id` | Unique match identifier |
| `player` | Target player (`test_player`) |
| `map` | Map name |
| `agent` | Agent played by the target player |
| `team_agents` | List of agents on the same team |
| `kills`, `deaths`, `assists`, `damage`, `headshot_pct` | Individual performance metrics |
| `win` | Match result (0 = lose, 1 = win) |

- **Dataset size:** 100 sample matches
- Stored in: `data/match_sample_preprocessed.csv`

---

## Preprocessing
- Convert JSON data to **pandas DataFrame**
- Extract **target stats** (KDA, headshot%)
- Encode categorical features:
  - Map: Label Encoding
  - Agent: Label Encoding
- Team composition: **One-Hot Encoding** (excluding the target player)
- Target variable: `win` as binary
- Save processed data as CSV

---

## Models

### 1. Baseline: Logistic Regression
- Features: Map encoding, team agent One-Hot encoding
- Target: `win` (0/1)
- Recommendation: Predict win probability for all agents and select the highest

### 1-2 Scaled_Baseline: Logistic Regression
- Features: Map encoding, team agent One-Hot encoding
- Target: `win` (0/1)
- Recommendation: Predict win probability for all agents and select the highest

### 2. Random Forest
- Features: Same as baseline
- Ensemble of decision trees to model non-linear relationships between map, team composition, and win probability
- Handles categorical and numerical features without much preprocessing
- Generates predicted win probabilities for each agent
- Hyperparameters can be tuned (e.g., number of trees, max depth) to improve performance

### 3. Advanced: XGBoost
- Features: Same as baseline
- Generates predicted win probabilities for recommendation
- Hyperparameters optimized for small dataset

### eda.py result
=== Basic Statistics ===
         match_id       kills      deaths     assists      damage  headshot_pct  ...        Veto       Viper      Waylay        Yoru         win         KDA
count  100.000000  100.000000  100.000000  100.000000  100.000000     100.00000  ...  100.000000  100.000000  100.000000  100.000000  100.000000  100.000000
mean    50.500000   16.690000   15.920000    6.800000  198.300000      23.10500  ...    0.230000    0.230000    0.180000    0.230000    0.460000    1.794542
std     29.011492    6.949595    5.936839    4.417596   60.014561       6.82678  ...    0.422953    0.422953    0.386123    0.422953    0.500908    1.212430
min      1.000000    5.000000    5.000000    0.000000  100.000000      10.20000  ...    0.000000    0.000000    0.000000    0.000000    0.000000    0.444444
25%     25.750000   11.000000   11.000000    3.000000  144.750000      17.70000  ...    0.000000    0.000000    0.000000    0.000000    0.000000    1.097727
50%     50.500000   16.000000   16.000000    7.000000  211.000000      23.85000  ...    0.000000    0.000000    0.000000    0.000000    0.000000    1.398026
75%     75.250000   21.250000   21.000000   10.250000  241.000000      28.15000  ...    0.000000    0.000000    0.000000    0.000000    1.000000    2.203571
max    100.000000   30.000000   25.000000   15.000000  300.000000      34.40000  ...    1.000000    1.000000    1.000000    1.000000    1.000000    6.800000

[8 rows x 32 columns]

=== Categorical Value Counts ===
map       agent
Fracture  Viper        3
Bind      Breach       2
Breeze    Omen         2
Sunset    Clove        2
Pearl     Tejo         2
                      ..
Sunset    Brimstone    1
          Cypher       1
          Iso          1
          Phoenix      1
          Waylay       1
Name: count, Length: 82, dtype: int64

📝 Analysis: Overall win rate is 46.0%. Class balance looks okay.

📝 Analysis: Map-wise win rate indicates which maps are more favorable.

📝 Analysis: Agent-wise win rate can show which agents perform better on average.

📝 Analysis: KDA ranges from 0.44 to 6.80, mean 1.79.

📝 Analysis: Headshot % ranges from 10.2% to 34.4%, mean 23.1%.

📝 Analysis: Correlation matrix shows relationships among KDA, headshot %, and win. Higher KDA/headshot % may correlate with higher win probability.
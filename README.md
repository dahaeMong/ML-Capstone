# Valorant Agent Recommendation System

![Valorant](https://upload.wikimedia.org/wikipedia/en/9/99/Valorant_cover_art.jpg)

This project implements a **player-centric agent recommendation system** for Valorant. The system predicts the expected win probability for each agent based on the current map and team composition, helping players make strategic agent choices in competitive matches.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Usage](#usage)
- [Future Work](#future-work)
- [Requirements](#requirements)

---

## Project Overview
- Goal: Recommend the best agent for a given map and team composition.
- Approach: Predict win probability for all available agents and select the one with the highest probability.
- Note: Currently uses **synthetic sample data** due to Riot API access limitations.

---

## Data

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


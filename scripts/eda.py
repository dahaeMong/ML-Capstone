import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load preprocessed data
# -------------------------------
df = pd.read_csv("data/match_sample_preprocessed.csv")
df["KDA"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0,1)


# -------------------------------
# 2. Basic statistics
# -------------------------------
print("=== Basic Statistics ===")
print(df.describe())

print("\n=== Categorical Value Counts ===")
print(df[['map', 'agent']].value_counts())

# -------------------------------
# 3. Win/Loss distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='win', data=df)
plt.title("Win/Loss Distribution")
plt.show()

win_ratio = df['win'].mean()
print(f"\nAnalysis: Overall win rate is {win_ratio*100:.1f}%. "
      f"{'Class balance looks okay.' if 0.4 <= win_ratio <= 0.6 else 'Potential class imbalance detected.'}")

# -------------------------------
# 4. Map-wise win rate
# -------------------------------
map_win_rate = df.groupby('map')['win'].mean().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x=map_win_rate.index, y=map_win_rate.values)
plt.title("Win Rate by Map")
plt.ylabel("Win Rate")
plt.xticks(rotation=45)
plt.show()
print("\nAnalysis: Map-wise win rate indicates which maps are more favorable.")

# -------------------------------
# 5. Agent-wise win rate
# -------------------------------
agent_win_rate = df.groupby('agent')['win'].mean().sort_values()
plt.figure(figsize=(12,5))
sns.barplot(x=agent_win_rate.index, y=agent_win_rate.values)
plt.title("Win Rate by Agent")
plt.ylabel("Win Rate")
plt.xticks(rotation=45)
plt.show()
print("\nAnalysis: Agent-wise win rate can show which agents perform better on average.")

# -------------------------------
# 6. KDA and Headshot % distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['KDA'], bins=20, kde=True)
plt.title("KDA Distribution")
plt.show()
print(f"\nAnalysis: KDA ranges from {df['KDA'].min():.2f} to {df['KDA'].max():.2f}, mean {df['KDA'].mean():.2f}.")

plt.figure(figsize=(6,4))
sns.histplot(df['headshot_pct'], bins=20, kde=True)
plt.title("Headshot % Distribution")
plt.show()
print(f"\nAnalysis: Headshot % ranges from {df['headshot_pct'].min():.1f}% to {df['headshot_pct'].max():.1f}%, mean {df['headshot_pct'].mean():.1f}%.")

# -------------------------------
# 7. Correlation matrix
# -------------------------------
numeric_cols = ['KDA', 'headshot_pct', 'win']
corr = df[numeric_cols].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

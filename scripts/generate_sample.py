import os
import json
import random

os.makedirs("data", exist_ok=True)

# basic setup for valid choices
maps = [
    "Ascent", "Bind", "Haven", "Lotus", "Split", "Pearl", "Icebox",
    "Breeze", "Fracture", "Sunset", "Abyss", "Corrode"
]
agents = [
    "Brimstone", "Viper", "Omen", "Killjoy", "Cypher", "Sova", "Sage",
    "Phoenix", "Jett", "Reyna", "Raze", "Breach", "Astra", "KAY/O", "Yoru",
    "Neon", "Fade", "Veto", "Clove", "Iso", "Waylay", "Tejo"
]
def generate_match(i):
    """Random match generating focus on this player"""
    chosen_agent = random.choice(agents)
    available_agents = [a for a in agents if a != chosen_agent]

    match = {
        "match_id": i,
        "player": "test_player",
        "map": random.choice(maps),
        "agent": chosen_agent,  
        "kills": random.randint(5, 30),
        "deaths": random.randint(5, 25),
        "assists": random.randint(0, 15),
        "headshot%": round(random.uniform(10.0, 35.0), 1),
        "damage": random.randint(100, 300),
        "team": random.sample(available_agents, 4), 
        "result": random.choice([0, 1]), 
    }
    return match

sample_data = [generate_match(i) for i in range(1, 101)]

# Save as json file
output_path = "data/match_sample.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=4, ensure_ascii=False)

print(f"✅ Generated {len(sample_data)} matches and saved to {output_path}")

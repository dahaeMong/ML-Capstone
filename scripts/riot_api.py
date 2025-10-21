import os
import requests
import json

# -------------------------------
# 1. Account info
# -------------------------------
NAME = "GEN t3xture"  
TAG = "9999"     
REGION = "kr"  

# -------------------------------
# 2. Riot API 
# -------------------------------

def get_puuid(name, tag):
    url = f"https://api.henrikdev.xyz/valorant/v1/account/{name}/{tag}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()["data"]
        print(f"✅ Found player: {data['name']}#{data['tag']}")
        print(f"PUUID: {data['puuid']}")
        return data["puuid"]
    else:
        print(f"❌ Error fetching PUUID: {response.status_code}")
        return None


def get_match_history(puuid, region="na", count=5):
    url = f"https://api.henrikdev.xyz/valorant/v3/by-puuid/matches/{region}/{puuid}?size={count}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()["data"]
        print(f"✅ Retrieved {len(data)} matches.")
        return data
    else:
        print(f"❌ Error fetching matches: {response.status_code}")
        return None


# -------------------------------
#  3. Run & Save
# -------------------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    puuid = get_puuid(NAME, TAG)
    if puuid:
        matches = get_match_history(puuid, REGION)
        
        if matches:
            file_path = "data/match_sample.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(matches, f, indent=4, ensure_ascii=False)
            print(f"💾 Saved to {file_path}")
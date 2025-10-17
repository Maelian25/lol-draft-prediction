import json
import os
import requests
import time
import logging
from datetime import datetime
import random

# 100 requests are allowed every 2 minutes, and 20 requests per seconds
TIME_LIMIT = 120 
REQUEST_LIMIT = 100

DDRAGONVERSION = "15.20.1"

logger = logging.getLogger(__name__)

def riot_request(url, headers):
    while True:
        resp = requests.get(url=url, headers=headers)
        # Rate limit reached
        if resp.status_code == 429:
            delay = int(resp.headers.get("Retry-After", "2"))
            logging.info(f"Rate limited. Sleeping for {delay}s...")
            time.sleep(delay)
            continue
        if resp.status_code != 200:
            logging.error(f"Error {resp.status_code} : {resp.text}")
            return None
        else :
            time.sleep(TIME_LIMIT / REQUEST_LIMIT)
        return resp.json()
    
def save_json_to_dir(data, dir, region, idx):
    try:
        os.makedirs(dir, exist_ok=True)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dir}/{region}_picks_bans_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saving file after {idx} iterations...")
    except IOError as e:
        logger.error(f"Failed to write in file : {e}")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error when saving : {e}")
        return None
    
def convert_unix_timestamp_to_date(ts):
    timestamp = int(ts)
    
    return str(datetime.fromtimestamp(timestamp))

def shuffle_picks_order_with_weights(picks, weights = [0.5,0.6,0.6,0.3,0.3]):
    for team in list(picks):
        picks_list = list(picks[team].items())
        order = sorted(range(len(picks_list)), key=lambda i: random.random() ** (weights[i]))
        shuffle_weighted = dict([picks_list[i] for i in order])
        picks[team] = shuffle_weighted
    
    return(picks)

# Function to take a random champ and replace de -1 in bans so that we dont lose too many games for no real reason
def replace_missed_bans(bans):
    champions_file_url = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGONVERSION}/data/en_US/champion.json"
    data = requests.get(champions_file_url).json()
    champion_data = data["data"]
    
    champions_id_and_name = [{int(champion_data[name]["key"]) : champion_data[name]["name"]} for name in champion_data ]
    
    used_champ_ids = [list(b.keys())[0] for b in bans if list(b.keys())[0] != -1]
    champ_replacement = random.choice(list(champions_id_and_name))
    
    for x in bans:
        # Get the current key of the ban
        current_key = list(x.keys())[0]

        if current_key == -1:
            champ_replacement = random.choice(list(champions_id_and_name))
            new_key = list(champ_replacement.keys())[0]
            while new_key in used_champ_ids:
                champ_replacement = random.choice(champions_id_and_name)
                new_key = list(champ_replacement.keys())[0]
            
            x[new_key] = x.pop(current_key)
            used_champ_ids.append(new_key)
        
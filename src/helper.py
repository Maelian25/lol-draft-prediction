import json
import os
from typing import Tuple
import requests
import time
import logging
from datetime import datetime
import random
import pandas as pd

from src.api_key_helper import get_api_key, wait_for_new_key
from src.logger_config import get_logger

# 100 requests are allowed every 2 minutes, and 20 requests per seconds
TIME_LIMIT = 120
REQUEST_LIMIT = 100

DDRAGONVERSION = "15.20.1"

logger = get_logger("Helper", "helper.log")


def riot_request(url, max_retries=5):
    """Make a Riot API request with dynamic key reload and retry logic"""
    for attempt in range(max_retries):
        api_key = get_api_key()
        if not api_key:
            logger.error("No key available.")
            return None
        headers = {"X-Riot-Token": api_key}

        try:
            resp = requests.get(url=url, headers=headers)
        except requests.RequestException as e:
            logger.error(f"Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue

        # Success
        if resp.status_code == 200:
            time.sleep(TIME_LIMIT / REQUEST_LIMIT)
            return resp.json()

        # Rate limit reached
        elif resp.status_code == 429:
            delay = float(resp.headers.get("Retry-After", 5))
            logging.warning(f"Rate limited. Sleeping for {delay}s...")
            time.sleep(delay)
            continue

        # Invalid or expired key
        elif resp.status_code == 401 or resp.status_code == 403:
            logger.error("API key expired or invalid. Waiting for update...")
            global _CACHE_VALID
            _CACHE_VALID = False
            wait_for_new_key()
            continue

        else:
            logging.error(f"Error {resp.status_code} : {resp.text[:200]}")
            time.sleep(5)
            continue

    logger.critical("Max retries reached. Giving up.")
    return None


def save_json_to_dir(data, dir, region, idx, elo):
    try:
        os.makedirs(dir, exist_ok=True)
        os.makedirs(os.path.join(dir, region), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dir}/{region}/{region}_{elo}_picks_bans_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saving file after {idx} iteration(s)...")
    except IOError as e:
        logger.error(f"Failed to write in file : {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error when saving : {e}")
        return None


def convert_unix_timestamp_to_date(ts):
    timestamp = int(ts)

    return str(datetime.fromtimestamp(timestamp))


def shuffle_picks_order_with_weights(picks, weights=[0.5, 0.7, 0.6, 0.2, 0.2]):
    blue_side_picks = []
    red_side_picks = []

    # Create teams from each side
    for p in picks:
        entry = {"order": p["order"], "position": p["position"]}

        (
            blue_side_picks.append(entry)
            if p["side"] == "blue"
            else red_side_picks.append(entry)
        )

    # Internal functions to shuffle sides
    def shuffle_side(side_picks):
        n = len(side_picks)

        shuffled_indices = sorted(
            range(n), key=lambda i: random.random() ** (weights[i])
        )

        new_side_picks = [
            {
                "order": new_order + 1,
                "position": side_picks[position]["position"],
            }
            for new_order, position in enumerate(shuffled_indices)
        ]

        return new_side_picks

    new_blue_side_picks = shuffle_side(blue_side_picks)
    new_red_side_picks = shuffle_side(red_side_picks)

    new_blue_side = []
    new_red_side = []
    for p in picks:
        for obj in new_blue_side_picks:
            if obj["position"] == p["position"] and p["side"] == "blue":
                new_blue_side.append(
                    {
                        "side": p["side"],
                        "championId": p["championId"],
                        "position": p["position"],
                        "order": obj["order"],
                    }
                )
                new_blue_side.sort(key=lambda x: x["order"])
        for obj in new_red_side_picks:
            if obj["position"] == p["position"] and p["side"] == "red":
                new_red_side.append(
                    {
                        "side": p["side"],
                        "championId": p["championId"],
                        "position": p["position"],
                        "order": obj["order"],
                    }
                )
                new_red_side.sort(key=lambda x: x["order"])

    return new_blue_side + new_red_side


# Function to take a random champ and replace de -1 in bans
# so that we dont lose too many games for no real reason
def replace_missed_bans(bans):
    champions_file_url = (
        f"https://ddragon.leagueoflegends.com/cdn/"
        f"{DDRAGONVERSION}/data/en_US/champion.json"
    )
    data = requests.get(champions_file_url).json()
    champion_data = data["data"]

    champions_id_and_name = [
        {int(champion_data[name]["key"]): champion_data[name]["name"]}
        for name in champion_data
    ]

    used_champ_ids = [b["championId"] for b in bans if b["championId"] != -1]
    champ_replacement = random.choice(list(champions_id_and_name))

    for b in bans:
        if b["championId"] == -1:
            champ_replacement = random.choice(list(champions_id_and_name))
            new_champ = list(champ_replacement.keys())[0]

            while new_champ in used_champ_ids:
                champ_replacement = random.choice(champions_id_and_name)
                new_champ = list(champ_replacement.keys())[0]

            b["championId"] = new_champ
            used_champ_ids.append(new_champ)


def load_scrapped_data(save_path, regionId, elo) -> Tuple[pd.DataFrame, bool]:

    if not os.path.exists(save_path):
        logger.warning("The directory doesn't exist")
        return pd.DataFrame(), False

    data_files = [
        f
        for f in os.listdir(save_path)
        if f.endswith(".json") and f.startswith(f"{regionId}_{elo}")
    ]
    if not data_files:
        logger.warning("No file found in the directory")
        return pd.DataFrame(), False

    return_file = ""
    max_matches_in_a_file = 0
    matches_number = 0

    for f in data_files:
        full_path = os.path.join(save_path, f)
        try:
            with open(full_path, "r") as file:
                data_dict = json.load(file)
                matches_number = len(data_dict)
        except Exception as e:
            logger.error(f"Error while trying to read file {f} : {e}")

        if matches_number > max_matches_in_a_file:
            max_matches_in_a_file = matches_number
            return_file = f
        elif matches_number == max_matches_in_a_file:
            if os.path.getmtime(full_path) > os.path.getmtime(
                os.path.join(save_path, return_file)
            ):
                return_file = f

    if not return_file:
        logger.info("Did not find a proper file")
        return pd.DataFrame(), False

    logger.info(f"Found a proper file! Loading file {return_file}")

    with open(os.path.join(save_path, return_file)) as file:
        json_string = json.load(file)
        return pd.json_normalize(json_string), True

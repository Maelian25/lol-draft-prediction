import json
import os
from typing import Any, Dict, List, Tuple
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
CHAMPION_FILE_URL = (
    f"https://ddragon.leagueoflegends.com/cdn/"
    f"{DDRAGONVERSION}/data/en_US/champion.json"
)

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
    """Save a json scrapped into a directory proper to elo and region"""
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
    """Convert timestamp to actual date"""
    timestamp = int(ts)

    return str(datetime.fromtimestamp(timestamp))


def shuffle_picks_order_with_weights(picks, weights=[0.5, 0.7, 0.6, 0.2, 0.2]):
    """Shuffle picks order to make data more realistic since
    there is no way to get pick order from api"""
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
    """Replace a ban in the dataset that is -1 so that there is 10 bans per game"""
    champions_id_and_name = get_champions_id_name_map()
    all_champ_ids = list(champions_id_and_name.keys())

    used_champ_ids = [b["championId"] for b in bans if b["championId"] != -1]

    for b in bans:
        if b["championId"] == -1:
            available_champs = list(set(all_champ_ids) - set(used_champ_ids))
            if not available_champs:
                break
            new_champ = random.choice(available_champs)

            b["championId"] = new_champ
            used_champ_ids.append(new_champ)


def load_scrapped_data(save_path, regionId, elo) -> Tuple[pd.DataFrame, bool]:
    """Allow loading data from files instead of scrapping it again"""
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


def get_champions_id_name_map() -> Dict[int, str]:
    """Provide mapping id to name for champions in the dataset"""
    response = requests.get(CHAMPION_FILE_URL)
    response.raise_for_status()

    data = response.json().get("data", {})

    champions_id_and_name = {
        int(champ_info["key"]): champ_info["name"].capitalize()
        for champ_info in data.values()
    }

    return champions_id_and_name


def get_champions_data() -> dict:
    response = requests.get(CHAMPION_FILE_URL)
    response.raise_for_status()

    data = response.json().get("data", {})

    champ_data: dict[int, Any] = {int(v["key"]): v for _, v in data.items()}

    return champ_data


def champ_id_to_idx_map():
    """Provide mapping btw champ id and idx"""
    champ_id_to_idx_map = {
        champ: idx for idx, champ in enumerate(get_champions_id_name_map().keys())
    }

    return champ_id_to_idx_map


def champName_to_champId(champName: str):
    """Provide corresponding id for a given name"""
    champions_name_and_id = {k: v for v, k in get_champions_id_name_map().items()}
    champ_id = champions_name_and_id.get(champName.capitalize())

    if not champ_id:
        return -1

    return champ_id


def replace_wrong_position(dataset: pd.DataFrame):
    """Replace positions that would be corrupted in the dataset"""

    def fix_positions(picks):
        for pick in picks:
            if pick.get("position", "") == "":
                pick["position"] = "SUPPORT"
        return picks

    dataset["picks"] = dataset["picks"].apply(fix_positions)
    return dataset


def tags_one_hot_encoder(unique_tags: List[str]):
    champ_tags_dict: Dict[int, Dict[str, int]] = {}

    for champ_id, data in get_champions_data().items():
        current_champ_tags = data["tags"]
        champ_tags_dict[champ_id] = {}

        for tag in unique_tags:
            if tag in current_champ_tags:
                champ_tags_dict[champ_id][f"tag_{tag}"] = 1
            else:
                champ_tags_dict[champ_id][f"tag_{tag}"] = 0

    return champ_tags_dict


def unique_tags():
    unique_tags = set()

    for _, data in get_champions_data().items():
        current_champ_tags: List[str] = data["tags"]
        unique_tags.update(current_champ_tags)

    return list(unique_tags)

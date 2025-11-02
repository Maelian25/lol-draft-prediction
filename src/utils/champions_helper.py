from typing import Any, Dict, List
import requests

from src.utils.logger_config import get_logger

DDRAGONVERSION = "15.20.1"

CHAMPION_FILE_URL = (
    f"https://ddragon.leagueoflegends.com/cdn/"
    f"{DDRAGONVERSION}/data/en_US/champion.json"
)

logger = get_logger("Helper", "champions_helper.log")


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


def unique_tags():
    unique_tags = set()

    for _, data in get_champions_data().items():
        current_champ_tags: List[str] = data["tags"]
        unique_tags.update(current_champ_tags)

    return list(unique_tags)


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

from typing import List, Optional
import pandas as pd
from sklearn import preprocessing

from src.utils.champions_helper import (
    champ_id_to_idx_map,
    get_champions_data,
    get_champions_id_name_map,
    tags_one_hot_encoder,
    unique_tags,
)
from src.utils.data_helper import replace_wrong_position
from src.utils.logger_config import get_logger


class BaseAnalysis:
    """
    Basic class containing every module needed for a functional analysis

    Args :
        data -> dataset retrieve from Riot API
        patches -> list of patches to filter data on
    """

    def __init__(self, data: pd.DataFrame, patches: Optional[List[str]] = None) -> None:

        self.logger = get_logger(self.__class__.__name__, "analysis.log")

        self.dataset = data.copy()

        if patches:
            self.dataset = self.dataset[self.dataset["game_version"].isin(patches)]
        else:
            self.logger.info("Analysing full dataset")

        self.__clean_dataset()
        self.__define_maps()
        self.__define_encoders()

        self.num_matches = len(self.dataset)

        self.logger.info(f"Dataset succesfully loaded ({self.num_matches} matches)")

    def __clean_dataset(self):
        """
        Private fn helping clean the dataset
        """

        self.dataset = self.dataset.drop_duplicates(
            subset=["match_id"], ignore_index=True
        ).dropna()

        # Convert game_duration to numeric, coercing errors to NaN
        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )

        # Replace wrong positions
        self.dataset = replace_wrong_position(self.dataset)

    def __define_maps(self):
        """
        Private fn loading several mappings
        """

        # Load champion name and ID maps
        self.champ_id_name_map = get_champions_id_name_map()
        self.champ_name_id_map = {v: k for k, v in self.champ_id_name_map.items()}

        # Load champion ID to index map and reciprocal
        self.champ_id_to_idx_map = champ_id_to_idx_map()
        self.idx_to_champ_id_map = {v: k for k, v in self.champ_id_to_idx_map.items()}

        # Get champions data once and for all
        self.champions_data = get_champions_data()

        # Number of champions in our system
        self.n_champs = len(self.champ_id_name_map)

    def __define_encoders(self):
        """
        Private fn providing encoders for future actions
        """

        self.scaler = preprocessing.StandardScaler()

        self.unique_tags = unique_tags()
        self.tags_encoder = tags_one_hot_encoder(self.unique_tags)

    def get_champion_infos(self, champ_id: int):
        """
        Encode tags and returns every info needed in a list
        """

        tags_encoded = list(self.tags_encoder[champ_id].values())

        infos = list(self.champions_data[champ_id]["info"].values())
        stats = list(self.champions_data[champ_id]["stats"].values())

        # Concat all three lists
        stats_list = infos + stats + tags_encoded

        return stats_list

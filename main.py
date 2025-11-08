from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List

import numpy as np
import pandas as pd

from src.analysis import DatasetAnalysis
from src.dataset import Dataset
from src.utils.champions_helper import champName_to_champId
from src.utils.data_helper import (
    load_scrapped_data,
)
from src.utils.logger_config import get_logger

from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("Main", "main_file.log")

SAVE_AFTER_ITERATION = 1000
SAVE_PATH = os.path.join(os.getcwd(), "datasets")
REGIONS = dict({"EUROPE": "EU", "KOREA": "KR", "AMERICA": "US"})  # region : save_dir
ELOS = dict(
    {"challenger": 125, "grandmaster": 250, "master": 500}
)  # elo : player_count


def get_regional_data_from_api_threaded(region, short_name):
    """
    Provide regional data given a region.
    Necessary to thread requests
    """
    region_logger = get_logger(region, f"data_scrapping_{region.lower()}.log")
    region_logger.info(f"Starting region {region}")

    regional_dataframes: List[pd.DataFrame] = []

    for elo, player_count in ELOS.items():
        region_save_dir = os.path.join(SAVE_PATH, short_name)
        os.makedirs(region_save_dir, exist_ok=True)

        region_logger.info(f"Started scrapping {region}-{elo}")

        df_region, data_scrapped = load_scrapped_data(region_save_dir, short_name, elo)

        if not data_scrapped:
            # Creating dataset
            dataset = Dataset(
                region=region,
                queue="RANKED_SOLO_5x5",
                game_count=50,
                player_count=player_count,
                elo=elo,
                save_after_iteration=SAVE_AFTER_ITERATION,
            )

            # Extracting and save data
            matches = dataset.extract_match_data(short_name)
            df_region = pd.DataFrame(matches)

        regional_dataframes.append(df_region)

        region_logger.info(f"Finished scrapping {region}-{elo}")
    region_logger.info(f"Finished scrapping {region}")
    return pd.concat(regional_dataframes, ignore_index=True)


if __name__ == "__main__":

    try:
        # --- Scrapping data ---
        # Parallelism
        with ThreadPoolExecutor(max_workers=len(REGIONS)) as executor:
            futures = {
                executor.submit(
                    get_regional_data_from_api_threaded, region, short_name
                ): region
                for region, short_name in REGIONS.items()
            }

            df_world_list: List[pd.DataFrame] = []
            for future in as_completed(futures):
                region = futures[future]
                try:
                    df_regional = future.result()
                    df_world_list.append(df_regional)
                    logger.info(f"Region {region} succesfully completed")
                except Exception as e:
                    logger.error(f"Region {region} failed: {e}", exc_info=True)

        if not df_world_list:
            raise RuntimeError("No data was successfully loaded or scraped.")

        df_world: pd.DataFrame = pd.concat(df_world_list, ignore_index=True)

        # --- Analysing data ---
        # Analysing the full dataset without patches restriction
        full_dataset_analysis = DatasetAnalysis(df_world, compute_matrices=False)
        full_dataset_analysis.get_win_rate_per_side()
        full_dataset_analysis.get_patch_distribution()
        full_dataset_analysis.get_game_duration_stats(True)
        full_dataset_analysis.get_draft_order_correlation()
        logger.info("Analysis ended for the full dataset")

        # Analysing the full dataset with patches restriction
        patches = ["15.19", "15.20", "15.21"]
        dataset_on_patch_restriction = DatasetAnalysis(
            df_world, compute_matrices=True, build_matches_for_ml=True, patches=patches
        )

        dataset_on_patch_restriction.get_win_rate_per_side()
        dataset_on_patch_restriction.get_game_duration_stats(plot=False)
        pick_rate = dataset_on_patch_restriction.get_champ_pick_or_ban_rate(
            pick=True, plot=True
        )
        ban_rate = dataset_on_patch_restriction.get_champ_pick_or_ban_rate(
            pick=False, plot=True
        )

        dataset_on_patch_restriction.get_first_pick_stats()

        # Currently without any plotting of any sort
        win_rate = dataset_on_patch_restriction.get_champ_win_rate()

        # Analysing a champion as an example on chosen patches
        champ_to_analyse = champName_to_champId("kai'sa")
        dataset_on_patch_restriction.get_role_distribution(champ=champ_to_analyse)
        all_counters = dataset_on_patch_restriction.top_10_matchups_for(
            champ_to_analyse, plot=True
        )
        logger.info(f"Pick rate for Kai'Sa: {pick_rate[champ_to_analyse] * 100:.2f}%")
        logger.info(f"Ban rate for Kai'Sa: {ban_rate[champ_to_analyse] * 100:.2f}%")

        team_comp_to_analyse = ["ornn", "lee sin", "orianna", "yunara", "nautilus"]
        team_comp_to_analyse_ids = [
            champName_to_champId(champ) for champ in team_comp_to_analyse
        ]

        dataset_on_patch_restriction.get_synergy(
            champ_id_1=team_comp_to_analyse_ids[0],
            champ_id_2=team_comp_to_analyse_ids[1],
            log=True,
        )
        dataset_on_patch_restriction.get_team_synergy(team_comp_to_analyse_ids)

        champ_embeddings = dataset_on_patch_restriction.champion_embeddings()

        ahri = "Ahri"
        syndra = "Syndra"
        malphite = "Malphite"

        cos_sim_1 = cosine_similarity(
            np.array(champ_embeddings.loc[champName_to_champId(ahri)]).reshape(1, -1),
            np.array(champ_embeddings.loc[champName_to_champId(syndra)]).reshape(1, -1),
        )

        cos_sim_2 = cosine_similarity(
            np.array(champ_embeddings.loc[champName_to_champId(ahri)]).reshape(1, -1),
            np.array(champ_embeddings.loc[champName_to_champId(malphite)]).reshape(
                1, -1
            ),
        )

        logger.info(f"Similarity btw {ahri} and {syndra} : {cos_sim_1[0][0]:.2f}")
        logger.info(f"Similarity btw {ahri} and {malphite} : {cos_sim_2[0][0]:.2f}")

        logger.info("Analysis ended for the restricted dataset")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

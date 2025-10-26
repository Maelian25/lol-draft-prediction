from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List

import pandas as pd

from src.analysis import DatasetAnalysis
from src.dataset import Dataset
from src.helper import load_scrapped_data
from src.logger_config import get_logger

logger = get_logger("Main", "main_file.log")

SAVE_AFTER_ITERATION = 1000
SAVE_PATH = os.path.join(os.getcwd(), "datasets")
REGIONS = dict({"EUROPE": "EU", "KOREA": "KR", "AMERICA": "US"})  # region : save_dir
ELOS = dict(
    {"challenger": 125, "grandmaster": 250, "master": 500}
)  # elo : player_count


def get_regional_data_from_api_threaded(region, short_name):
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
        analysis = DatasetAnalysis(df_world)
        analysis.get_win_rate_per_side()
        analysis.get_patch_distribution()
        analysis.get_game_duration_stats()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

import logging
import os
from typing import List

import pandas as pd

from src.analysis import DatasetAnalysis
from src.dataset import Dataset
from src.helper import load_scrapped_data


if __name__ == "__main__":

    # Logger setup
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./logs/main_file.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Constants
    SAVE_AFTER_ITERATION = 1000

    try:
        # --- Scrapping data ---
        save_path = os.path.join(os.getcwd(), "datasets")

        regions = dict(
            {"EUROPE": "EU", "KOREA": "KR", "AMERICA": "US"}
        )  # region : save_dir
        elos = dict(
            {"challenger": 125, "grandmaster": 250, "master": 500}
        )  # elo : player_count
        data_from_every_region: List[pd.DataFrame] = []

        for region, short_name in regions.items():
            for elo, player_count in elos.items():
                region_save_dir = os.path.join(save_path, short_name)
                os.makedirs(region_save_dir, exist_ok=True)

                logger.info(f"Starting data processing for {region} ({elo})")

                df_region, data_scrapped = load_scrapped_data(
                    os.path.join(save_path, regions[region]), region, elo
                )

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

                data_from_every_region.append(df_region)

                logger.info(f"Finished {region}-{elo}")

            logger.info(f"Finished {region}")

        if not data_from_every_region:
            raise RuntimeError("No data was successfully loaded or scraped.")

        df_all_matches_scrapped = pd.concat(data_from_every_region, ignore_index=True)

        # --- Analysing data ---
        # First dataset scrapped for now
        analysis = DatasetAnalysis(df_all_matches_scrapped)
        blue_side_win_rate, red_side_win_rate = analysis.get_win_rate_per_side()
        analysis.get_patch_distribution()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

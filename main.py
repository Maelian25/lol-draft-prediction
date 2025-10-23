import logging
import os

from dotenv import load_dotenv

from src.analysis import DatasetAnalysis
from src.dataset import Dataset
from src.helper import load_scrapped_data


if __name__ == "__main__":

    # Logger config
    os.makedirs("./logs", exist_ok=True)

    logger = logging.getLogger(__name__)

    # Constants
    load_dotenv("secrets/api_keys.key")
    API_KEY = os.getenv("RIOT_API_KEY")
    HEADERS = {"X-Riot-Token": API_KEY}
    SAVE_AFTER_ITERATION = 1000

    try:
        # --- Scrapping data ---
        save_path = os.path.join(os.getcwd(), "datasets")
        df_matches, data_scrapped = load_scrapped_data(save_path)

        if not data_scrapped:
            # Creating dataset
            europe_chall_dataset = Dataset(
                region="EUROPE",
                queue="RANKED_SOLO_5x5",
                game_count=60,
                player_count=250,
                elo="challenger",
                headers=HEADERS,
                save_after_iteration=SAVE_AFTER_ITERATION,
            )

            # Extracting data
            european_chall_matches = europe_chall_dataset.extract_match_data()
            # Need to do more here
            # Need to rename files so that we can have more regions...

        # --- Analysing data ---
        analysis = DatasetAnalysis(df_matches)
        blue_side_win_rate, red_side_win_rate = analysis.get_win_rate_per_side()
        analysis.get_patch_distribution()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

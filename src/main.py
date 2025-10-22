import logging
import os

from dotenv import load_dotenv

from dataset import Dataset
from helper import load_scrapped_data


if __name__ == "__main__":

    # Logger config
    os.makedirs("./logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("./logs/data_scrapping.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Constants
    load_dotenv("secrets/api_keys.key")
    API_KEY = os.getenv("RIOT_API_KEY")
    HEADERS = {"X-Riot-Token": API_KEY}
    SAVE_AFTER_ITERATION = 1000
    try:
        # Checking to see if the scrapping is already done
        save_path = os.path.join(os.getcwd(), "datasets")
        df_matches, data_scrapped = load_scrapped_data(save_path)

        if not data_scrapped:
            # Creating dataset
            europe__chall_dataset = Dataset(
                region="EUROPE",
                queue="RANKED_SOLO_5x5",
                game_count=60,
                player_count=250,
                elo="challenger",
            )

            # Extracting data
            european_chall_matches = europe__chall_dataset.extract_match_data()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

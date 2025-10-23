import os
import time
import logging
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_VALID_API_KEY = None


# --- Load key dynamically ---
def get_api_key(retry_if_invalid=True):
    """
    Load Riot API key dynamically from .env.
    If invalid, waits for a new one if retry_if_invalid=True.
    """
    global _VALID_API_KEY

    if _VALID_API_KEY:
        return _VALID_API_KEY

    while True:
        load_dotenv("secrets/api_keys.key", override=True)
        api_key = os.getenv("RIOT_API_KEY")

        if api_key and api_key.startswith("RGAPI-"):
            # Checking api key validity
            logger.info("Checking API key...")
            if test_api_key(api_key):
                _VALID_API_KEY = api_key
                return _VALID_API_KEY
            else:
                logger.warning("Current Riot API key invalid or expired.")

        else:
            logger.warning("No Riot API key found in file.")

        if not retry_if_invalid:
            return None

        wait_for_new_key()  # block until a valid key is added


def wait_for_new_key(check_interval=30):
    """
    Blocks until a valid key appears in the .env file and passes a live test.
    """
    global _VALID_API_KEY
    logger.info("Waiting for a valid Riot API key...")
    _VALID_API_KEY = None

    while True:
        time.sleep(check_interval)
        load_dotenv("secrets/api_keys.key", override=True)
        api_key = os.getenv("RIOT_API_KEY")

        if api_key and api_key.startswith("RGAPI-"):
            logger.info("Checking API key...")
            if test_api_key(api_key):
                logger.info("New Riot API key is valid. Resuming scraping...")
                _VALID_API_KEY = api_key
                time.sleep(5)
                return
            else:
                logger.warning("New key invalid (401 or 403). Waiting for another...")
        else:
            logger.debug("No valid key format detected.")


def test_api_key(api_key: str) -> bool:
    """
    Test the Riot API key by hitting a harmless endpoint.
    Returns True if valid, False otherwise.
    """
    url = "https://euw1.api.riotgames.com/lol/platform/v3/champion-rotations"
    headers = {"X-Riot-Token": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return True
        elif resp.status_code in (401, 403):
            return False
        else:
            logger.warning(f"Unexpected response {resp.status_code}: {resp.text[:100]}")
            return False
    except requests.RequestException as e:
        logger.error(f"Network error while testing API key: {e}")
        return False

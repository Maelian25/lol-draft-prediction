import os
import time
from typing import Optional
import requests
from dotenv import load_dotenv

from src.logger_config import get_logger

logger = get_logger("Api_key_helper", "api_key_helper.log")

# global caching
_CACHED_API_KEY: Optional[str] = None
_CACHE_VALID: bool = False

# endpoint test (harmless)
_TEST_URL = "https://euw1.api.riotgames.com/lol/platform/v3/champion-rotations"
_REQUEST_TIMEOUT = 5  # sec
_WAIT_CHECK_INTERVAL = 15  # sec


def load_key_from_file(env_path: str = "secrets/api_keys.key") -> Optional[str]:
    """Reloads .env file and returns found key (or None)."""
    load_dotenv(env_path, override=True)
    return os.getenv("RIOT_API_KEY")


# --- Load key dynamically ---
def get_api_key(env_path="secrets/api_keys.key", retry_if_invalid=True):
    """
    Load Riot API key dynamically from .env.
    If invalid, waits for a new one if retry_if_invalid=True.
    """
    global _CACHED_API_KEY, _CACHE_VALID

    key_in_file = load_key_from_file(env_path)

    if _CACHED_API_KEY and _CACHE_VALID and key_in_file == _CACHED_API_KEY:
        return _CACHED_API_KEY

    if key_in_file and key_in_file.startswith("RGAPI-"):
        logger.info("New API key detected. Testing key...")
        if test_api_key(key_in_file):
            _CACHED_API_KEY = key_in_file
            _CACHE_VALID = True
            logger.info(
                "Valid key. Using it without retesting until key remains the same"
            )
            return _CACHED_API_KEY
        else:
            _CACHED_API_KEY = key_in_file
            _CACHE_VALID = False
            logger.warning("Current Riot API key invalid or expired.")

    if retry_if_invalid:
        wait_for_new_key(env_path)
        return _CACHED_API_KEY
    else:
        return None


def wait_for_new_key(
    env_path="secrets/api_keys.key", check_interval=_WAIT_CHECK_INTERVAL
):
    """
    Blocks until a valid key appears in the .env file and passes a live test.
    """
    global _CACHED_API_KEY, _CACHE_VALID
    logger.info("Waiting for a valid Riot API key...")

    while True:
        api_key = load_key_from_file(env_path)

        if api_key and api_key.startswith("RGAPI-"):
            logger.info("Checking API key...")
            if test_api_key(api_key):
                _CACHED_API_KEY = api_key
                _CACHE_VALID = True
                logger.info("New Riot API key is valid. Resuming scraping...")
                time.sleep(5)
                return
            else:
                logger.warning("New key invalid (401 or 403). Waiting for another...")
        else:
            logger.debug("No valid key format detected.")
        time.sleep(check_interval)


def test_api_key(api_key: str) -> bool:
    """
    Test the Riot API key by hitting a harmless endpoint.
    Returns True if valid, False otherwise.
    """
    if not api_key:
        return False
    headers = {"X-Riot-Token": api_key}

    try:
        resp = requests.get(url=_TEST_URL, headers=headers, timeout=_REQUEST_TIMEOUT)
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

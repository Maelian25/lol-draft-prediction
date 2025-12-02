from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Dict, List, Optional

import pandas as pd

from src.data_scrapping.dataset import Dataset
from src.utils.data_helper import load_scrapped_data
from src.utils.logger_config import get_logger
from src.utils.constants import DATASETS_FOLDER, ELOS, SAVE_AFTER_ITERATION, REGIONS

logger = get_logger("scraper", "data_scraping.log")


def get_regional_data_from_api_threaded(
    region: str, short_name: str, no_scrape: bool = False
) -> pd.DataFrame:
    """Return regional data as a DataFrame.

    If `no_scrape` is True, the function will only use already-saved files and
    will skip API calls for missing cached files.
    """
    region_logger = get_logger(region, f"data_scrapping_{region.lower()}.log")
    region_logger.info(f"Starting region {region}")

    regional_dataframes: List[pd.DataFrame] = []

    for elo, player_count in ELOS.items():
        region_save_dir = os.path.join(DATASETS_FOLDER, short_name)
        os.makedirs(region_save_dir, exist_ok=True)

        region_logger.info(f"Started scrapping {region}-{elo}")

        if no_scrape:
            region_logger.info(
                f"--no-scrape set; will only load existing data for {region}-{elo}"
            )

            df_region, data_scrapped = load_scrapped_data(
                region_save_dir, short_name, elo
            )
            if not data_scrapped:
                region_logger.warning(
                    f"No cached data for {region}-{elo} and --no-scrape set; skipping."
                )
                continue
        else:
            dataset = Dataset(
                region=region,
                queue="RANKED_SOLO_5x5",
                game_count=100,
                player_count=player_count,
                elo=elo,
                save_after_iteration=SAVE_AFTER_ITERATION,
            )

            matches = dataset.extract_match_data(short_name)
            df_region = pd.DataFrame(matches)

        if df_region is not None and not df_region.empty:
            regional_dataframes.append(df_region)

        region_logger.info(f"Finished scrapping {region}-{elo}")

    region_logger.info(f"Finished scrapping {region}")
    if not regional_dataframes:
        region_logger.warning(f"No data available for region {region}")
        return pd.DataFrame()

    return pd.concat(regional_dataframes, ignore_index=True)


def scrape_world_data(
    regions: Optional[Dict[str, str]] = None,
    max_workers: Optional[int] = None,
    no_scrape: bool = False,
) -> pd.DataFrame:
    """Scrape or load datasets for all regions and return a DataFrame.

    The `no_scrape` flag is propagated to per-region loader so that no API
    calls are made when set.
    """
    if regions is None:
        regions = REGIONS

    if max_workers is None:
        max_workers = min(len(regions), max(1, (os.cpu_count() or 1)))

    logger.info(f"Starting data loading/scraping for regions: {list(regions.keys())}")

    df_world_list: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                get_regional_data_from_api_threaded, region, short_name, no_scrape
            ): region
            for region, short_name in regions.items()
        }

        for future in as_completed(futures):
            region = futures[future]
            try:
                df_regional = future.result()
                if df_regional is None or df_regional.empty:
                    logger.warning(f"No data returned for region {region}; skipping.")
                    continue

                df_world_list.append(df_regional)
                logger.info(f"Region {region} successfully completed")
            except Exception:
                logger.exception(f"Region {region} failed during scraping/loading")

    if not df_world_list:
        raise RuntimeError("No data was successfully loaded or scraped.")

    return pd.concat(df_world_list, ignore_index=True)

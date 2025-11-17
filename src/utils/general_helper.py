from datetime import datetime
import os

import pandas as pd
import torch

from src.utils.logger_config import get_logger

logger = get_logger("Helper", "general_helper.log")


def convert_unix_timestamp_to_date(ts):
    """Convert timestamp to actual date"""
    timestamp = int(ts)

    return str(datetime.fromtimestamp(timestamp))


def find_file(filename, search_path):

    for _, _, files in os.walk(search_path):
        if filename in files:
            logger.info(f"Found file {filename}!")
            return True
        else:
            logger.info("Didn't find any matching file.")
            return False


def save_file(
    data: pd.DataFrame,
    location: str,
    filename: str,
):

    os.makedirs(location, exist_ok=True)
    format = filename.split(".")[1]
    full_loc = location + filename

    try:
        match (format):
            case "csv":
                data.to_csv(full_loc, index=False)
            case "json":
                data.to_json(full_loc, orient="records", indent=2)
            case "parquet":
                data.to_parquet(full_loc, engine="pyarrow")
    except Exception as e:
        logger.info(f"An error occurred when saving : {e}")
    finally:
        logger.info(f"Successfully saved file {filename}")


def save_model(
    model: object,
    location: str,
    filename: str,
):

    os.makedirs(location, exist_ok=True)
    format = filename.split(".")[1]
    full_loc = location + filename

    try:
        match (format):
            case "pth":
                torch.save(model, full_loc)
    except Exception as e:
        logger.info(f"An error occurred when saving : {e}")
    finally:
        logger.info(f"Successfully Saved file {filename}")


def load_file(location: str, filename: str):

    if not find_file(filename, location):
        return None
    format = filename.split(".")[1]
    full_loc = location + filename
    try:
        match (format):
            case "csv":
                data = pd.read_csv(full_loc)
            case "json":
                data = pd.read_json(full_loc)
            case "parquet":
                data = pd.read_parquet(full_loc, engine="pyarrow")
            case "pt":
                data = torch.load(full_loc)
            case _:
                logger.info(f"Unable to load file {filename}, returning None.")
                logger.info(f"Format not supported : {format}.")
                data = None

        return data
    except Exception as e:
        logger.info(f"An error occurred when saving : {e}")
    finally:
        logger.info(f"Successfully loaded file {filename}")

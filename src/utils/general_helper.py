from datetime import datetime
import os

from src.utils.logger_config import get_logger

logger = get_logger("Helper", "general_helper.log")


def convert_unix_timestamp_to_date(ts):
    """Convert timestamp to actual date"""
    timestamp = int(ts)

    return str(datetime.fromtimestamp(timestamp))


def find_files(filename, search_path):

    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
        else:
            return False

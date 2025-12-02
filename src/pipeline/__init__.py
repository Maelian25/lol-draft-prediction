"""Pipeline helpers for scraping, analysis and training.

Expose the high-level functions used by `main.py` so the main script
remains small and easy to maintain.
"""

from .scraper import scrape_world_data
from .analysis import run_analysis
from .training import run_training

__all__ = ["scrape_world_data", "run_analysis", "run_training"]

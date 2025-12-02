import argparse
import os

from src.pipeline import scrape_world_data, run_analysis, run_training
from src.utils.logger_config import get_logger

logger = get_logger("Main", "main_file.log")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run scraping, analysis and training pipelines."
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip scraping/loading datasets (use existing data).",
    )
    parser.add_argument("--no-train", action="store_true", help="Skip training phase.")
    parser.add_argument("--no-analysis", action="store_true", help="Skip analysis.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max workers for scraping threads.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of preprocessed datasets before training.",
    )
    parser.add_argument(
        "--model",
        choices=["transformer", "mlp"],
        default="transformer",
        help="Model to train: 'transformer' (default) or 'mlp'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        logger.info("Starting pipeline")

        df_world = scrape_world_data(
            max_workers=args.max_workers, no_scrape=args.no_scrape
        )

        matches_states = run_analysis(df_world, no_analysis=args.no_analysis)

        if not args.no_train:
            run_training(matches_states, rebuild=args.rebuild, model_choice=args.model)
        else:
            logger.info("Training skipped (--no-train)")

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

    finally:
        logger.info("Pipeline finished")
        os._exit(0)


if __name__ == "__main__":
    main()

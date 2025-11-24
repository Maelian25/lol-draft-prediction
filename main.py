from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List

import numpy as np
import pandas as pd

from src.ML_models.draft_MLP import DraftMLPModel
from src.ML_training.trainer import TrainerClass
from src.ML_training.utils import preprocess_and_save
from src.analysis.dataset_analysis import DatasetAnalysis
from src.data_scrapping.dataset import Dataset
from src.utils.champions_helper import champName_to_champId
from src.utils.constants import DATASETS_FOLDER, ELOS, REGIONS, SAVE_AFTER_ITERATION
from src.utils.data_helper import (
    load_scrapped_data,
)
from src.utils.logger_config import get_logger

from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger("Main", "main_file.log")


def get_regional_data_from_api_threaded(region, short_name):
    """
    Provide regional data given a region.
    Necessary to thread requests
    """
    region_logger = get_logger(region, f"data_scrapping_{region.lower()}.log")
    region_logger.info(f"Starting region {region}")

    regional_dataframes: List[pd.DataFrame] = []

    for elo, player_count in ELOS.items():
        region_save_dir = os.path.join(DATASETS_FOLDER, short_name)
        os.makedirs(region_save_dir, exist_ok=True)

        region_logger.info(f"Started scrapping {region}-{elo}")

        df_region, data_scrapped = load_scrapped_data(region_save_dir, short_name, elo)

        if not data_scrapped:
            # Creating dataset
            dataset = Dataset(
                region=region,
                queue="RANKED_SOLO_5x5",
                game_count=100,
                player_count=player_count,
                elo=elo,
                save_after_iteration=SAVE_AFTER_ITERATION,
            )

            # Extracting and save data
            matches = dataset.extract_match_data(short_name)
            df_region = pd.DataFrame(matches)

        regional_dataframes.append(df_region)

        region_logger.info(f"Finished scrapping {region}-{elo}")
    region_logger.info(f"Finished scrapping {region}")
    return pd.concat(regional_dataframes, ignore_index=True)


if __name__ == "__main__":

    try:
        # --- Scrapping data ---
        # Parallelism
        with ThreadPoolExecutor(max_workers=len(REGIONS)) as executor:
            futures = {
                executor.submit(
                    get_regional_data_from_api_threaded, region, short_name
                ): region
                for region, short_name in REGIONS.items()
            }

            df_world_list: List[pd.DataFrame] = []
            for future in as_completed(futures):
                region = futures[future]
                try:
                    df_regional = future.result()
                    df_world_list.append(df_regional)
                    logger.info(f"Region {region} succesfully completed")
                except Exception as e:
                    logger.error(f"Region {region} failed: {e}", exc_info=True)

        if not df_world_list:
            raise RuntimeError("No data was successfully loaded or scraped.")

        df_world: pd.DataFrame = pd.concat(df_world_list, ignore_index=True)

        # Analysing patches on the dataset
        patches = ["15.22"]
        analysis_patch = DatasetAnalysis(df_world, patches)

        logger.info("Global data analysis on the dataset")

        analysis_patch.global_analysis.get_game_duration_stats(plot=True)
        analysis_patch.global_analysis.get_patch_distribution(plot=True)
        analysis_patch.global_analysis.get_win_rate_per_side()

        logger.info("Analysing data for one champion : Kai'Sa")

        champ_to_analyse = champName_to_champId("kai'sa")
        pick_rate = analysis_patch.champion_stats.get_champ_pick_or_ban_rate(
            pick=True, plot=True
        )
        ban_rate = analysis_patch.champion_stats.get_champ_pick_or_ban_rate(
            pick=False, plot=True
        )

        logger.info(f"Pick rate for Kai'Sa: {pick_rate[champ_to_analyse] * 100:.2f}%")
        logger.info(f"Ban rate for Kai'Sa: {ban_rate[champ_to_analyse] * 100:.2f}%")

        logger.info("Analysing one team comp")
        team_comp_to_analyse = ["ornn", "lee sin", "orianna", "yunara", "nautilus"]
        team_comp_to_analyse_ids = [
            champName_to_champId(champ) for champ in team_comp_to_analyse
        ]

        synergy_matrix = analysis_patch.synergy.compute_synergy_matrix()
        analysis_patch.synergy.get_synergy(
            synergy_matrix=synergy_matrix,
            champ_id_1=team_comp_to_analyse_ids[0],
            champ_id_2=team_comp_to_analyse_ids[1],
            log=True,
        )
        analysis_patch.synergy.team_synergy_score(
            synergy_matrix=synergy_matrix, team_champs=team_comp_to_analyse_ids
        )

        logger.info("Checking champion embeddings and analysing similiraty btw champs")

        champ_embeddings = analysis_patch.get_champion_embeddings()

        ahri = "Ahri"
        syndra = "Syndra"
        malphite = "Malphite"

        cos_sim_1 = cosine_similarity(
            np.array(champ_embeddings[champName_to_champId(ahri)].values).reshape(
                1, -1
            ),
            np.array(champ_embeddings[champName_to_champId(syndra)].values).reshape(
                1, -1
            ),
        )

        cos_sim_2 = cosine_similarity(
            np.array(champ_embeddings[champName_to_champId(ahri)].values).reshape(
                1, -1
            ),
            np.array(champ_embeddings[champName_to_champId(malphite)].values).reshape(
                1, -1
            ),
        )

        logger.info(f"Similarity btw {ahri} and {syndra} : {cos_sim_1[0][0]:.2f}")
        logger.info(f"Similarity btw {ahri} and {malphite} : {cos_sim_2[0][0]:.2f}")

        logger.info("Analysis ended for the restricted dataset")
        matches_states = analysis_patch.build_matches_states()

        # Process matches to be torch ready and load faster for training
        preprocess_and_save(matches_states, rebuild=False)

        mlp_model = DraftMLPModel(
            num_champions=171,
            num_roles=5,
            mode="learnable",
            embed_size=128,
            hidden_dim=1024,
            num_res_blocks=3,
            dropout=0.4,
        )

        trainer = TrainerClass(
            model=mlp_model,
            batch_size=512,
            num_epochs=20,
            base_lr=2e-4,
            weight_decay=1e-3,
            experiment_name="draft_mlp_v2",
        )

        trainer.train()

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise

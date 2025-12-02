import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

import pandas as pd

from src.analysis.dataset_analysis import DatasetAnalysis
from src.utils.champions_helper import champName_to_champId
from src.utils.logger_config import get_logger

logger = get_logger("analysis", "analysis.log")


def run_analysis(df_world: pd.DataFrame,
                 patches: Optional[List[str]] = None,
                 no_analysis : bool = False) -> pd.DataFrame:
    """Run dataset analysis and return the processed match states for training."""
    if patches is None:
        patches = ["15.22"]

    analysis_patch = DatasetAnalysis(df_world, patches)

    if no_analysis:
        logger.info("Skipping analysis as per --no-analysis flag")
        matches_states = analysis_patch.build_matches_states()
        return matches_states

    logger.info("Global data analysis on the dataset")
    try:
        analysis_patch.global_analysis.get_game_duration_stats(plot=True)
        analysis_patch.global_analysis.get_patch_distribution(plot=True)
        analysis_patch.global_analysis.get_win_rate_per_side()
    except Exception:
        logger.exception("Error while running global analysis")

    champ_to_analyse = champName_to_champId("kai'sa")
    try:
        pick_rate = (
            analysis_patch.champion_stats.get_champ_pick_or_ban_rate(
                pick=True, plot=True
            )
        )
        ban_rate = (
            analysis_patch.champion_stats.get_champ_pick_or_ban_rate(
                pick=False, plot=True
            )
        )

        logger.info(
            f"Pick rate for Kai'Sa: {pick_rate[champ_to_analyse] * 100:.2f}%"
        )
        logger.info(
            f"Ban rate for Kai'Sa: {ban_rate[champ_to_analyse] * 100:.2f}%"
        )
    except Exception:
        logger.exception("Error while computing champion pick/ban rates")

    # Example team comp analysis (kept as-is — adjust champions as needed)
    team_comp_to_analyse = ["ornn", "lee sin", "orianna", "yunara", "nautilus"]
    team_comp_to_analyse_ids = [
        champName_to_champId(champ) for champ in team_comp_to_analyse
    ]

    try:
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
    except Exception:
        logger.exception("Error while computing synergy / team score")

    # Champion embeddings similarity check
    try:
        champ_embeddings = analysis_patch.get_champion_embeddings()

        def _cos_sim(a, b):
            return (
                cosine_similarity(
                    np.array(a.values).reshape(1, -1),
                    np.array(b.values).reshape(1, -1),
                )[0][0]
            )

        ahri = "Ahri"
        syndra = "Syndra"
        malphite = "Malphite"

        ahri_id = champName_to_champId(ahri)
        syndra_id = champName_to_champId(syndra)
        malphite_id = champName_to_champId(malphite)

        cos_sim_1 = _cos_sim(champ_embeddings[ahri_id], champ_embeddings[syndra_id])
        cos_sim_2 = _cos_sim(champ_embeddings[ahri_id], champ_embeddings[malphite_id])

        logger.info(f"Similarity btw {ahri} and {syndra} : {cos_sim_1:.2f}")
        logger.info(f"Similarity btw {ahri} and {malphite} : {cos_sim_2:.2f}")
    except Exception:
        logger.exception("Error while computing champion embeddings similarity")

    logger.info("Analysis ended for the restricted dataset")
    matches_states = analysis_patch.build_matches_states()
    return matches_states

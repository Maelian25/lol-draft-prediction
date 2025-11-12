import itertools
import math
from typing import NoReturn, cast
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.analysis.base_analysis import BaseAnalysis
from src.utils.constants import MATRICES_FOLDER, SYN_MAT
from src.utils.general_helper import load_file, save_file


class SynergyAnalysis(BaseAnalysis):
    """
    Computation of synergy matrix and similarity between champs
    """

    def compute_synergy_matrix(self, plot=False, alpha=1) -> pd.DataFrame:
        """
        Compute synergy matrix based on winrate optimized and returns a dataframe

        Args:
            plot: Whether to plot or not the heatmap of the matrix
            alpha: Parameter to compute Laplace smoothing

        """
        data = load_file(MATRICES_FOLDER, SYN_MAT)

        if data is not None:
            return data

        game_count_matrix = np.zeros((self.n_champs, self.n_champs), dtype=np.float32)
        game_won_count_matrix = np.zeros(
            (self.n_champs, self.n_champs), dtype=np.float32
        )

        for match in self.dataset.itertuples():

            picks = getattr(match, "picks", [])
            if not picks:
                continue

            blue_team_champs = [
                self.champ_id_to_idx_map[p["championId"]]
                for p in picks
                if p["side"] == "blue"
            ]
            red_team_champs = [
                self.champ_id_to_idx_map[p["championId"]]
                for p in picks
                if p["side"] == "red"
            ]

            blue_team_won = getattr(match, "blue_side_win", bool)

            for champs, won in [
                (blue_team_champs, blue_team_won),
                (red_team_champs, not blue_team_won),
            ]:
                team_representation = np.zeros((self.n_champs,), dtype=np.float32)
                team_representation[champs] = 1.0

                outer = np.outer(team_representation, team_representation)
                np.fill_diagonal(outer, 0)

                game_count_matrix += outer
                if won:
                    game_won_count_matrix += outer

        synergy_matrix = (game_won_count_matrix + alpha) / (
            game_count_matrix + 2 * alpha
        )

        synergy_matrix = np.nan_to_num(synergy_matrix, nan=0.5)

        symetry_df = pd.DataFrame(
            synergy_matrix,
            index=list(self.champ_id_to_idx_map.keys()),
            columns=list(self.champ_id_to_idx_map.keys()),
        )

        save_file(symetry_df, MATRICES_FOLDER, SYN_MAT)

        if plot:
            plt.figure(figsize=(12, 10))
            sns.heatmap(synergy_matrix, cmap="coolwarm", center=0.5)
            plt.title("Champion Synergy Matrix (Winrate)")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        return symetry_df

    def get_synergy(
        self, synergy_matrix: pd.DataFrame, champ_id_1: int, champ_id_2: int, log=False
    ):
        """Returns synergy between two given champ ids"""

        if synergy_matrix is NoReturn:
            return

        synergy_btw_champs = cast(float, synergy_matrix.loc[champ_id_1, champ_id_2])

        if log:
            self.logger.info(
                f"Synergy between {self.champ_id_name_map[champ_id_1]} "
                f"and {self.champ_id_name_map[champ_id_2]} : "
                f"{synergy_btw_champs * 100:.2f}%"
            )

        return synergy_btw_champs

    def team_synergy_score(self, synergy_matrix: pd.DataFrame, team_champs, log=True):
        """Returns team synergy score"""
        team_synergy = 0.0
        team_champs_cleaned = [champs for champs in team_champs if champs != 0]
        if len(team_champs_cleaned) > 1:
            pairs = itertools.combinations(team_champs_cleaned, 2)
            for c in pairs:
                champ_id_1 = c[0]
                champ_id_2 = c[1]
                team_synergy += cast(float, synergy_matrix.loc[champ_id_1, champ_id_2])

            team_synergy /= math.comb(len(team_champs_cleaned), 2)

        if log:
            self.logger.info(f"The team synergy is about {team_synergy * 100:.2f}%")

        return team_synergy

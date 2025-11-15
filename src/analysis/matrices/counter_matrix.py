import itertools
import os
from typing import List
import numpy as np
import pandas as pd
import torch
from src.ML_models.counter_matrix_model import BTFeatureCounter
from src.analysis.base_analysis import BaseAnalysis
from src.utils.constants import (
    BT_MODEL,
    COUNT_MAT,
    MATRICES_FOLDER,
    MODELS_PARAMETER_FOLDER,
)
from src.utils.general_helper import find_file, load_file, save_file


class CounterAnalysis(BaseAnalysis):
    """
    Computation of counter matrix
    """

    def __wins_vs_counter(self):
        """
        Returns count of wins between every 2 champs
        """
        matrix = np.zeros((self.n_champs, self.n_champs), dtype=np.float32)

        for match in self.dataset.itertuples():
            picks = getattr(match, "picks", [])

            blue_team_champs = [
                self.champ_id_to_idx_map[str(p["championId"])]
                for p in picks
                if p["side"] == "blue"
            ]
            red_team_champs = [
                self.champ_id_to_idx_map[str(p["championId"])]
                for p in picks
                if p["side"] == "red"
            ]

            blue_team_won = getattr(match, "blue_side_win", False)

            for i in blue_team_champs:
                for j in red_team_champs:
                    if blue_team_won:
                        matrix[i, j] += 1
                    else:
                        matrix[j, i] += 1

        return matrix

    def __counter_win_rate_matrix(self, matrix) -> pd.DataFrame:
        """
        Returns dict to have every data ready to train our model

        Args:
            matrix -> matrix containing wins counter per duo
        """

        pairs_data = []
        alpha, beta = 1.0, 1.0
        for i, j in itertools.combinations(range(self.n_champs), 2):
            if i != j:
                n = matrix[i, j] + matrix[j, i]
                smoothed_wr = (matrix[i, j] + alpha) / (n + alpha + beta)
                pairs_data.append(
                    (
                        i,
                        j,
                        self.idx_to_champ_id_map[i],
                        self.idx_to_champ_id_map[j],
                        smoothed_wr,
                        n,
                    )
                )

        df_cross_champ_wr = pd.DataFrame(
            pairs_data,
            columns=[
                "champ_idx_1",
                "champ_idx_2",
                "champ_id_1",
                "champ_id_2",
                "target",
                "weight",
            ],
        )

        return df_cross_champ_wr

    def __champions_features(self) -> dict[int, List[float]]:
        """
        Create an embedding based on the basic infos of the champions
        Passing this object through model to define counter_matrix
        """

        champ_features = {
            champ_id: self.get_champion_infos(champ_id)
            for champ_id in self.champ_id_name_map.keys()
        }

        all_stats = np.vstack(list(champ_features.values()))
        scaled_all_stats = self.scaler.fit_transform(all_stats)

        champ_features = {
            champ_id: scaled_all_stats[i].tolist()
            for i, champ_id in enumerate(self.champ_id_name_map.keys())
        }

        return champ_features

    def compute_counter_matrix(self):
        """
        Compute counter matrix based on winrate and a small machine learning model
        to finetune values based on Bradley–Terry model

        Provide the probability of a champ to win against another
        """
        data = load_file(MATRICES_FOLDER, COUNT_MAT)

        if data is not None:
            return data

        bt_counter = BTFeatureCounter(
            input_dim=30, num_champs=self.n_champs, embed_dim=32, device="cpu"
        )
        champ_features = self.__champions_features()

        if find_file(filename=BT_MODEL, search_path=MODELS_PARAMETER_FOLDER):
            bt_counter.model.load_state_dict(
                torch.load(
                    os.path.join(MODELS_PARAMETER_FOLDER, BT_MODEL),
                    weights_only=True,
                )
            )
        else:
            self.logger.info("No parameter file found. Training model")

            wins_vs_matrix = self.__wins_vs_counter()
            pairs_data_df = self.__counter_win_rate_matrix(wins_vs_matrix)

            X_1 = torch.tensor(
                np.stack(
                    [
                        champ_features[r["champ_id_1"]]
                        for _, r in pairs_data_df.iterrows()
                    ]
                ),
                dtype=torch.float32,
            )
            X_2 = torch.tensor(
                np.stack(
                    [
                        champ_features[r["champ_id_2"]]
                        for _, r in pairs_data_df.iterrows()
                    ]
                ),
                dtype=torch.float32,
            )
            idx_1 = torch.tensor(pairs_data_df["champ_idx_1"].values, dtype=torch.long)
            idx_2 = torch.tensor(pairs_data_df["champ_idx_2"].values, dtype=torch.long)

            target = torch.tensor(pairs_data_df["target"].values, dtype=torch.float32)
            weight = torch.tensor(pairs_data_df["weight"].values, dtype=torch.float32)
            weight = torch.clamp(torch.sqrt(weight), 0.01, 50)
            weight = weight / weight.mean()

            bt_counter.train(
                X_1, X_2, idx_1, idx_2, target, weight, num_epochs=2500, lr=3e-3
            )

        counter_matrix_df = bt_counter.evaluate(
            champ_features, self.champ_id_to_idx_map
        )
        save_file(counter_matrix_df, MATRICES_FOLDER, COUNT_MAT)

        return counter_matrix_df

    def team_counter_score(self, counter_matrix: pd.DataFrame, blue_team, red_team):
        counter_score = 0.0
        blue_team_cleaned = [champs for champs in blue_team if champs != 0]
        red_team_cleaned = [champs for champs in red_team if champs != 0]
        i = 0

        if len(blue_team_cleaned + red_team_cleaned) < 2:
            return counter_score
        for blue_champ in blue_team_cleaned:
            for red_champ in red_team_cleaned:
                i += 1
                counter_score += counter_matrix[blue_champ][red_champ]

        counter_score /= i

        return counter_score

    def top_10_matchups_for(self, champ_id, plot=False):

        counter_map = self.compute_counter_matrix()
        top_counters = {}

        champ_id_counters = dict(counter_map.loc[champ_id])
        del champ_id_counters[champ_id]

        sorted_counters = sorted(champ_id_counters.items(), key=lambda x: x[1])

        if plot:
            champ_name = self.champ_id_name_map[champ_id]

            for label, matchup_list in [
                (
                    f"Top 10 hardest 1v1 for {champ_name}:",
                    sorted(sorted_counters[:10], key=lambda x: x[1]),
                ),
                (
                    f"Top 10 easiest 1v1 for {champ_name}:",
                    sorted(sorted_counters[-10:], key=lambda x: x[1], reverse=True),
                ),
            ]:
                self.logger.info(label)
                for counter_id, win_rate in matchup_list:
                    counter_name = self.champ_id_name_map[counter_id]
                    self.logger.info(
                        f"  vs {counter_name} : {win_rate*100:.2f}% win rate"
                    )

        return pd.DataFrame(top_counters)

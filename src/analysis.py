from collections import Counter, defaultdict
import itertools
import math
from typing import Dict, List, Any, NoReturn, Optional, cast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import torch
import os

from src.utils.data_helper import (
    get_champions_id_name_map,
    replace_wrong_position,
)
from src.utils.champions_helper import (
    champ_id_to_idx_map,
    champName_to_champId,
    get_champions_data,
    tags_one_hot_encoder,
    unique_tags,
)
from src.utils.general_helper import find_files
from src.utils.logger_config import get_logger
from src.ML_models.counter_matrix_model import BTFeatureCounter

ROLE_MAP = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "SUPPORT": 5}


class DatasetAnalysis:
    """Enables analysis on a given dataset"""

    def __init__(
        self,
        dataset: pd.DataFrame,
        compute_matrices=False,
        patches: Optional[List[str]] = None,
    ) -> None:

        self.logger = get_logger("DatasetAnalysis", "analysis.log")
        self.dataset = dataset.copy()

        if patches:
            self.logger.info(f"Analysing dataset for patch {'-'.join(patches)}")
            self.dataset = self.dataset[self.dataset["game_version"].isin(patches)]
            self.patches = patches
        else:
            self.logger.info("Analysing full dataset")
            self.patches = None

        self.dataset = self.dataset.drop_duplicates(
            subset=["match_id"], ignore_index=True
        ).dropna()

        # Convert game_duration to numeric, coercing errors to NaN
        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )

        # Replace wrong positions
        self.dataset = replace_wrong_position(self.dataset)

        # Load champion name and ID maps
        self.champ_id_name_map = get_champions_id_name_map()
        self.champ_name_id_map = {v: k for k, v in self.champ_id_name_map.items()}

        # Load champion ID to index map and reciprocal
        self.champ_id_to_idx_map = champ_id_to_idx_map()
        self.idx_to_champ_id_map = {v: k for k, v in self.champ_id_to_idx_map.items()}

        # Calculate number of matches and unique champions
        self.num_matches = len(self.dataset)
        self.unique_champs = len(self.champ_id_name_map)

        # Get champions data once and for all
        self.champions_data = get_champions_data()

        # Get encoded tags for champions
        self.unique_tags = unique_tags()
        self.tags_encoder = tags_one_hot_encoder(self.unique_tags)

        self.champions_info_scaler = preprocessing.StandardScaler()

        # Compute synergy matrix and counter matrix once and for all
        if compute_matrices:
            self._precompute_matrices()

    def _precompute_matrices(self):
        """Compute synergy matrix and counter matrix"""

        self.synergy_matrix = self._compute_synergy_matrix()
        self.counter_matrix = self._compute_counter_matrix()

    # --- Global stats ---
    def get_win_rate_per_side(self):
        """Provide win rate for each side"""
        blue_side_win = self.dataset["blue_side_win"].value_counts(normalize=True)

        self.logger.info(
            f"The blue side win rate in this dataset is {blue_side_win[True]*100:.3f}%"
        )
        self.logger.info(
            f"The red side win rate in this dataset is {blue_side_win[False]*100:.3f}%"
        )

        return blue_side_win

    def get_game_duration_stats(self, plot: bool):
        """Provide game duration stats through the dataset"""
        stats = self.dataset["game_duration"].describe()
        self.logger.info(f"Average game time : {stats["mean"]:.2f}")

        # Creation of a figure to analyze game time
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            fig.suptitle("Game time analysis", fontsize=16)

            # Histo + density
            ax = axes[0]
            self.dataset["game_duration"].plot(
                kind="hist",
                bins=30,
                density=True,
                alpha=0.6,
                color="skyblue",
                edgecolor="black",
                ax=ax,
            )
            self.dataset["game_duration"].plot(kind="kde", color="darkblue", ax=ax)
            ax.set_title("Global distribution")
            ax.set_xlabel("Game time (minutes)")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)

            # Density per side
            ax = axes[1]
            self.dataset[self.dataset["blue_side_win"]]["game_duration"].plot(
                kind="kde", label="Blue wins", color="blue", ax=ax
            )
            self.dataset[~self.dataset["blue_side_win"]]["game_duration"].plot(
                kind="kde", label="Red wins", color="red", ax=ax
            )
            ax.set_title("Distribution per side winner")
            ax.set_xlabel("Game time (minutes)")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        return {
            "count": int(stats["count"]),
            "mean": round(stats["mean"], 2),
            "min": round(stats["min"], 2),
            "max": round(stats["max"], 2),
            "std": round(stats["std"], 2),
        }

    def get_patch_distribution(self):
        """Provide games distribution through patches"""
        stats = self.dataset["game_version"].describe()

        self.logger.info(
            f"There is a total of {stats["unique"]} patches in this dataset"
        )
        self.logger.info(
            f"The patch on which most games has been played is {stats["top"]}"
        )
        self.logger.info(f"Total games on patch {stats["top"]} : {stats["freq"]}")

        patch_counts = self.dataset["game_version"].value_counts().sort_index()
        patch_counts.plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)

        plt.title("Patch distribution")
        plt.xlabel("Patch")
        plt.ylabel("Game number")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        return stats["top"]

    # --- Champion stats ---
    def get_champ_win_rate(self):
        """Provide champ win rate"""
        win_rate_dict = defaultdict(Counter)

        for row in self.dataset.itertuples():
            picks: List[Dict[str, Any]] = getattr(row, "picks", [{}])
            blue_side_win = getattr(row, "blue_side_win", bool)

            for ch in picks:
                champion_id = ch["championId"]
                side = ch["side"]
                if blue_side_win:
                    if side == "blue":
                        win_rate_dict[champion_id]["win"] += 1
                    else:
                        win_rate_dict[champion_id]["lose"] += 1
                else:
                    if side == "blue":
                        win_rate_dict[champion_id]["lose"] += 1
                    else:
                        win_rate_dict[champion_id]["win"] += 1

        data = []

        for champ_id, counts in win_rate_dict.items():
            total_games = counts["win"] + counts["lose"]
            win_rate = counts["win"] / total_games if total_games > 0 else 0
            data.append(
                {
                    "championId": champ_id,
                    "games": total_games,
                    "wins": counts["win"],
                    "losses": counts["lose"],
                    "win_rate": float(win_rate),
                }
            )

        df_result = (
            pd.DataFrame(data)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
            .set_index("championId")
        )

        return df_result["win_rate"]

    def get_champ_pick_or_ban_rate(self, pick: bool, plot=False):
        """Provide champ pick or ban rate"""

        champ_rates = dict()

        for row in self.dataset.itertuples():
            if pick:
                champs_data: List[Dict[str, Any]] = getattr(row, "picks", [{}])
            else:
                champs_data: List[Dict[str, Any]] = getattr(row, "bans", [{}])

            for ch in champs_data:
                current_champ = ch["championId"]
                if current_champ not in champ_rates:
                    champ_rates[current_champ] = 1
                else:
                    champ_rates[current_champ] += 1
        champ_rates.update((x, y / self.num_matches) for x, y in champ_rates.items())

        highest_rate_id = max(champ_rates, key=(lambda key: champ_rates[key]))
        lowest_rate_id = min(champ_rates, key=(lambda key: champ_rates[key]))

        highest_rate_champ = self.champ_id_name_map[highest_rate_id]
        lowest_rate_champ = self.champ_id_name_map[lowest_rate_id]

        if plot:

            self.logger.info(
                f"{highest_rate_champ} has the highest "
                f"{"pick" if pick else "ban"} rate "
                f"with {champ_rates[highest_rate_id]*100:.3f}%"
            )
            self.logger.info(
                f"{lowest_rate_champ} has the lowest "
                f"{"pick" if pick else "ban"} rate "
                f"with {champ_rates[lowest_rate_id]*100:.3f}%"
            )

        return champ_rates

    def get_role_distribution(self, champ: str | int | None = None, plot=False):
        """Provide role distribution for a champ given a patch"""
        role_counts = defaultdict(Counter)

        for row in self.dataset.itertuples():

            champs_data: List[Dict[str, Any]] = getattr(row, "picks", [{}])

            for ch in champs_data:
                champion_id = ch["championId"]
                role = ch.get("position") or "SUPPORT"
                role_counts[champion_id][role] += 1

        df_role_counts = pd.DataFrame(role_counts).fillna(0).astype(int).T

        if not champ:
            self.logger.info("Returning the whole table...")
            return df_role_counts

        if isinstance(champ, str):
            champ_id = champName_to_champId(champ)
            champ_name = champ
        else:
            champ_id = champ
            champ_name = self.champ_id_name_map[champ_id]

        if champ_id not in df_role_counts.index:
            self.logger.warning(f"Champ '{champ_name}' not found in the dataset")
            return df_role_counts

        df_champ_role = df_role_counts.loc[[champ_id]].T
        if plot:
            df_champ_role.plot(
                kind="bar", color="skyblue", edgecolor="black", alpha=0.7
            )
            plt.title(f"Role distribution for {champ_name.capitalize()}")
            plt.xlabel("Roles")
            plt.ylabel("Number of game")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        df_champ_role.rename(index=ROLE_MAP).sort_index()

        role_distribution_percentage = list(
            map(
                lambda x: x / sum(list(df_champ_role[champ_id])),
                list(df_champ_role[champ_id]),
            )
        )

        return role_distribution_percentage

    # --- Matchup / synergy ---
    def get_counters(self, champ, plot):
        counter_map: dict[int, dict[int, dict[str, int]]] = defaultdict(dict)
        # {champ : {counter_pick : {games_played_versus, game_won_versus}
        for row in self.dataset.itertuples():
            picks = getattr(row, "picks", [{}])

            blue_side_picks = [pick for pick in picks if pick["side"] == "blue"]
            red_side_picks = [pick for pick in picks if pick["side"] == "red"]

            blue_side_win = True if getattr(row, "blue_side_win", bool) else False

            for pick in picks:

                champ_id = pick["championId"]
                pick_position = pick["position"]
                pick_side = pick["side"]
                counter_pick = -1
                won = 0

                if pick_side == "blue":
                    if blue_side_win:
                        won = 1
                    for x in red_side_picks:
                        if x["position"] == pick_position:
                            counter_pick = x["championId"]
                else:
                    if not blue_side_win:
                        won = 1
                    for x in blue_side_picks:
                        if x["position"] == pick_position:
                            counter_pick = x["championId"]

                counter_map[champ_id][counter_pick] = {
                    "games_played": counter_map[champ_id]
                    .get(counter_pick, {})
                    .get("games_played", 0)
                    + 1,
                    "games_won_against": counter_map[champ_id]
                    .get(counter_pick, {})
                    .get("games_won_against", 0)
                    + won,
                }
        if plot:
            top_counters = {}

            for champ_id, counter_data in counter_map.items():

                sorted_counter = sorted(
                    counter_data.items(),
                    key=lambda x: x[1]["games_played"],
                    reverse=True,
                )

                top_counters[champ_id] = sorted(
                    sorted_counter[: int(len(sorted_counter) * 0.2)],
                    key=lambda x: x[1]["games_won_against"] / x[1]["games_played"],
                )

            champ_name = self.champ_id_name_map[champ]
            self.logger.info(f"Top 20% counters for {champ_name}:")
            for opp_id, stats in top_counters[champ]:
                opp_name = self.champ_id_name_map[opp_id]
                self.logger.info(
                    f"  vs {opp_name}: {stats['games_played']} games, "
                    f"{stats['games_won_against']} wins, "
                    f"{stats['games_won_against']/stats['games_played'] * 100:.2f}% "
                    "win rate"
                )

        new_counter_map: dict[int, dict[int, float]] = defaultdict(dict)
        for champ_id, counter_data in counter_map.items():

            for counter_id, stats in counter_data.items():
                new_counter_map[champ_id][counter_id] = (
                    float(stats["games_won_against"] / stats["games_played"])
                    if stats["games_played"] > 0
                    else 0
                )

        return pd.DataFrame(new_counter_map)

        return pd.DataFrame(new_counter_map)

    def get_synergy(self, champ_id_1: int, champ_id_2: int, log=False):
        """Returns synergy between two given champ ids"""

        if self.synergy_matrix is NoReturn:
            return

        synergy_btw_champs = cast(
            float, self.synergy_matrix.loc[champ_id_1, champ_id_2]
        )

        if log:
            self.logger.info(
                f"Synergy between {self.champ_id_name_map[champ_id_1]} "
                f"and {self.champ_id_name_map[champ_id_2]} : "
                f"{synergy_btw_champs * 100:.2f}%"
            )

        return synergy_btw_champs

    def get_team_synergy(self, team_champs: List[int]):
        """Returns team synergy score"""
        team_synergy = 0
        pairs = itertools.combinations(team_champs, 2)
        for c in pairs:
            champ_id_1 = c[0]
            champ_id_2 = c[1]
            team_synergy += cast(float, self.synergy_matrix.loc[champ_id_1, champ_id_2])

        team_synergy /= math.comb(len(team_champs), 2)

        self.logger.info(f"The team synergy is about {team_synergy * 100:.2f}%")

        return team_synergy

    # --- Draft analysis ---
    def get_first_pick_stats(
        self,
    ):
        """Provide first and last pick stats"""
        fp_rates = {"fp": defaultdict(int), "lp": defaultdict(int)}

        for row in self.dataset.itertuples():
            for pick in getattr(row, "picks", [{}]):
                champ_id = pick.get("championId")
                side = pick.get("side")
                order = pick.get("order")

                if not champ_id or not side or not order:
                    continue

                # First pick = blue side, order 1
                if side == "blue" and order == 1:
                    fp_rates["fp"][champ_id] += 1

                # Last pick = red side, order 5
                elif side == "red" and order == 5:
                    fp_rates["lp"][champ_id] += 1

        fp_df = pd.DataFrame(
            [
                {
                    "championId": champ_id,
                    "first_pick_count": fp_rates["fp"].get(champ_id, 0),
                    "last_pick_count": fp_rates["lp"].get(champ_id, 0),
                }
                for champ_id in set(
                    list(fp_rates["fp"].keys()) + list(fp_rates["lp"].keys())
                )
            ]
        )

        fp_df["first_pick_rate"] = fp_df["first_pick_count"] / self.num_matches
        fp_df["last_pick_rate"] = fp_df["last_pick_count"] / self.num_matches

        for x in ["first_pick_rate", "last_pick_rate"]:
            fp_df = fp_df.sort_values(x, ascending=False).reset_index(drop=True)

            highest_rate_id = fp_df.iloc[0]["championId"]
            highest_rate = fp_df.iloc[0][x]

            champ_name = self.champ_id_name_map[highest_rate_id]

            self.logger.info(
                f"{champ_name} has the highest {" ".join(x.split("_"))} "
                f"with {highest_rate * 100:.3f}%"
            )

        return fp_df.sort_values("first_pick_rate", ascending=False).reset_index(
            drop=True
        )

    def get_draft_order_correlation(self):
        """Provide some correlations between order and win rate"""

        data = []

        for row in self.dataset.itertuples():
            blue_win = getattr(row, "blue_side_win", False)
            for pick in getattr(row, "picks", []):
                champ_id = pick["championId"]
                order = pick["order"]
                side = pick["side"]
                position = pick["position"]

                won = (blue_win and side == "blue") or (not blue_win and side == "red")

                data.append(
                    {
                        "championId": champ_id,
                        "order": order,
                        "side": side,
                        "win": 1 if won else 0,
                        "position": position,
                    }
                )

        df_corr = pd.DataFrame(data)
        df_blue = df_corr[df_corr["side"] == "blue"]
        df_red = df_corr[df_corr["side"] == "red"]
        grouped = df_corr.groupby(["position", "side"])

        correlation = df_corr["order"].corr(df_corr["win"])
        red_side__correlation = df_red["order"].corr(df_red["win"])
        blue_side__correlation = df_blue["order"].corr(df_blue["win"])

        self.logger.info(
            f"Overall correlation between pick order and win rate: {correlation:.4f}"
        )
        self.logger.info(f"Blue side correlation: {blue_side__correlation:.4f}")
        self.logger.info(f"Red side correlation: {red_side__correlation:.4f}")

        for (position, side), df_pos in grouped:
            pos_corr = df_pos["order"].corr(df_pos["win"])
            self.logger.info(
                "Correlation "
                f"considering {position} and {side} side : {pos_corr*100:.2f}%"
            )

        return correlation

    # --- Feature generation ---
    def champion_embeddings(self):

        win_rate = self.get_champ_win_rate()
        pick_rate = self.get_champ_pick_or_ban_rate(pick=True)
        ban_rate = self.get_champ_pick_or_ban_rate(pick=False)

        champ_embeddings = {
            champ_id: np.concatenate(
                [
                    self._get_champion_infos(champ_id),
                    np.array(
                        [win_rate[champ_id], pick_rate[champ_id], ban_rate[champ_id]]
                    ),
                    self.get_role_distribution(champ_id),
                    np.array(
                        [
                            self.counter_matrix[champ_id].mean(),
                            self.synergy_matrix[champ_id].mean(),
                        ]
                    ),
                ]
            )
            for champ_id in self.champ_id_name_map.keys()
        }

        all_stats = np.vstack(list(champ_embeddings.values()))
        scaled_all_stats = self.champions_info_scaler.fit_transform(all_stats)

        champ_embeddings = {
            champ_id: scaled_all_stats[i].tolist()
            for i, champ_id in enumerate(self.champ_id_name_map.keys())
        }

        champ_embeddings_df = pd.DataFrame(champ_embeddings).T
        champ_embeddings_df.to_json("champion_embeddings.json")

        return champ_embeddings_df

    def get_team_features(self, team_champs): ...

    def _get_champion_infos(self, champ_id: int):
        """
        Encode tags and returns every info needed in a list
        """

        tags_encoded = list(self.tags_encoder[champ_id].values())

        infos = list(self.champions_data[champ_id]["info"].values())
        stats = list(self.champions_data[champ_id]["stats"].values())

        # Concat all three lists
        stats_list = infos + stats + tags_encoded

        return stats_list

    # --- Synergy matrix computation ---
    def _compute_synergy_matrix(self, plot=False, alpha=1) -> pd.DataFrame:
        """
        Compute synergy matrix based on winrate optimized and returns a dataframe

        Args:
            plot: Whether to plot or not the heatmap of the matrix
            alpha: Parameter to compute Laplace smoothing

        """
        game_count_matrix = np.zeros(
            (self.unique_champs, self.unique_champs), dtype=np.float32
        )
        game_won_count_matrix = np.zeros(
            (self.unique_champs, self.unique_champs), dtype=np.float32
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
                team_representation = np.zeros((self.unique_champs,), dtype=np.float32)
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

        if plot:
            plt.figure(figsize=(12, 10))
            sns.heatmap(synergy_matrix, cmap="coolwarm", center=0.5)
            plt.title("Champion Synergy Matrix (Winrate)")
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        return symetry_df

    # --- Counters matrix computation ---
    def _count_wins_vs(self):
        """
        Returns count of wins between 2 champs
        """
        matrix = np.zeros((self.unique_champs, self.unique_champs), dtype=np.float32)

        for match in self.dataset.itertuples():
            picks = getattr(match, "picks", [])

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

            blue_team_won = getattr(match, "blue_side_win", False)

            for i in blue_team_champs:
                for j in red_team_champs:
                    if blue_team_won:
                        matrix[i, j] += 1
                    else:
                        matrix[j, i] += 1

        return matrix

    def _prepare_counter_matrix_data(self, matrix):
        """
        Returns dict to have every data ready to train our model
        """
        pairs_data = []
        alpha, beta = 1.0, 1.0
        for i, j in itertools.combinations(range(self.unique_champs), 2):
            if i != j:
                n = matrix[i, j] + matrix[j, i]
                smoothed_p = (matrix[i, j] + alpha) / (n + alpha + beta)
                pairs_data.append(
                    (
                        i,
                        j,
                        self.idx_to_champ_id_map[i],
                        self.idx_to_champ_id_map[j],
                        matrix[i, j],
                        matrix[j, i],
                        smoothed_p,
                        np.sqrt(n),
                    )
                )

        df = pd.DataFrame(
            pairs_data,
            columns=[
                "champ_idx_1",
                "champ_idx_2",
                "champ_id_1",
                "champ_id_2",
                "wins_1_vs_2",
                "wins_2_vs_1",
                "target",
                "weight",
            ],
        )
        df["weight"] /= df["weight"].mean()

        return df

    def _static_champions_embedding(self) -> dict[int, List[float]]:
        """
        Create an embedding based on the basic infos of the champions
        Passing this object through model to defin counter_matrix
        """

        champ_features = {
            champ_id: self._get_champion_infos(champ_id)
            for champ_id in self.champ_id_name_map.keys()
        }

        all_stats = np.vstack(list(champ_features.values()))
        scaled_all_stats = self.champions_info_scaler.fit_transform(all_stats)

        champ_features = {
            champ_id: scaled_all_stats[i].tolist()
            for i, champ_id in enumerate(self.champ_id_name_map.keys())
        }

        return champ_features

    def _compute_counter_matrix(self, plot=False):
        """
        Compute synergy matrix based on winrate and a small machine learning model
        to finetune values based on Bradley–Terry model
        Provide the probability of a champ to win against another
        """
        game_won_by_1_vs_2_matrix = self._count_wins_vs()
        pairs_data_df = self._prepare_counter_matrix_data(game_won_by_1_vs_2_matrix)
        champ_features = self._static_champions_embedding()

        X_1 = torch.tensor(
            np.stack(
                [champ_features[r["champ_id_1"]] for _, r in pairs_data_df.iterrows()]
            ),
            dtype=torch.float32,
        )
        X_2 = torch.tensor(
            np.stack(
                [champ_features[r["champ_id_2"]] for _, r in pairs_data_df.iterrows()]
            ),
            dtype=torch.float32,
        )
        idx_1 = torch.tensor(pairs_data_df["champ_idx_1"].values, dtype=torch.long)
        idx_2 = torch.tensor(pairs_data_df["champ_idx_2"].values, dtype=torch.long)

        target = torch.tensor(pairs_data_df["target"].values, dtype=torch.float32)
        weight = torch.tensor(pairs_data_df["weight"].values, dtype=torch.float32)

        bt_counter = BTFeatureCounter(
            input_dim=30, num_champs=self.unique_champs, embed_dim=12, device="cpu"
        )

        save_dir = os.getcwd() + "/models_parameter/"
        filename = "BTFeature_param.pth"

        if find_files(filename=filename, search_path=save_dir):
            bt_counter.model.load_state_dict(
                torch.load(os.path.join(save_dir, filename), weights_only=True)
            )
            self.logger.info("Find a parameter file. Loading file...")
        else:
            self.logger.info("No parameter file found. Training model")
            bt_counter.train(X_1, X_2, idx_1, idx_2, target, weight, num_epochs=1000)

        P_df = bt_counter.counter_matrix(
            champ_features, self.champ_id_to_idx_map
        ).round(3)

        return P_df

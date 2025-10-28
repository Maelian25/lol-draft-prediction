from collections import Counter, defaultdict
import itertools
import math
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.helper import (
    champ_id_to_idx_map,
    champId_to_champName,
    champName_to_champId,
    get_champions_id_name_dict,
    replace_wrong_position,
)
from src.logger_config import get_logger


class DatasetAnalysis:
    """Enables analysis on a given dataset"""

    def __init__(self, dataset: pd.DataFrame, patches: List[str]) -> None:
        self.dataset = dataset.drop_duplicates(subset=["match_id"], ignore_index=True)
        self.dataset = self.dataset.dropna()
        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )
        self.dataset = replace_wrong_position(self.dataset)

        self.num_matches = len(self.dataset)
        self.patches = patches
        self.logger = get_logger("Analysis", "data_analysis.log")
        self.logger.info(f"Analysing dataset for patch {"-".join(patches)}")

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

    def get_game_duration_stats(self):
        """Provide game duration stats through the dataset"""
        stats = self.dataset["game_duration"].describe()
        self.logger.info(f"Average game time : {stats["mean"]:.2f}")

        # Creation of a figure to analyze game time
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
        plt.show()

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
        plt.show()
        return stats["top"]

    def get_unique_champs_number(self):
        return len(get_champions_id_name_dict())

    # --- Champion stats ---
    def get_champ_win_rate(self):
        """Provide champ win rate on a patch"""
        win_rate_dict = defaultdict(Counter)
        for row in self.dataset[
            self.dataset["game_version"].isin(self.patches)
        ].itertuples():
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
                    "win_rate": win_rate * 100,
                }
            )

        df_result = (
            pd.DataFrame(data)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
        )

        return df_result

    def get_champ_pick_or_ban_rate(self, pick: bool):
        """Provide champ pick or ban rate given a patch"""
        champ_rates = dict()
        for row in self.dataset[
            self.dataset["game_version"].isin(self.patches)
        ].itertuples():
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

        highest_rate_champ = champId_to_champName(highest_rate_id)
        lowest_rate_champ = champId_to_champName(lowest_rate_id)

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

    def get_role_distribution(self, champ: str | int | None = None):
        """Provide role distribution for a champ given a patch"""
        role_counts = defaultdict(Counter)

        for row in self.dataset[
            self.dataset["game_version"].isin(self.patches)
        ].itertuples():

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
            champ_name = champId_to_champName(champ_id)

        if champ_id not in df_role_counts.index:
            self.logger.warning(f"Champ '{champ_name}' not found in the dataset")
            return df_role_counts

        df_champ_role = df_role_counts.loc[[champ_id]].T
        df_champ_role.plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"Role distribution for {champ_name.capitalize()}")
        plt.xlabel("Roles")
        plt.ylabel("Number of game")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return df_champ_role

    # --- Matchup / synergy ---
    def get_counters(self, champ):
        counter_map: dict[int, dict[int, dict[str, int]]] = defaultdict(dict)
        # {champ : {counter_pick : {games_played_versus, game_won_versus}
        for row in self.dataset[
            self.dataset["game_version"].isin(self.patches)
        ].itertuples():
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

        top5_counters = {}

        for champ_id, counter_data in counter_map.items():

            sorted_counter = sorted(
                counter_data.items(),
                key=lambda x: x[1]["games_played"],
                reverse=True,
            )

            top5_counters[champ_id] = sorted(
                sorted_counter[: int(len(sorted_counter) * 0.2)],
                key=lambda x: x[1]["games_won_against"] / x[1]["games_played"],
            )

        champ_name = champId_to_champName(champ)
        self.logger.info(f"Top 10% counters for {champ_name}:")
        for opp_id, stats in top5_counters[champ]:
            opp_name = champId_to_champName(opp_id)
            self.logger.info(
                f"  vs {opp_name}: {stats['games_played']} games, "
                f"{stats['games_won_against']} wins, "
                f"{stats['games_won_against']/stats['games_played'] * 100:.2f}% "
                "win rate"
            )

        return counter_map[champ]

    def get_matchup_stats(self, champ1, champ2): ...

    def get_synergy_matrix(self, plot=False):
        """Return a n_champion*n_champion matrix, containing synergy value"""
        num_champ = self.get_unique_champs_number()

        game_count_matrix = np.zeros((num_champ, num_champ), dtype=np.float32)
        game_won_count_matrix = np.zeros((num_champ, num_champ), dtype=np.float32)

        champ_to_idx = champ_id_to_idx_map()

        for match in self.dataset.itertuples():

            picks = getattr(match, "picks", [{}])
            blue_team_won = getattr(match, "blue_side_win", bool)

            blue_team, red_team = [], []
            for p in picks:
                idx = champ_to_idx[p["championId"]]
                if p["side"] == "blue":
                    blue_team.append(idx)
                else:
                    red_team.append(idx)

            for side, team in enumerate([blue_team, red_team]):
                team_won = (blue_team_won and side == 0) or (
                    not blue_team_won and side == 1
                )

                pairs = itertools.combinations(team, 2)
                for i, j in pairs:
                    game_count_matrix[i, j] += 1
                    game_count_matrix[j, i] += 1
                    if team_won:
                        game_won_count_matrix[i, j] += 1
                        game_won_count_matrix[j, i] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            synergy_matrix = np.divide(
                game_won_count_matrix,
                game_count_matrix,
                out=np.zeros_like(game_won_count_matrix),
                where=game_count_matrix != 0,
            )

        if plot:
            champ_names = list(get_champions_id_name_dict().values())
            df = pd.DataFrame(synergy_matrix, index=champ_names, columns=champ_names)

            plt.figure(figsize=(16, 14))
            sns.heatmap(df, cmap="coolwarm", center=0.5)
            plt.title("Champion Synergy Matrix (Winrate when played together)")
            plt.xlabel("Teammate Champion")
            plt.ylabel("Champion")
            plt.tight_layout()
            plt.show()

        return synergy_matrix

    def get_synergy(self, champ1: int, champ2: int, print=False) -> float:
        synergy_matrix = self.get_synergy_matrix()
        champ_to_idx = champ_id_to_idx_map()
        synergy_btw_champs = synergy_matrix[champ_to_idx[champ1], champ_to_idx[champ2]]

        if print:
            self.logger.info(
                f"Synergy between {champId_to_champName(champ1)} "
                f"and {champId_to_champName(champ2)} : {synergy_btw_champs * 100:.2f}%"
            )

        return synergy_btw_champs

    def get_team_synergy(self, team_champs: List[int]) -> float:
        team_synergy = 0
        pairs = itertools.combinations(team_champs, 2)
        for c in pairs:
            team_synergy += self.get_synergy(c[0], c[1])
        team_synergy /= math.comb(len(team_champs), 2)

        self.logger.info(f"The team synergy is about {team_synergy * 100:.2f}%")

        return team_synergy

    # --- Draft analysis ---
    def get_first_pick_stats(
        self,
    ):
        """Provide first and last pick stats"""
        fp_rates = {"fp": defaultdict(int), "lp": defaultdict(int)}

        df_patch = self.dataset[self.dataset["game_version"].isin(self.patches)]
        num_games = len(df_patch)

        for row in df_patch.itertuples():
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

        fp_df["first_pick_rate"] = fp_df["first_pick_count"] / num_games
        fp_df["last_pick_rate"] = fp_df["last_pick_count"] / num_games

        for x in ["first_pick_rate", "last_pick_rate"]:
            fp_df = fp_df.sort_values(x, ascending=False).reset_index(drop=True)

            highest_rate_id = fp_df.iloc[0]["championId"]
            highest_rate = fp_df.iloc[0][x]

            champ_name = champId_to_champName(highest_rate_id)

            self.logger.info(
                f"{champ_name} has the highest {" ".join(x.split("_"))} "
                f"with {highest_rate * 100:.3f}%"
            )

        return fp_df.sort_values("first_pick_rate", ascending=False).reset_index(
            drop=True
        )

    def get_draft_order_correlation(self):
        """Provide some correlations between order and win rate"""
        df_patch = self.dataset[self.dataset["game_version"].isin(self.patches)]

        data = []

        for row in df_patch.itertuples():
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
            self.logger.info(f"{position} - {side}: {pos_corr:.4f}")

        return correlation

    # --- Feature generation ---
    def build_feature_vector(self, champ_id): ...
    def get_team_features(self, team_champs): ...

    # --- Data quality ---
    def check_missing_data(self): ...

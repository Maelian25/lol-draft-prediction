from collections import Counter, defaultdict
from typing import Any, Dict, List

from matplotlib import pyplot as plt
import pandas as pd
from src.analysis.base_analysis import BaseAnalysis
from src.utils.constants import ROLE_MAP


class ChampionAnalysis(BaseAnalysis):
    """
    Analyse champions stats like win rate, pick rate, role distribution and so on

    """

    def get_champ_win_rate(self, plot=False) -> pd.Series:
        """
        Provide champ win rate

        Args:
            plot -> specify whether or not to plot
        """
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

        df_champ_rates = (
            pd.DataFrame(data)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
            .set_index("championId")
        )

        if plot:
            df_champ_rates["win_rate"].head(15).plot(
                kind="bar", color="skyblue", edgecolor="black", alpha=0.7
            )
            plt.title("Top 15 win rates in the game")
            plt.xlabel("Champions")
            plt.ylabel("Win rate")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()

        return df_champ_rates["win_rate"]

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
            champ_id = self.champ_name_id_map[champ]
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

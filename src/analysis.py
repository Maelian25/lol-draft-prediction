from collections import Counter, defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd

from src.helper import champId_to_champName, champName_to_champId
from src.logger_config import get_logger


class DatasetAnalysis:
    def __init__(self, dataset: pd.DataFrame) -> None:
        # Operations on dataset to make data expoloitable
        self.dataset = dataset.drop_duplicates(subset=["match_id"], ignore_index=True)
        self.dataset = self.dataset.dropna()
        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )

        self.num_matches = len(self.dataset)
        self.logger = get_logger("Analysis", "data_analysis.log")

    # --- Global stats ---
    def get_win_rate_per_side(self):

        blue_side_win = self.dataset["blue_side_win"].value_counts(normalize=True)

        self.logger.info(
            f"The blue side win rate in this dataset is {blue_side_win[True]*100:.3f}%"
        )
        self.logger.info(
            f"The red side win rate in this dataset is {blue_side_win[False]*100:.3f}%"
        )

        return blue_side_win

    def get_game_duration_stats(self):

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
        stats = self.dataset["game_version"].describe()

        self.logger.info(
            f"There is a total of {stats["unique"]} patches in this dataset"
        )
        self.logger.info(
            f"The patch on which most games has been played is {stats["top"]}"
        )

        patch_counts = self.dataset["game_version"].value_counts().sort_index()

        self.logger.info(
            f"Total games on patch {stats["top"]} : {patch_counts[stats["top"]]}"
        )

        patch_counts.plot(kind="bar", color="skyblue", edgecolor="black", alpha=0.7)

        plt.title("Patch distribution")
        plt.xlabel("Patch")
        plt.ylabel("Game number")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_dataset_summary(self): ...

    # --- Champion stats ---
    def get_champ_win_rate(self, patch: str):

        win_rate_dict = defaultdict(Counter)
        for row in self.dataset[self.dataset["game_version"] == patch].itertuples():
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
                    "championName": champId_to_champName(champ_id),
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

        df_result.head(10).plot(
            x="championName",
            y="win_rate",
            kind="bar",
            color="skyblue",
            title=f"Top 10 champion winrates – Patch {patch}",
        )
        plt.ylabel("Win rate")
        plt.tight_layout()
        plt.show()
        return df_result

    def get_champ_pick_or_ban_rate(self, pick: bool, patch: str):

        champ_rates = dict()
        for row in self.dataset[self.dataset["game_version"] == patch].itertuples():
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

    def get_role_distribution(self, patch: str, champ: str | int | None = None):
        role_counts = defaultdict(Counter)

        for row in self.dataset[self.dataset["game_version"] == patch].itertuples():

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
    def get_counters(self, champ): ...
    def get_matchup_stats(self, champ1, champ2): ...
    def get_synergy(self, champ1, champ2): ...
    def get_team_synergy(self, team_champs): ...
    def get_synergy_matrix(self): ...

    # --- Draft analysis ---
    def get_first_pick_stats(self, patch):
        fp_rates = {"fp": defaultdict(int), "lp": defaultdict(int)}

        df_patch = self.dataset[self.dataset["game_version"] == patch]
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

    def get_draft_order_correlation(self, patch: str):

        df = self.dataset.copy()
        if patch:
            df = df[df["game_version"] == patch]

        data = []

        for row in df.itertuples():
            blue_win = getattr(row, "blue_side_win", False)
            for pick in getattr(row, "picks", []):
                champ_id = pick["championId"]
                order = pick["order"]
                side = pick["side"]

                # Déterminer si ce pick a "gagné"
                won = (blue_win and side == "blue") or (not blue_win and side == "red")

                data.append(
                    {
                        "championId": champ_id,
                        "order": order,
                        "win": 1 if won else 0,
                    }
                )

        df_corr = pd.DataFrame(data)
        print(df_corr)

        correlation = df_corr["order"].corr(df_corr["win"])

        self.logger.info(
            f"Overall correlation between pick order and win rate: {correlation:.4f}"
        )

        return correlation

    # --- Feature generation ---
    def build_feature_vector(self, champ_id): ...
    def get_team_features(self, team_champs): ...

    # --- Data quality ---
    def check_missing_data(self): ...

import matplotlib.pyplot as plt
import pandas as pd

from src.logger_config import get_logger


class DatasetAnalysis:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.drop_duplicates(subset=["match_id"], ignore_index=True)
        self.dataset = self.dataset.dropna()
        self.num_matches = len(dataset)
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

        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )

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
    def get_champ_win_rate(self): ...
    def get_champ_pick_rate(self): ...
    def get_champ_ban_rate(self): ...
    def get_role_distribution(self, champ_id): ...
    def get_patch_winrate(self, champ_id): ...

    # --- Matchup / synergy ---
    def get_counters(self, champ): ...
    def get_matchup_stats(self, champ1, champ2): ...
    def get_synergy(self, champ1, champ2): ...
    def get_team_synergy(self, team_champs): ...
    def get_synergy_matrix(self): ...

    # --- Draft analysis ---
    def get_first_pick_stats(self): ...
    def get_last_pick_stats(self): ...
    def get_draft_order_correlation(self): ...

    # --- Feature generation ---
    def build_feature_vector(self, champ_id): ...
    def get_team_features(self, team_champs): ...

    # --- Data quality ---
    def check_missing_data(self): ...

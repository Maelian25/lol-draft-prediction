import logging


import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/data_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DatasetAnalysis:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset.dropna(subset=["game_duration"])
        self.dataset["game_duration"] = pd.to_numeric(
            self.dataset["game_duration"], errors="coerce"
        )
        self.num_matches = len(dataset)

    # --- Global stats ---
    def get_win_rate_per_side(self):
        blue_side_win = 0

        for match in self.dataset.itertuples():
            blue_side_win = blue_side_win + 1 if match.blue_side_win else blue_side_win

        blue_side_win_rate = blue_side_win / self.num_matches * 100
        red_side_win_rate = 100 - blue_side_win_rate

        logger.info(
            f"The blue side win rate in this dataset is {blue_side_win_rate:.3f}%"
        )
        logger.info(
            f"The red side win rate in this dataset is {red_side_win_rate:.3f}%"
        )

        return blue_side_win_rate, red_side_win_rate

    def get_game_duration_stats(self):

        stats = self.dataset["game_duration"].describe()
        logger.info(f"Average game time : {stats["mean"]:.2f}")

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
        logger.info(f"The patch on which most games has been played is {stats["top"]}")
        logger.info(f"There is a total of {stats["unique"]} patches in this dataset")

        patch_counts = self.dataset["game_version"].value_counts().sort_index()

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

from typing import Dict
from matplotlib import pyplot as plt
import pandas as pd
from src.analysis.base_analysis import BaseAnalysis


class GlobalAnalysis(BaseAnalysis):
    """
    Provide methods for global anaylysis of the dataset
    """

    def get_win_rate_per_side(self) -> pd.Series:
        """Provide win rate for each side"""
        blue_side_win = self.dataset["blue_side_win"].value_counts(normalize=True)

        self.logger.info(
            f"The blue side win rate in this dataset is {blue_side_win[True]*100:.3f}%"
        )
        self.logger.info(
            f"The red side win rate in this dataset is {blue_side_win[False]*100:.3f}%"
        )

        return blue_side_win

    def get_game_duration_stats(self, plot: bool) -> Dict[str, float]:
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

    def get_patch_distribution(self, plot=False) -> str:
        """Provide games distribution through patches"""
        stats = self.dataset["game_version"].describe()

        self.logger.info(
            f"There is a total of {stats["unique"]} patches in this dataset"
        )
        self.logger.info(
            f"The patch on which most games has been played is {stats["top"]}"
        )
        self.logger.info(f"Total games on patch {stats["top"]} : {stats["freq"]}")

        if plot:
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

from collections import defaultdict

import pandas as pd
from src.analysis.base_analysis import BaseAnalysis


class DraftAnalysis(BaseAnalysis):
    """
    Provide methods to analyse draft
    """

    def first_last_pick_stats(self) -> pd.DataFrame:
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

    def get_draft_order_correlation(self) -> None:
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

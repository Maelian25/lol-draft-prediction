from time import perf_counter
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from src.analysis.matrices.counter_matrix import CounterAnalysis
from src.analysis.matrices.synergy_matrix import SynergyAnalysis
from src.analysis.stats.champion_stats import ChampionAnalysis
from src.analysis.stats.global_stats import GlobalAnalysis
from src.utils.champions_helper import champ_id_to_idx_map
from src.utils.constants import (
    CHAMP_EMBEDS,
    COUNT_MAT,
    DATA_REPRESENTATION_FOLDER,
    DRAFT_STATES_CSV,
    DRAFT_STATES_PARQUET,
    MATRICES_FOLDER,
    ROLE_MAP,
    SYN_MAT,
)
from src.utils.general_helper import load_file, save_file
from src.utils.logger_config import get_logger


class DatasetAnalysis:
    """
    Orchestrate every analysis
    """

    def __init__(self, data: pd.DataFrame, patches: Optional[List[str]] = None) -> None:

        self.logger = get_logger(self.__class__.__name__, "analysis.log")

        if patches:
            self.logger.info(f"Analysing dataset for patch {'-'.join(patches)}")

        self.global_analysis = GlobalAnalysis(data, patches)
        self.champion_stats = ChampionAnalysis(data, patches)
        self.synergy = SynergyAnalysis(data, patches)
        self.counter = CounterAnalysis(data, patches)

    def get_champion_embeddings(self) -> pd.DataFrame:

        data_champ_embeds = load_file(DATA_REPRESENTATION_FOLDER, CHAMP_EMBEDS)
        data_syn_matrix = load_file(MATRICES_FOLDER, SYN_MAT)
        data_count_matrix = load_file(MATRICES_FOLDER, COUNT_MAT)

        if data_champ_embeds is not None:
            if data_syn_matrix is None or data_count_matrix is None:
                self.counter_matrix = self.counter.compute_counter_matrix()
                self.synergy_matrix = self.synergy.compute_synergy_matrix()
            else:
                self.synergy_matrix = data_syn_matrix
                self.counter_matrix = data_count_matrix

            return data_champ_embeds

        win_rate = self.champion_stats.get_champ_win_rate(plot=False)
        pick_rate = self.champion_stats.get_champ_pick_or_ban_rate(pick=True)
        ban_rate = self.champion_stats.get_champ_pick_or_ban_rate(pick=False)
        self.counter_matrix = self.counter.compute_counter_matrix()
        self.synergy_matrix = self.synergy.compute_synergy_matrix()

        champ_embeddings = {
            champ_id: np.concatenate(
                [
                    self.champion_stats.get_champion_infos(champ_id),
                    np.array(
                        [win_rate[champ_id], pick_rate[champ_id], ban_rate[champ_id]]
                    ),
                    self.champion_stats.get_role_distribution(champ_id),
                    np.array(
                        [
                            self.counter_matrix[champ_id].mean(),
                            self.synergy_matrix[champ_id].mean(),
                        ]
                    ),
                ]
            )
            for champ_id in self.champion_stats.champ_id_name_map.keys()
        }

        all_stats = np.vstack(list(champ_embeddings.values()))
        scaled_all_stats = self.champion_stats.scaler.fit_transform(all_stats)

        champ_embeddings = {
            champ_id: scaled_all_stats[i].tolist()
            for i, champ_id in enumerate(self.champion_stats.champ_id_name_map.keys())
        }

        champ_embeddings_df = pd.DataFrame(champ_embeddings)
        save_file(champ_embeddings_df, DATA_REPRESENTATION_FOLDER, CHAMP_EMBEDS)

        return champ_embeddings_df

    def build_matches_states(self) -> pd.DataFrame:

        data = load_file(DATA_REPRESENTATION_FOLDER, DRAFT_STATES_PARQUET)

        if data is not None:
            return data

        matches_info = list(dict())

        # Pre load every matrices needed for computation
        t_compute_start = perf_counter()
        champion_embeddings = self.get_champion_embeddings()
        champ_id_to_idx = champ_id_to_idx_map()
        pad_idx = champ_id_to_idx["PAD"]
        t_compute_end = perf_counter()

        self.logger.info(
            f"Took {t_compute_end - t_compute_start:2f} s to pre-load matrices"
        )

        embed_dim = len(champion_embeddings.index)

        drafting_phase: dict[str, list[dict[str, int]]] = {
            "ban_phase_1": [
                {"blue": 0},
                {"red": 0},
                {"blue": 1},
                {"red": 1},
                {"blue": 2},
                {"red": 2},
            ],
            "pick_phase_1": [
                {"blue": 0},
                {"red": 0},
                {"red": 1},
                {"blue": 1},
                {"blue": 2},
                {"red": 2},
            ],
            "ban_phase_2": [
                {"red": 3},
                {"blue": 3},
                {"red": 4},
                {"blue": 4},
            ],
            "pick_phase_2": [
                {"red": 3},
                {"blue": 3},
                {"blue": 4},
                {"red": 4},
            ],
        }

        t_compute_start = perf_counter()
        self.logger.info("Starting match states generation")

        for match in self.champion_stats.dataset.itertuples():

            match_id = getattr(match, "match_id")
            picks = getattr(match, "picks")
            bans = getattr(match, "bans")
            blue_side_win = getattr(match, "blue_side_win")

            blue_bans = np.full(5, pad_idx, dtype=np.int32, device="cpu")
            blue_picks = np.full(5, pad_idx, dtype=np.int32, device="cpu")
            red_bans = np.full(5, pad_idx, dtype=np.int32, device="cpu")
            red_picks = np.full(5, pad_idx, dtype=np.int32, device="cpu")
            blue_roles = np.ones(5, dtype=np.int32)
            red_roles = np.ones(5, dtype=np.int32)
            champ_availability = np.ones(self.champion_stats.n_champs, dtype=np.float32)

            blue_picks_embed = np.zeros(5 * embed_dim, np.float32)
            red_picks_embed = np.zeros(5 * embed_dim, np.float32)
            blue_bans_embed = np.zeros(5 * embed_dim, np.float32)
            red_bans_embed = np.zeros(5 * embed_dim, np.float32)

            step = 0

            for phase, phase_order in drafting_phase.items():
                phase_type = phase.split("_")[0]
                for side_dict in phase_order:
                    side, val = next(iter(side_dict.items()))

                    target_winrate = float(
                        (side == "blue" and blue_side_win)
                        or (side == "red" and not blue_side_win)
                    )

                    record = {
                        "match_id": match_id,
                        "step": step,
                        "blue_picks": blue_picks.copy(),
                        "blue_picks_embed": blue_picks_embed.copy(),
                        "red_picks": red_picks.copy(),
                        "red_picks_embed": red_picks_embed.copy(),
                        "blue_bans": blue_bans.copy(),
                        "blue_bans_embed": blue_bans_embed.copy(),
                        "red_bans": red_bans.copy(),
                        "red_bans_embed": red_bans_embed.copy(),
                        "blue_roles_available": blue_roles.copy(),
                        "red_roles_available": red_roles.copy(),
                        "champ_availability": champ_availability.copy(),
                        "blue_synergy_score": self.synergy.team_synergy_score(
                            self.synergy_matrix, blue_picks, is_idx=True, log=False
                        ),
                        "red_synergy_score": self.synergy.team_synergy_score(
                            self.synergy_matrix, red_picks, is_idx=True, log=False
                        ),
                        # how much blue team counters red team
                        "counter_score": self.counter.team_counter_score(
                            self.counter_matrix, blue_picks, red_picks, is_idx=True
                        ),
                        "next_phase": phase_type,
                        "next_side": side,
                        "target_winrate": target_winrate,
                    }

                    if phase_type == "ban":
                        target_ban = (
                            bans[val]["championId"]
                            if side == "blue"
                            else bans[val + 5]["championId"]
                        )
                        record.update(
                            {
                                "target_ban": champ_id_to_idx[str(target_ban)],
                                "target_pick": np.nan,
                                "target_role": np.nan,
                            }
                        )
                    else:
                        pick_info = picks[val] if side == "blue" else picks[val + 5]
                        target_pick = pick_info["championId"]
                        target_role = pick_info["position"]

                        record.update(
                            {
                                "target_ban": np.nan,
                                "target_pick": champ_id_to_idx[str(target_pick)],
                                "target_role": ROLE_MAP[target_role] - 1,
                            }
                        )

                    matches_info.append(record)
                    step += 1

                    start = val * embed_dim
                    end = start + embed_dim

                    # Ban phases
                    if phase_type == "ban":
                        ban_info = bans[val] if side == "blue" else bans[val + 5]
                        champ_id = ban_info["championId"]
                        champ_idx = champ_id_to_idx[str(champ_id)]

                        champ_embed = champion_embeddings[champ_id].values.astype(
                            np.float32
                        )

                        if side == "blue":
                            blue_bans[val] = champ_idx

                            champ_availability[champ_idx] = 0

                            blue_bans_embed[start:end] = champ_embed
                        else:
                            red_bans[val] = champ_idx

                            champ_availability[champ_idx] = 0

                            red_bans_embed[start:end] = champ_embed
                    # Pick phases
                    else:
                        pick_info = picks[val] if side == "blue" else picks[val + 5]

                        role_idx = ROLE_MAP[pick_info["position"]] - 1
                        champ_id = pick_info["championId"]
                        champ_idx = champ_id_to_idx[str(champ_id)]

                        champ_embed = champion_embeddings[champ_id].values.astype(
                            np.float32
                        )

                        if side == "red":
                            red_roles[role_idx] = 0
                            red_picks[val] = champ_idx

                            champ_availability[champ_idx] = 0

                            red_picks_embed[start:end] = champ_embed
                        else:
                            blue_roles[role_idx] = 0
                            blue_picks[val] = champ_idx

                            champ_availability[champ_idx] = 0

                            blue_picks_embed[start:end] = champ_embed

        df = pd.DataFrame(matches_info)
        df["next_phase"] = (df["next_phase"] == "pick").astype(np.float32)
        df["next_side"] = (df["next_side"] == "blue").astype(np.float32)

        t_compute_end = perf_counter()

        scaler = StandardScaler()
        score_cols = ["blue_synergy_score", "red_synergy_score", "counter_score"]
        df[score_cols] = scaler.fit_transform(df[score_cols])

        self.logger.info("Match states generation done. Now saving file...")
        self.logger.info(
            f"Took {t_compute_end - t_compute_start} to build every match states"
        )

        save_file(
            df.head(100),
            DATA_REPRESENTATION_FOLDER,
            DRAFT_STATES_CSV,
        )
        save_file(df, DATA_REPRESENTATION_FOLDER, DRAFT_STATES_PARQUET)

        return df

    def get_overview(self):
        winrate = self.champion_stats.get_champ_win_rate()
        synergy = self.synergy.compute_synergy_matrix()
        return {
            "winrate": winrate.head(10),
            "synergy_sample": synergy.iloc[:5, :5],
        }

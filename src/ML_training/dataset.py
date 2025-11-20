from typing import Any
import torch
from torch.utils.data import Dataset

from src.utils.general_helper import load_file


class DraftDataset(Dataset):
    """
    Build data ready for machine learning
    """

    def __init__(self, data_folder, data_file):

        data: Any = load_file(data_folder, data_file)

        if data is None:
            return

        self.match_states = data

        self.blue_picks_emb = data["blue_picks_emb"]
        self.red_picks_emb = data["red_picks_emb"]
        self.blue_bans_emb = data["blue_bans_emb"]
        self.red_bans_emb = data["red_bans_emb"]

        self.blue_picks = data["blue_picks"]
        self.red_picks = data["red_picks"]
        self.blue_bans = data["blue_bans"]
        self.red_bans = data["red_bans"]

        self.champ_mask = data["champ_availability"]
        self.blue_roles_mask = data["blue_roles_available"]
        self.red_roles_mask = data["red_roles_available"]

        self.step = data["step"]
        self.blue_syn = data["blue_syn"]
        self.red_syn = data["red_syn"]
        self.counter = data["counter"]
        self.side = data["side"]

        self.phase = data["phase"]
        self.t_pick = data["t_pick"]
        self.t_ban = data["t_ban"]
        self.t_role = data["t_role"]
        self.t_wr = data["t_wr"]

        self.X_static = torch.cat(
            [
                self.blue_picks_emb,
                self.red_picks_emb,
                self.blue_bans_emb,
                self.red_bans_emb,
                self.blue_syn,
                self.red_syn,
                self.counter,
                self.side,
            ],
            dim=1,
        ).contiguous()

    def __len__(self):
        return len(self.phase)

    def __getitem__(self, idx):

        return (
            self.X_static[idx],
            self.blue_picks[idx],
            self.red_picks[idx],
            self.blue_bans[idx],
            self.red_bans[idx],
            self.champ_mask[idx],
            self.blue_roles_mask[idx],
            self.red_roles_mask[idx],
            self.step[idx],
            self.side[idx],
            self.phase[idx],
            self.t_pick[idx],
            self.t_ban[idx],
            self.t_role[idx],
            self.t_wr[idx],
        )

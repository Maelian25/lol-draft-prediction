from typing import Any
import numpy as np
import torch
import torch.nn as nn

from src.utils.constants import DATA_REPRESENTATION_FOLDER, DRAFT_STATES_TORCH
from src.utils.logger_config import get_logger

logger = get_logger("Torch process", "torch.log")


def preprocess_and_save(df: Any, rebuild=True):

    if rebuild:

        save_path = DATA_REPRESENTATION_FOLDER + DRAFT_STATES_TORCH

        blue_picks_emb = torch.tensor(
            np.stack(df["blue_picks_embed"]), dtype=torch.float32
        )
        red_picks_emb = torch.tensor(
            np.stack(df["red_picks_embed"]), dtype=torch.float32
        )
        blue_bans_emb = torch.tensor(
            np.stack(df["blue_bans_embed"]), dtype=torch.float32
        )
        red_bans_emb = torch.tensor(np.stack(df["red_bans_embed"]), dtype=torch.float32)

        blue_picks = torch.tensor(np.stack(df["blue_picks"]), dtype=torch.long)
        red_picks = torch.tensor(np.stack(df["red_picks"]), dtype=torch.long)
        blue_bans = torch.tensor(np.stack(df["blue_bans"]), dtype=torch.long)
        red_bans = torch.tensor(np.stack(df["red_bans"]), dtype=torch.long)

        champ_availability = torch.tensor(
            np.stack(df["champ_availability"]), dtype=torch.long
        )
        blue_roles_available = torch.tensor(
            np.stack(df["blue_roles_available"]), dtype=torch.long
        )
        red_roles_available = torch.tensor(
            np.stack(df["red_roles_available"]), dtype=torch.long
        )

        champ_availability = expand_mask(champ_availability)

        # simple scalars
        step = torch.tensor(df["step"].values, dtype=torch.long)
        blue_syn = torch.tensor(
            df["blue_synergy_score"].values, dtype=torch.float32
        ).unsqueeze(1)
        red_syn = torch.tensor(
            df["red_synergy_score"].values, dtype=torch.float32
        ).unsqueeze(1)
        counter = torch.tensor(
            df["counter_score"].values, dtype=torch.float32
        ).unsqueeze(1)
        side = torch.tensor(df["next_side"].values, dtype=torch.long).unsqueeze(1)

        # labels
        phase = torch.tensor(df["next_phase"].values, dtype=torch.long)
        t_pick = torch.tensor(df["target_pick"].values, dtype=torch.long)
        t_ban = torch.tensor(df["target_ban"].values, dtype=torch.long)
        t_role = torch.tensor(df["target_role"].values, dtype=torch.long)
        t_wr = torch.tensor(df["target_winrate"].values, dtype=torch.float32)

        # Save in a single efficient file
        torch.save(
            {
                # static embedding inputs
                "blue_picks_emb": blue_picks_emb,
                "red_picks_emb": red_picks_emb,
                "blue_bans_emb": blue_bans_emb,
                "red_bans_emb": red_bans_emb,
                # integer input for learnable embedding mode
                "blue_picks": blue_picks,
                "red_picks": red_picks,
                "blue_bans": blue_bans,
                "red_bans": red_bans,
                # masks
                "champ_availability": champ_availability,
                "blue_roles_available": blue_roles_available,
                "red_roles_available": red_roles_available,
                # scalar features
                "step": step,
                "blue_syn": blue_syn,
                "red_syn": red_syn,
                "counter": counter,
                "side": side,
                # supervision labels
                "phase": phase,
                "t_pick": t_pick,
                "t_ban": t_ban,
                "t_role": t_role,
                "t_wr": t_wr,
            },
            save_path,
        )

        logger.info(f"Preprocessed dataset saved to {save_path}")


def expand_mask(mask: torch.Tensor):

    num_rows, _ = mask.shape
    pad_col = torch.zeros(num_rows, 1, dtype=mask.dtype, device=mask.device)

    return torch.cat([mask, pad_col], dim=1)


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, k_list=[1, 5]):
    """
    Provide best metrics for my model
    """

    with torch.no_grad():

        max_k = max(k_list)

        _, pred = logits.topk(max_k, 1, True, True)
        pred = pred.t()

        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = {}
        for k in k_list:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = correct_k.item()

        return res


class MultiTaskLossWrapper(nn.Module):

    def __init__(self, vec_init, requires_grad):
        super().__init__()

        initial_log_vars = torch.tensor(vec_init, device="cuda:0")
        self.log_vars = nn.Parameter(initial_log_vars, requires_grad=requires_grad)

    def forward(self, loss_champ, loss_role, loss_win):
        # Formulas from Kendall et al. (CVPR 2018)

        # Champ pick or ban loss
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * loss_champ + self.log_vars[0]

        # Role loss
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * loss_role + self.log_vars[1]

        # Winrate loss
        precision3 = torch.exp(-self.log_vars[2])
        loss3 = precision3 * loss_win + self.log_vars[2]

        # Somme finale
        return loss1 + loss2 + loss3

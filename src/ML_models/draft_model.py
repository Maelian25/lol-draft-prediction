import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils.logger_config import get_logger


class DraftDataset(Dataset):

    def __init__(self, matchs_states: pd.DataFrame):

        self.match_states = matchs_states
        self.__build_vectors()

    def __len__(self):
        return len(self.match_states)

    def __getitem__(self, idx):
        X_static = torch.cat(
            [
                self.blue_picks_emb[idx],
                self.red_picks_emb[idx],
                self.blue_bans_emb[idx],
                self.red_bans_emb[idx],
                self.blue_syn[idx],
                self.red_syn[idx],
                self.counter[idx],
                self.side[idx],
            ],
            dim=0,
        )

        return (
            X_static,
            self.blue_picks_idxs[idx],
            self.red_picks_idxs[idx],
            self.blue_bans_idxs[idx],
            self.red_bans_idxs[idx],
            self.side[idx],
            self.phase[idx],
            self.t_pick[idx],
            self.t_ban[idx],
            self.t_role[idx],
            self.t_wr[idx],
        )

    def __build_vectors(self):

        def list_to_tensor(col, dtype=torch.float32):
            return torch.tensor(np.stack(self.match_states[col].to_list()), dtype=dtype)

        def value_to_tensor(col, dtype=torch.float32):
            return torch.tensor(self.match_states[col].values, dtype=dtype)

        self.blue_picks_emb = list_to_tensor("blue_picks_embed", dtype=torch.float32)
        self.red_picks_emb = list_to_tensor("red_picks_embed", dtype=torch.float32)
        self.blue_bans_emb = list_to_tensor("blue_bans_embed", dtype=torch.float32)
        self.red_bans_emb = list_to_tensor("red_bans_embed", dtype=torch.float32)

        self.blue_picks_idxs = list_to_tensor("blue_picks", dtype=torch.long)
        self.red_picks_idxs = list_to_tensor("red_picks", dtype=torch.long)
        self.blue_bans_idxs = list_to_tensor("blue_bans", dtype=torch.long)
        self.red_bans_idxs = list_to_tensor("red_bans", dtype=torch.long)

        self.blue_syn = value_to_tensor(
            "blue_synergy_score", dtype=torch.float32
        ).unsqueeze(1)
        self.red_syn = value_to_tensor(
            "red_synergy_score", dtype=torch.float32
        ).unsqueeze(1)
        self.counter = value_to_tensor("counter_score", dtype=torch.float32).unsqueeze(
            1
        )
        self.side = value_to_tensor("next_side", dtype=torch.float32).unsqueeze(1)
        self.phase = value_to_tensor("next_phase", dtype=torch.float32)

        self.t_pick = value_to_tensor("target_pick", dtype=torch.long)
        self.t_ban = value_to_tensor("target_ban", dtype=torch.long)
        self.t_role = value_to_tensor("target_role", dtype=torch.long)
        self.t_wr = value_to_tensor("target_winrate", dtype=torch.float32)


class DraftUnifiedModel(nn.Module):

    def __init__(
        self,
        num_champions,
        num_roles,
        input_dim=0,
        hidden_dim=512,
        dropout=0.3,
        mode="static",
        embed_size=64,
    ) -> None:
        super().__init__()

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")
        self.mode = mode

        if mode == "learnable":
            self.champ_embedding = nn.Embedding(num_champions + 1, embed_size)
            input_dim = 4 * embed_size + 1

        self.logger.info(f"Input dim for {mode} : {input_dim}")

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        self.pick_head = nn.Linear(hidden_dim // 2, num_champions)
        self.ban_head = nn.Linear(hidden_dim // 2, num_champions)
        self.role_head = nn.Linear(hidden_dim // 2, num_roles)
        self.wr_head = nn.Linear(hidden_dim // 2, 1)

    def encode_team(self, champ_ids):
        emb = self.champ_embedding(champ_ids)
        return emb.mean(dim=1)

    def forward(
        self, X_static, blue_picks, red_picks, blue_bans, red_bans, side, phase
    ):

        if self.mode == "learnable":
            blue_picks_emb = self.encode_team(blue_picks)
            red_picks_emb = self.encode_team(red_picks)
            blue_bans_emb = self.encode_team(blue_bans)
            red_bans_emb = self.encode_team(red_bans)

            X = torch.cat(
                [blue_picks_emb, red_picks_emb, blue_bans_emb, red_bans_emb, side],
                dim=1,
            )
        else:
            X = X_static

        shared = self.shared(X)
        pick_logits = self.pick_head(shared)
        ban_logits = self.ban_head(shared)
        role_logits = self.role_head(shared)
        winrate = torch.sigmoid(self.wr_head(shared))

        champ_logits = torch.where(phase.unsqueeze(1) == 1, pick_logits, ban_logits)

        return champ_logits, role_logits, winrate


class DraftBrain:

    def __init__(
        self,
        matchs_states: pd.DataFrame,
        input_dim,
        num_champions,
        num_roles,
        batch_size=1024,
        hidden_dim=512,
        dropout=0.3,
        mode="static",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:

        self.device = device

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")

        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available:  {torch.cuda.is_available()}")
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        )
        self.logger.info(f"GPU name:{gpu_name}")

        train_df, val_df = train_test_split(
            matchs_states, test_size=0.2, random_state=42
        )
        train_ds = DraftDataset(train_df)
        val_ds = DraftDataset(val_df)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(val_ds, batch_size=batch_size)

        self.model = DraftUnifiedModel(
            num_champions=num_champions,
            num_roles=num_roles,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mode=mode,
        ).to(device)

        self.logger.info(f"Parameters used : Batch size = {batch_size} | Mode = {mode}")

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.loss_champ = nn.CrossEntropyLoss()
        self.loss_role = nn.CrossEntropyLoss()
        self.loss_wr = nn.MSELoss()

    def train(self, num_epochs=10):

        self.logger.info("Started training")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_loader):

                (X, bp, rp, bb, rb, side, phase, y_pick, y_ban, y_role, y_wr) = [
                    b.to(self.device, non_blocking=True) for b in batch
                ]

                self.opt.zero_grad()

                champ_logits, role_logits, wr_pred = self.model(
                    X, bp, rp, bb, rb, side, phase
                )

                y_champ = torch.where(phase == 1, y_pick, y_ban)

                loss_c = self.loss_champ(champ_logits, y_champ)

                mask = phase == 1

                loss_r = (
                    self.loss_role(role_logits[mask], y_role[mask])
                    if mask.any()
                    else 0.0
                )
                loss_w = (
                    self.loss_wr(wr_pred[mask].squeeze(-1), y_wr[mask])
                    if mask.any()
                    else 0.0
                )

                loss = loss_c + 0.5 * loss_r + 0.2 * loss_w
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: loss={total_loss/len(self.train_loader):.4f}")

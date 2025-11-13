import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.utils.logger_config import get_logger


class Draft_Dataset(Dataset):

    def __init__(self, matchs_states: pd.DataFrame):
        (
            self.X,
            self.phase_flag,
            self.y_champ_pick,
            self.y_champ_ban,
            self.y_role,
            self.y_wr,
        ) = self.__build_vectors(matchs_states)

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.phase_flag[idx],
            self.y_champ_pick[idx],
            self.y_champ_ban[idx],
            self.y_role[idx],
            self.y_wr[idx],
        )

    def __build_vectors(self, data: pd.DataFrame):

        # 800 features here
        blue_picks = np.stack(data["blue_picks_embed"].to_list())
        red_picks = np.stack(data["red_picks_embed"].to_list())
        blue_bans = np.stack(data["blue_bans_embed"].to_list())
        red_bans = np.stack(data["red_bans_embed"].to_list())

        # 171 + 10 + 10 features here
        availability = np.stack(data["availability"].tolist())
        blue_roles = np.stack(data["blue_roles_filled"].to_list())
        red_roles = np.stack(data["red_roles_filled"].to_list())

        scaler = StandardScaler()
        score_cols = ["blue_synergy_score", "red_synergy_score", "counter_score"]
        data[score_cols] = scaler.fit_transform(data[score_cols])

        # 4 features here
        blue_synergy_score = (
            data["blue_synergy_score"].to_numpy(dtype=np.float32).reshape(-1, 1)
        )
        red_synergy_score = (
            data["red_synergy_score"].to_numpy(dtype=np.float32).reshape(-1, 1)
        )
        counter_score = data["counter_score"].to_numpy(dtype=np.float32).reshape(-1, 1)
        side = data["next_side"].to_numpy(dtype=np.float32).reshape(-1, 1)

        X = np.hstack(
            [
                blue_picks,
                red_picks,
                blue_bans,
                red_bans,
                availability,
                blue_roles,
                red_roles,
                blue_synergy_score,
                red_synergy_score,
                counter_score,
                side,
            ]
        )

        print("X shape:", X.shape)
        print("dtype:", X.dtype)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(data["next_phase"].to_numpy(), dtype=torch.float32),
            torch.tensor(data["target_pick"].to_numpy(), dtype=torch.long),
            torch.tensor(data["target_ban"].to_numpy(), dtype=torch.long),
            torch.tensor(data["target_role"].to_numpy(), dtype=torch.long),
            torch.tensor(data["target_winrate"].to_numpy(), dtype=torch.float32),
        )


class Draft_Unified_Model(nn.Module):

    def __init__(
        self, input_dim, num_champions, num_roles, hidden_dim=512, dropout=0.3
    ) -> None:
        super().__init__()

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

    def forward(self, X, phase_flag):

        shared = self.shared(X)
        pick_logits = self.pick_head(shared)
        ban_logits = self.ban_head(shared)
        role_logits = self.role_head(shared)
        winrate = self.wr_head(shared)

        out_champ = torch.where(phase_flag.unsqueeze(1).bool(), pick_logits, ban_logits)
        out_role = torch.where(phase_flag.unsqueeze(1).bool(), role_logits, torch.nan)

        return out_champ, out_role, winrate


class Draft_Brain:

    def __init__(
        self,
        matchs_states: pd.DataFrame,
        num_champions,
        num_roles,
        batch_size=128,
        hidden_dim=512,
        dropout=0.3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:

        self.device = device

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print(
            "GPU name:",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        )

        self.dataset = Draft_Dataset(matchs_states)
        self.input_dim = len(self.dataset)

        self.model = nn.Module()
        self.model = Draft_Unified_Model(
            self.input_dim, num_champions, num_roles, hidden_dim, dropout
        ).to(device)

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")

    def train(self, num_epochs=1000, lr=1e-3):

        self.logger.info(f"Training on {self.device} | Input dim = {self.input_dim}")

        champ_loss_fn = nn.CrossEntropyLoss()
        role_loss_fn = nn.CrossEntropyLoss()
        winrate_loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            for X, phase, y_pick, y_ban, y_role, y_wr in self.dataloader:

                X, phase = X.to(self.device), phase.to(self.device)
                y_pick, y_ban, y_role, y_wr = (
                    y_pick.to(self.device),
                    y_ban.to(self.device),
                    y_role.to(self.device),
                    y_wr.to(self.device),
                )

                optimizer.zero_grad()
                champ_logits, role_logits, winrate_pred = self.model(X, phase)

                # --- Champ loss ---
                y_champ = torch.where(phase == 1, y_pick, y_ban).long()

                loss_champ = champ_loss_fn(champ_logits, y_champ)

                # --- Role loss & Winrate loss (pick phase only) ---
                is_pick = phase == 1
                winrate_pred_selected = winrate_pred[is_pick].float()

                winrate_pred_selected = winrate_pred_selected.squeeze(-1)

                y_wr_selected = y_wr[is_pick].float()

                # Role loss
                if is_pick.any():
                    loss_role = role_loss_fn(
                        role_logits[is_pick], y_role[is_pick].long()
                    )
                else:
                    loss_role = 0.0

                # Winrate loss
                if is_pick.any():
                    winrate_pred_selected = winrate_pred[is_pick].squeeze(-1).float()
                    y_wr_selected = y_wr[is_pick].float()
                    loss_wr = winrate_loss_fn(winrate_pred_selected, y_wr_selected)
                else:
                    loss_wr = 0.0

                # Total loss
                loss = loss_champ + loss_role + loss_wr
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"Loss: {total_loss / len(self.dataloader):.4f}"
                )

from typing import Any
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils.constants import DATA_REPRESENTATION_FOLDER, DRAFT_STATES_TORCH
from src.utils.general_helper import load_file
from src.utils.logger_config import get_logger


class DraftDataset(Dataset):

    def __init__(self):

        data: Any = load_file(DATA_REPRESENTATION_FOLDER, DRAFT_STATES_TORCH)

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

        self.blue_syn = data["blue_syn"]
        self.red_syn = data["red_syn"]
        self.counter = data["counter"]
        self.side = data["side"]

        self.phase = data["phase"]
        self.t_pick = data["t_pick"]
        self.t_ban = data["t_ban"]
        self.t_role = data["t_role"]
        self.t_wr = data["t_wr"]

    def __len__(self):
        return len(self.phase)

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
        ).contiguous()

        return (
            X_static,
            self.blue_picks[idx],
            self.red_picks[idx],
            self.blue_bans[idx],
            self.red_bans[idx],
            self.side[idx],
            self.phase[idx],
            self.t_pick[idx],
            self.t_ban[idx],
            self.t_role[idx],
            self.t_wr[idx],
        )


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

        dataset = DraftDataset()
        idx_train, idx_val = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )

        train_subset = torch.utils.data.Subset(dataset, idx_train)
        val_subset = torch.utils.data.Subset(dataset, idx_val)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(val_subset, batch_size=batch_size)

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
                    b.to(self.device, non_blocking=True) for b in batch  # type: ignore
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

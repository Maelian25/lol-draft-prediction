from typing import Any
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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

        self.champ_mask = data["champ_availability"]
        self.blue_roles_mask = data["blue_roles_available"]
        self.red_roles_mask = data["red_roles_available"]

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
            self.champ_mask[idx],
            self.blue_roles_mask[idx],
            self.red_roles_mask[idx],
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
        embed_size,
        mode,
        input_dim,
        hidden_dim,
        dropout,
    ) -> None:
        super().__init__()

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")
        self.mode = mode

        if mode == "learnable":
            self.champ_embedding = nn.Embedding(
                num_champions + 1, embed_size, padding_idx=num_champions
            )
            input_dim = 20 * embed_size + 1

        self.logger.info(f"Input dim for {mode}: {input_dim}")

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        self.pick_head = nn.Linear(hidden_dim // 2, num_champions + 1)
        self.ban_head = nn.Linear(hidden_dim // 2, num_champions + 1)
        self.role_head = nn.Linear(hidden_dim // 2, num_roles)
        self.wr_head = nn.Linear(hidden_dim // 2, 1)

    def encode_team(self, champ_ids):
        emb = self.champ_embedding(champ_ids)
        return emb.view(emb.shape[0], -1)

    def forward(
        self,
        X_static,
        blue_picks,
        red_picks,
        blue_bans,
        red_bans,
        champ_mask,
        b_role_mask,
        r_role_mask,
        side,
        phase,
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
        winrate = self.wr_head(shared)

        champ_logits = torch.where(phase.unsqueeze(1) == 1, pick_logits, ban_logits)
        champ_logits = self.__apply_mask(champ_logits, champ_mask)

        role_mask = torch.where(
            side == 1,
            b_role_mask,
            r_role_mask,
        )

        role_logits = self.__apply_mask(role_logits, role_mask)

        return champ_logits, role_logits, winrate

    def __apply_mask(self, logits: torch.Tensor, mask: torch.Tensor):

        unavailable = mask == 0

        logits = logits.masked_fill(unavailable, float("-inf"))

        return logits


class DraftBrain:

    def __init__(
        self,
        input_dim,
        num_champions,
        num_roles,
        num_epochs=10,
        batch_size=1024,
        hidden_dim=512,
        embed_size=64,
        dropout=0.3,
        mode="static",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")

        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        )
        self.logger.info(f"GPU name: {gpu_name}")

        self.model = DraftUnifiedModel(
            num_champions=num_champions,
            num_roles=num_roles,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mode=mode,
            embed_size=embed_size,
        ).to(device)

        self.__build_dataset()
        self.__training_fns()

        self.logger.info(f"Parameters used : Batch size = {batch_size} | Mode = {mode}")

    def __build_dataset(self):

        dataset = DraftDataset()
        idx_train, idx_val = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )

        train_subset = torch.utils.data.Subset(dataset, idx_train)
        val_subset = torch.utils.data.Subset(dataset, idx_val)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def __training_fns(self):

        base_batch = 128
        base_lr = 3e-4
        base_warmup_steps = 800
        warmup_steps_ceiling = 2500
        total_steps = self.num_epochs * len(self.train_loader)

        scaled_lr = base_lr * (self.batch_size / base_batch)
        scaled_warmup_steps = min(
            warmup_steps_ceiling,
            max(
                base_warmup_steps,
                int(base_warmup_steps * (self.batch_size / base_batch)),
            ),
        )

        self.logger.info(f"Scaled learning rate: {scaled_lr}")
        self.logger.info(f"Warmup steps: {scaled_warmup_steps}")

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=scaled_lr, weight_decay=1e-2
        )

        warmup_scheduler = LinearLR(
            self.opt,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=scaled_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.opt,
            T_max=(total_steps - scaled_warmup_steps),
            eta_min=0.0,
        )

        self.scheduler = SequentialLR(
            self.opt,
            [
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[scaled_warmup_steps],
        )

        self.scaler = torch.GradScaler(self.device)

        self.loss_champ = nn.CrossEntropyLoss()
        self.loss_role = nn.CrossEntropyLoss()
        self.loss_wr = nn.BCEWithLogitsLoss()

    def train(self):

        for epoch in range(self.num_epochs):

            self.logger.info(f"Started training for Epoch {epoch + 1}")
            self.model.train()
            total_loss = 0.0
            total_loss_c = 0.0
            total_loss_r = 0.0
            total_loss_w = 0.0

            count_role_batches = 0
            count_wr_batches = 0

            for batch in tqdm(self.train_loader):

                (
                    X,
                    bp,
                    rp,
                    bb,
                    rb,
                    champ_mask,
                    b_role_mask,
                    r_role_mask,
                    side,
                    phase,
                    y_pick,
                    y_ban,
                    y_role,
                    y_wr,
                ) = [b.to(self.device, non_blocking=True) for b in batch]

                self.opt.zero_grad(set_to_none=True)

                with torch.autocast(self.device):

                    champ_logits, role_logits, wr_pred = self.model(
                        X,
                        bp,
                        rp,
                        bb,
                        rb,
                        champ_mask,
                        b_role_mask,
                        r_role_mask,
                        side,
                        phase,
                    )

                    # Champ loss
                    y_champ = torch.where(phase == 1, y_pick, y_ban)
                    loss_c = self.loss_champ(champ_logits, y_champ)
                    total_loss_c += loss_c.item()

                    mask = phase == 1

                    # role loss and winrate loss
                    if mask.any():
                        loss_r = self.loss_role(role_logits[mask], y_role[mask])
                        total_loss_r += loss_r.item()
                        count_role_batches += 1

                        loss_w = self.loss_wr(wr_pred[mask].squeeze(-1), y_wr[mask])
                        total_loss_w += loss_w.item()
                        count_wr_batches += 1
                    else:
                        loss_r = 0.0
                        loss_w = 0.0

                    loss = loss_c + 0.5 * loss_r + 0.2 * loss_w

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                self.scheduler.step()

                total_loss += loss.item()

            avg_total = total_loss / len(self.train_loader)
            avg_c = total_loss_c / len(self.train_loader)
            avg_r = total_loss_r / max(1, count_role_batches)
            avg_w = total_loss_w / max(1, count_wr_batches)

            self.logger.info(f"Training losses for Epoch {epoch + 1}:")
            self.logger.info(f"champ_loss={avg_c:.4f}")
            self.logger.info(f"role_loss={avg_r:.4f}")
            self.logger.info(f"wr_loss={avg_w:.4f}")
            self.logger.info(f"total_loss={avg_total:.4f}")

            self.evaluate(epoch=epoch)

    def evaluate(self, epoch):

        self.logger.info(f"Started validation for Epoch {epoch+1}")
        self.model.eval()

        with torch.no_grad(), torch.autocast(self.device):

            total_loss = 0.0
            total_loss_c = 0.0
            total_loss_r = 0.0
            total_loss_w = 0.0

            count_role_batches = 0
            count_wr_batches = 0

            for batch in tqdm(self.val_loader):

                (
                    X,
                    bp,
                    rp,
                    bb,
                    rb,
                    champ_mask,
                    b_role_mask,
                    r_role_mask,
                    side,
                    phase,
                    y_pick,
                    y_ban,
                    y_role,
                    y_wr,
                ) = [b.to(self.device, non_blocking=True) for b in batch]

                # forward
                champ_logits, role_logits, wr_pred = self.model(
                    X, bp, rp, bb, rb, champ_mask, b_role_mask, r_role_mask, side, phase
                )

                # champion loss
                y_champ = torch.where(phase == 1, y_pick, y_ban)
                loss_c = self.loss_champ(champ_logits, y_champ)
                total_loss_c += loss_c.item()

                # phase mask
                mask = phase == 1

                # role loss and winrate loss
                if mask.any():
                    loss_r = self.loss_role(role_logits[mask], y_role[mask])
                    total_loss_r += loss_r.item()
                    count_role_batches += 1

                    loss_w = self.loss_wr(wr_pred[mask].squeeze(-1), y_wr[mask])
                    total_loss_w += loss_w.item()
                    count_wr_batches += 1
                else:
                    loss_r = 0.0
                    loss_w = 0.0

                loss = loss_c + 0.5 * loss_r + 0.2 * loss_w
                total_loss += loss.item()

            avg_total = total_loss / len(self.val_loader)
            avg_c = total_loss_c / len(self.val_loader)
            avg_r = total_loss_r / max(1, count_role_batches)
            avg_w = total_loss_w / max(1, count_wr_batches)

            self.logger.info(f"Validation losses for Epoch {epoch + 1}:")
            self.logger.info(f"champ_loss={avg_c:.4f}")
            self.logger.info(f"role_loss={avg_r:.4f}")
            self.logger.info(f"wr_loss={avg_w:.4f}")
            self.logger.info(f"total_loss={avg_total:.4f}")

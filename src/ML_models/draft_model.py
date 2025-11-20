import math
from typing import Any
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        step_embed_size=16,
        max_steps=25,
    ) -> None:
        super().__init__()

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")
        self.mode = mode
        team_size = num_roles

        if mode == "learnable":
            self.champ_embedding = nn.Embedding(
                num_champions + 1,
                embed_size,
                padding_idx=num_champions,
            )
            # Embed normalization
            self.embed_norm = nn.LayerNorm(embed_size)
            # Dropout on every embed values
            self.embed_dropout = nn.Dropout(0.1)
            # Global normalization for team
            self.team_norm = nn.LayerNorm(team_size * embed_size)
            self.team_dropout = nn.Dropout(dropout)

            self.step_embedding = nn.Embedding(max_steps, step_embed_size)

            # 20 picks/bans + side
            input_dim = 4 * team_size * embed_size + 1 + step_embed_size

        self.logger.info(f"Input dim for {mode}: {input_dim}")

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.pick_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_champions + 1),
        )

        self.ban_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_champions + 1),
        )
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, num_roles),
        )

        self.wr_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Initialize weights sensibly
        self._init_weights()

    def encode_team(self, champ_ids):
        # Embeddings (B, 5, E)
        emb = self.champ_embedding(champ_ids)
        emb = self.embed_norm(emb)
        emb = self.embed_dropout(emb)
        # flatten (B, 5E)
        emb = emb.reshape(emb.shape[0], -1)
        emb = self.team_norm(emb)
        emb = self.team_dropout(emb)
        return emb

    def _init_weights(self):
        # Kaiming for linear layers, normal for embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)

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
        step,
        side,
        phase,
    ):

        if self.mode == "learnable":
            blue_picks_emb = self.encode_team(blue_picks)
            red_picks_emb = self.encode_team(red_picks)
            blue_bans_emb = self.encode_team(blue_bans)
            red_bans_emb = self.encode_team(red_bans)
            step_emb = self.step_embedding(step.long())

            X = torch.cat(
                [
                    blue_picks_emb,
                    red_picks_emb,
                    blue_bans_emb,
                    red_bans_emb,
                    side,
                    step_emb,
                ],
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
        experiment_name="draft_v1",
    ) -> None:

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

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
            self.model.parameters(), lr=scaled_lr, weight_decay=3e-2
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

    def calculate_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, k_list=[1, 5]
    ):
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

    def train(self):

        self.model.train()

        for epoch in range(self.num_epochs):
            self.logger.info(
                f"Started training for Epoch {epoch + 1}/{self.num_epochs}"
            )

            epoch_metrics = {
                "loss": 0.0,
                "loss_c": 0.0,
                "loss_r": 0.0,
                "loss_w": 0.0,
                "acc_c_1": 0.0,  # Champion Accuracy
                "acc_c_5": 0.0,  # Champion Accuracy
                "acc_r": 0.0,  # Role Accuracy
                "wr_mae": 0.0,  # Winrate Mean Absolute Error
                "count_c": 0,
                "count_r": 0,
                "count_w": 0,
            }

            pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch+1}")

            for batch in pbar:

                (
                    X,
                    bp,
                    rp,
                    bb,
                    rb,
                    champ_mask,
                    b_role_mask,
                    r_role_mask,
                    step_idx,
                    side,
                    phase,
                    y_pick,
                    y_ban,
                    y_role,
                    y_wr,
                ) = [b.to(self.device, non_blocking=True) for b in batch]

                self.opt.zero_grad()

                with torch.autocast(self.device):

                    champ_logits, role_logits, wr_logits = self.model(
                        X,
                        bp,
                        rp,
                        bb,
                        rb,
                        champ_mask,
                        b_role_mask,
                        r_role_mask,
                        step_idx,
                        side,
                        phase,
                    )

                    # Champ loss
                    y_champ = torch.where(phase == 1, y_pick, y_ban)
                    loss_c = self.loss_champ(champ_logits, y_champ)

                    # Role loss and winrate loss
                    mask = phase == 1

                    loss_r = torch.tensor(0.0, device=self.device)
                    loss_w = torch.tensor(0.0, device=self.device)

                    if mask.any():
                        loss_r = self.loss_role(role_logits[mask], y_role[mask])
                        loss_w = self.loss_wr(wr_logits[mask].squeeze(-1), y_wr[mask])

                    loss = loss_c + 0.5 * loss_r + 0.5 * loss_w

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()

                batch_size = X.size(0)

                # Champion Accuracy
                acc_res = self.calculate_metrics(champ_logits, y_champ, k_list=[1, 5])
                epoch_metrics["acc_c_1"] += acc_res[1]
                epoch_metrics["acc_c_5"] += acc_res[5]
                epoch_metrics["count_c"] += batch_size

                # Role Accuracy & Winrate Diff
                if mask.any():
                    n_picks = mask.sum().item()

                    # Role
                    acc_r_res = self.calculate_metrics(
                        role_logits[mask], y_role[mask], k_list=[1]
                    )
                    epoch_metrics["acc_r"] += acc_r_res[1]
                    epoch_metrics["count_r"] += n_picks

                    # Winrate: logit conversion
                    probs_wr = torch.sigmoid(wr_logits[mask].squeeze(-1))
                    # MAE: (|pred - real|)
                    mae = torch.abs(probs_wr - y_wr[mask]).sum().item()
                    epoch_metrics["wr_mae"] += mae
                    epoch_metrics["count_w"] += n_picks

                # Accumulate losses
                epoch_metrics["loss"] += loss.item() * batch_size
                epoch_metrics["loss_c"] += loss_c.item() * batch_size
                epoch_metrics["loss_r"] += (
                    loss_r.item() * batch_size if mask.any() else 0
                )
                epoch_metrics["loss_w"] += (
                    loss_w.item() * batch_size if mask.any() else 0
                )

            avg_metrics = {
                "loss": epoch_metrics["loss"] / epoch_metrics["count_c"],
                "acc_champ_top1": epoch_metrics["acc_c_1"] / epoch_metrics["count_c"],
                "acc_champ_top5": epoch_metrics["acc_c_5"] / epoch_metrics["count_c"],
                "acc_role": epoch_metrics["acc_r"] / max(1, epoch_metrics["count_r"]),
                "wr_mae": epoch_metrics["wr_mae"] / max(1, epoch_metrics["count_w"]),
            }

            for k, v in avg_metrics.items():
                self.writer.add_scalar(f"Train/{k}", v, epoch)

            self.logger.info(
                f"Train Loss: {avg_metrics['loss']:.4f} | "
                f"Top1: {avg_metrics['acc_champ_top1']:.2%} | "
                f"WR Error: {avg_metrics['wr_mae']:.2%}"
            )

            # Validation
            self.evaluate(epoch)

    def evaluate(self, epoch):

        self.logger.info(f"Started validation for Epoch {epoch+1}")
        self.model.eval()
        val_metrics = {
            "loss": 0.0,
            "acc_c_1": 0.0,
            "acc_c_5": 0.0,
            "acc_r": 0.0,
            "wr_mae": 0.0,
            "count_c": 0,
            "count_r": 0,
            "count_w": 0,
        }

        with torch.no_grad(), torch.autocast(self.device):
            for batch in tqdm(self.val_loader, desc=f"Val Ep {epoch+1}"):

                (
                    X,
                    bp,
                    rp,
                    bb,
                    rb,
                    champ_mask,
                    b_role_mask,
                    r_role_mask,
                    step_idx,
                    side,
                    phase,
                    y_pick,
                    y_ban,
                    y_role,
                    y_wr,
                ) = [b.to(self.device, non_blocking=True) for b in batch]

                champ_logits, role_logits, wr_logits = self.model(
                    X,
                    bp,
                    rp,
                    bb,
                    rb,
                    champ_mask,
                    b_role_mask,
                    r_role_mask,
                    step_idx,
                    side,
                    phase,
                )

                # Champ loss
                y_champ = torch.where(phase == 1, y_pick, y_ban)
                loss_c = self.loss_champ(champ_logits, y_champ)

                # phase mask
                mask = phase == 1

                loss_r = 0.0
                loss_w = 0.0

                # Role loss and winrate loss
                if mask.any():
                    loss_r = self.loss_role(role_logits[mask], y_role[mask])
                    loss_w = self.loss_wr(wr_logits[mask].squeeze(-1), y_wr[mask])

                loss = loss_c.item() + 0.5 * loss_r + 0.5 * loss_w

                batch_size = X.size(0)
                val_metrics["loss"] += loss * batch_size
                val_metrics["count_c"] += batch_size

                # Champ Accuracy
                acc_res = self.calculate_metrics(champ_logits, y_champ, k_list=[1, 5])
                val_metrics["acc_c_1"] += acc_res[1]
                val_metrics["acc_c_5"] += acc_res[5]

                # Role Accuracy & Winrate Diff
                if mask.any():
                    n_picks = mask.sum().item()

                    # Role
                    acc_r_res = self.calculate_metrics(
                        role_logits[mask], y_role[mask], k_list=[1]
                    )
                    val_metrics["acc_r"] += acc_r_res[1]
                    val_metrics["count_r"] += n_picks

                    # Winrate
                    probs_wr = torch.sigmoid(wr_logits[mask].squeeze(-1))
                    mae = torch.abs(probs_wr - y_wr[mask]).sum().item()
                    val_metrics["wr_mae"] += mae
                    val_metrics["count_w"] += n_picks

            # Final averages
            avg_val = {
                "loss": val_metrics["loss"] / val_metrics["count_c"],
                "acc_champ_top1": val_metrics["acc_c_1"] / val_metrics["count_c"],
                "acc_champ_top5": val_metrics["acc_c_5"] / val_metrics["count_c"],
                "acc_role": val_metrics["acc_r"] / max(1, val_metrics["count_r"]),
                "wr_mae": val_metrics["wr_mae"] / max(1, val_metrics["count_w"]),
            }

            # LOGGING TENSORBOARD (Validation)
            for k, v in avg_val.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            self.logger.info(f"VAL RESULTS Ep {epoch+1}:")
            self.logger.info(f" > Loss: {avg_val['loss']:.4f}")
            self.logger.info(
                f" > Champ Acc (Top1/Top5): "
                f"{avg_val['acc_champ_top1']:.2%} / {avg_val['acc_champ_top5']:.2%}"
            )
            self.logger.info(f" > Winrate Avg Error: {avg_val['wr_mae']:.2%}")

    def sanity_check(self):
        self.logger.info("Started sanity check (Overfitting 1 Batch)")

        # Getting only one batch
        batch = next(iter(self.train_loader))
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
        ) = [b.to(self.device) for b in batch]

        self.model.train()

        # 5000 epochs on this batch
        for i in range(5000):
            self.opt.zero_grad()

            with torch.autocast(self.device):
                champ_logits, role_logits, wr_logits = self.model(
                    X, bp, rp, bb, rb, champ_mask, b_role_mask, r_role_mask, side, phase
                )

                y_champ = torch.where(phase == 1, y_pick, y_ban)
                loss_c = self.loss_champ(champ_logits, y_champ)

                mask_pick = phase == 1
                loss_r = torch.tensor(0.0, device=self.device)
                loss_w = torch.tensor(0.0, device=self.device)

                if mask_pick.any():
                    loss_r = self.loss_role(role_logits[mask_pick], y_role[mask_pick])
                    loss_w = self.loss_wr(
                        wr_logits[mask_pick].squeeze(-1), y_wr[mask_pick]
                    )

                loss = loss_c + loss_r + loss_w

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            if i % 100 == 0:
                # Check accuracy
                acc = (champ_logits.argmax(dim=1) == y_champ).float().mean()
                probs_wr = torch.sigmoid(wr_logits[mask_pick].squeeze(-1))
                wr_err = torch.abs(probs_wr - y_wr[mask_pick]).mean()

                print(
                    f"Epoch {i}: Loss={loss.item():.4f} | Acc={acc:.2%} "
                    f"| WR Error={wr_err:.4f}"
                )

                if acc > 0.99:
                    print("SUCCESS: Model has learned every detail of the dataset!")
                    return

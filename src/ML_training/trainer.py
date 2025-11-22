from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.ML_models.draft_MLP import DraftMLPModel
from src.ML_training.dataset import DraftDataset
from src.ML_training.utils import MultiTaskLossWrapper, calculate_metrics
from src.utils.logger_config import get_logger
from src.utils.constants import DATA_REPRESENTATION_FOLDER, DRAFT_STATES_TORCH


class TrainerClass:
    """
    Trainer class

    Give a model to train the data on
    """

    def __init__(
        self,
        model,
        data_folder=DATA_REPRESENTATION_FOLDER,
        data_file=DRAFT_STATES_TORCH,
        num_epochs=25,
        batch_size=64,
        base_lr=1e-4,
        warmup_steps=[500, 2000],
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="draft_v1",
    ) -> None:

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

        self.data_file = data_file
        self.data_folder = data_folder

        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

        self.model: DraftMLPModel = model.to(device)

        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
        )
        self.logger.info(f"GPU name: {gpu_name}")

        self.logger.info(f"Succesfully loaded model : {self.model.__class__.__name__}")

        self.__build_dataset()
        self.__training_fns()

        self.logger.info(
            f"Parameters in use : Batch size = {batch_size} | Device = {device} | "
            f"Num epochs = {num_epochs}"
        )

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

                with torch.autocast(self.device, dtype=torch.float32):

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

                    loss = self.mtl_loss(loss_c, loss_r, loss_w)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.scheduler.step()

                batch_size = X.size(0)

                # Champion Accuracy
                acc_res = calculate_metrics(champ_logits, y_champ, k_list=[1, 5])
                epoch_metrics["acc_c_1"] += acc_res[1]
                epoch_metrics["acc_c_5"] += acc_res[5]
                epoch_metrics["count_c"] += batch_size

                # Role Accuracy & Winrate Diff
                if mask.any():
                    n_picks = mask.sum().item()

                    # Role
                    acc_r_res = calculate_metrics(
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

            self.logger.info(f"TRAIN RESULTS Ep {epoch+1}:")
            self.logger.info(f" > Loss: {avg_metrics['loss']:.4f}")
            self.logger.info(
                f" > Champ Acc (Top1/Top5): "
                f"{avg_metrics['acc_champ_top1']:.2%} / "
                f"{avg_metrics['acc_champ_top5']:.2%}"
            )
            self.logger.info(f" > Role Acc : {avg_metrics['acc_role']:.2%}")
            self.logger.info(f" > Winrate Avg Error: {avg_metrics['wr_mae']:.2%}")

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

        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float32):
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

                loss_r = torch.tensor(0.0, device=self.device)
                loss_w = torch.tensor(0.0, device=self.device)

                # Role loss and winrate loss
                if mask.any():
                    loss_r = self.loss_role(role_logits[mask], y_role[mask])
                    loss_w = self.loss_wr(wr_logits[mask].squeeze(-1), y_wr[mask])

                loss = self.mtl_loss(loss_c, loss_r, loss_w)

                batch_size = X.size(0)
                val_metrics["loss"] += loss.item() * batch_size
                val_metrics["count_c"] += batch_size

                # Champ Accuracy
                acc_res = calculate_metrics(champ_logits, y_champ, k_list=[1, 5])
                val_metrics["acc_c_1"] += acc_res[1]
                val_metrics["acc_c_5"] += acc_res[5]

                # Role Accuracy & Winrate Diff
                if mask.any():
                    n_picks = mask.sum().item()

                    # Role
                    acc_r_res = calculate_metrics(
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
            self.logger.info(f" > Role Acc : {avg_val['acc_role']:.2%}")
            self.logger.info(f" > Winrate Avg Error: {avg_val['wr_mae']:.2%}")

    def sanity_check(self):
        self.logger.info("Started sanity check (Overfitting 1 Batch)")

        # Getting only one batch (batch size 64)
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
            step_idx,
            side,
            phase,
            y_pick,
            y_ban,
            y_role,
            y_wr,
        ) = [b.to(self.device) for b in batch]

        self.model.train()

        for i in range(self.num_epochs):
            self.opt.zero_grad()

            with torch.autocast(self.device, dtype=torch.float32):
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

                loss = self.mtl_loss(loss_c, loss_r, loss_w)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            if i % 10 == 0:
                # Check accuracy
                champ_acc = (champ_logits.argmax(dim=1) == y_champ).float().mean()
                role_acc = (role_logits.argmax(dim=1) == y_role).float().mean()
                probs_wr = torch.sigmoid(wr_logits[mask_pick].squeeze(-1))
                wr_err = torch.abs(probs_wr - y_wr[mask_pick]).mean()

                print(
                    f"Epoch {i}: Loss={loss.item():.4f} | Champ Acc={champ_acc:.2%} "
                    f"| Role Acc={role_acc:.2%}"
                    f"| WR Error={wr_err:.4f}"
                )

                if champ_acc > 0.99:
                    self.logger.info(
                        "SUCCESS: Model has learned every detail of the dataset!"
                    )
                    return

    def __build_dataset(self):

        dataset = DraftDataset(self.data_folder, self.data_file)
        idx_train, idx_val = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )

        train_subset = Subset(dataset, idx_train)
        val_subset = Subset(dataset, idx_val)

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.logger.info("Succesfully built dataset")

    def __training_fns(self):

        total_steps = self.num_epochs * len(self.train_loader)

        clamped_warmup_steps = min(
            self.warmup_steps[1],
            max(
                self.warmup_steps[0],
                int(self.warmup_steps[0] * (self.batch_size / 128)),
            ),
        )

        self.logger.info(f"Learning rate : {self.base_lr}")
        self.logger.info(f"Warmup steps : {clamped_warmup_steps}")

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.base_lr, weight_decay=5e-2
        )

        warmup_scheduler = LinearLR(
            self.opt,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=clamped_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            self.opt,
            T_max=(total_steps - clamped_warmup_steps),
            eta_min=0.0,
        )

        self.scheduler = SequentialLR(
            self.opt,
            [
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[clamped_warmup_steps],
        )

        self.scaler = torch.GradScaler(self.device)

        self.loss_champ = nn.CrossEntropyLoss()
        self.loss_role = nn.CrossEntropyLoss()
        self.loss_wr = nn.BCEWithLogitsLoss()

        self.mtl_loss = MultiTaskLossWrapper(num_tasks=3).to(self.device)

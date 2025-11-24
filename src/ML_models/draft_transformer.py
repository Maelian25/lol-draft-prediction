import torch.nn as nn
import torch.nn.functional as F
import torch

from src.ML_models.utils import head_mlp


class DraftTransformer(nn.Module):
    def __init__(
        self,
        num_champions,
        num_roles,
        dim_feedforward=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        d_model=128,
        max_steps=20,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Basic embedding
        self.champ_embedding = nn.Embedding(
            num_champions + 1, d_model, padding_idx=num_champions
        )
        self.pos_embedding = nn.Embedding(20, d_model)

        self.step_embedding = nn.Embedding(max_steps, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads
        self.pick_head = head_mlp("pick", d_model, num_champions + 1)

        self.role_head = head_mlp("role", d_model, num_roles)

        self.wr_head = head_mlp("wr", d_model, 1)

    def forward(
        self,
        _,
        blue_picks,
        red_picks,
        blue_bans,
        red_bans,
        champ_mask,
        b_role_mask,
        r_role_mask,
        step_idx,
        side,
        __,
    ):
        batch_size = blue_picks.size(0)

        # Champion sequence
        input_ids = torch.cat([blue_picks, red_picks, blue_bans, red_bans], dim=1)
        x = self.champ_embedding(input_ids.long())

        # Positional Embedding
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(
            0
        )
        x = x + self.pos_embedding(positions)

        # Step Embedding
        # step_emb shape: (Batch, Embed_Size)
        step_emb = self.step_embedding(step_idx.long())

        # x shape: (Batch, 20, Embed)
        # step_emb.unsqueeze(1) shape: (Batch, 1, Embed)
        x = x + step_emb.unsqueeze(1)

        # CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + step_emb.unsqueeze(1)

        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer
        x_out = self.transformer(x)

        # Preds
        context = x_out[:, 0, :]  # Cls token with step embedding

        wr_pred = self.wr_head(context)
        champ_logits = self.pick_head(context)
        role_logits = self.role_head(context)

        # Mask
        champ_logits = self.__apply_mask(champ_logits, champ_mask)
        final_role_mask = torch.where(side == 1, b_role_mask, r_role_mask)
        role_logits = self.__apply_mask(role_logits, final_role_mask)

        return champ_logits, role_logits, wr_pred

    def __apply_mask(self, logits: torch.Tensor, mask: torch.Tensor):

        unavailable = mask == 0

        logits = logits.masked_fill(unavailable, -1.0e9)

        return logits

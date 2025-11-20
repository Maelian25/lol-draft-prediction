import math
import torch
import torch.nn as nn
from src.ML_models.utils import shared_mlp, three_layers_mlp, two_layers_mlp
from src.utils.logger_config import get_logger


class DraftMLPModel(nn.Module):
    """
    MLP Model using only linear layers

    First model to try and see if the model can learn how to draft
    """

    def __init__(
        self,
        num_champions,
        num_roles,
        mode,
        hidden_dim=1024,
        dropout=0.3,
        input_dim=0,
        embed_size=96,
        step_embed_size=16,
        max_steps=20,
    ) -> None:
        super().__init__()

        self.logger = get_logger(self.__class__.__name__, "draft_training.log")

        self.mode = mode
        self.team_size = num_roles
        self.num_champions = num_champions
        self.embed_size = embed_size
        self.step_embed_size = step_embed_size
        self.max_steps = max_steps
        self.dropout = dropout

        if mode == "learnable":
            self.__init_embeddings()

            # 20 picks/bans + side + step_embedding
            input_dim = 4 * self.team_size * embed_size + 1 + step_embed_size

        self.logger.info(f"Input dim for {mode}: {input_dim}")

        self.shared = shared_mlp(input_dim, hidden_dim, dropout)

        self.pick_head = three_layers_mlp(hidden_dim, dropout, num_champions + 1)
        self.ban_head = three_layers_mlp(hidden_dim, dropout, num_champions + 1)

        self.role_head = two_layers_mlp(hidden_dim, dropout, num_roles)
        self.wr_head = two_layers_mlp(hidden_dim, dropout, 1)

        # Initialize weights sensibly
        self.__init_weights()

    def forward(
        self,
        X_static,
        bp,
        rp,
        bb,
        rb,
        champ_mask,
        b_role_mask,
        r_role_mask,
        step,
        side,
        phase,
    ):

        if self.mode == "learnable":
            blue_picks_emb = self.__encode_team(bp)
            red_picks_emb = self.__encode_team(rp)
            blue_bans_emb = self.__encode_team(bb)
            red_bans_emb = self.__encode_team(rb)
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

    def __encode_team(self, champ_ids):
        # Embeddings (B, 5, E)
        emb = self.champ_embedding(champ_ids)
        emb = self.embed_norm(emb)
        emb = self.embed_dropout(emb)
        # flatten (B, 5E)
        emb = emb.reshape(emb.shape[0], -1)
        emb = self.team_norm(emb)
        emb = self.team_dropout(emb)
        return emb

    def __init_weights(self):
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

    def __init_embeddings(self):
        self.champ_embedding = nn.Embedding(
            self.num_champions + 1,
            self.embed_size,
            padding_idx=self.num_champions,
        )
        # Embed normalization
        self.embed_norm = nn.LayerNorm(self.embed_size)
        # Dropout on every embed values
        self.embed_dropout = nn.Dropout(0.1)
        # Global normalization for team
        self.team_norm = nn.LayerNorm(self.team_size * self.embed_size)
        self.team_dropout = nn.Dropout(self.dropout)

        self.step_embedding = nn.Embedding(self.max_steps, self.step_embed_size)

    def __apply_mask(self, logits: torch.Tensor, mask: torch.Tensor):

        unavailable = mask == 0

        logits = logits.masked_fill(unavailable, float("-inf"))

        return logits

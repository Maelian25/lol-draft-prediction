import math
import torch
import torch.nn as nn
from src.ML_models.utils import head_mlp
from src.utils.logger_config import get_logger


class ResidualBlock(nn.Module):
    """
    Residual block to allow modle to be deep without losing too much info

    Structure : Input -> [Linear -> BN -> GELU -> Dropout]
                      -> [Linear -> BN] + Input
                      -> Gelu
    """

    def __init__(self, hidden_dim, dropout=0.3) -> None:
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        output = self.linear1(x)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.dropout(output)

        output = self.linear2(output)
        output = self.bn2(output)

        output += residual
        output = self.activation(output)

        return output


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
        hidden_dim=512,
        dropout=0.15,
        input_dim=0,
        num_res_blocks=1,
        embed_size=64,
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
        self.side_embed_size = step_embed_size
        self.max_steps = max_steps
        self.dropout = dropout

        if mode == "learnable":
            self.__init_embeddings()

            # 20 picks/bans + side_embedding + step_embedding
            input_dim = (
                4 * self.team_size * embed_size + step_embed_size + step_embed_size
            )

        self.logger.info(f"Input dim using {mode} mode : {input_dim}")

        # Backbone (shared representation)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        self.pick_head = head_mlp("pick", hidden_dim, num_champions + 1)
        self.ban_head = head_mlp("ban", hidden_dim, num_champions + 1)

        self.role_head = head_mlp("role", hidden_dim, num_roles)
        self.wr_head = head_mlp("wr", hidden_dim, 1)

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
            bp_emb = self.champ_embedding(bp).flatten(1)
            rp_emb = self.champ_embedding(rp).flatten(1)
            bb_emb = self.champ_embedding(bb).flatten(1)
            rb_emb = self.champ_embedding(rb).flatten(1)
            step_emb = self.step_embedding(step.long())
            side_embed = self.side_embedding(side.squeeze(1))

            X = torch.cat(
                [
                    bp_emb,
                    rp_emb,
                    bb_emb,
                    rb_emb,
                    side_embed,
                    step_emb,
                ],
                dim=1,
            )

        else:
            X = X_static

        out = self.input_projection(X)
        for block in self.blocks:
            out = block(out)

        pick_logits = self.pick_head(out)
        ban_logits = self.ban_head(out)
        role_logits = self.role_head(out)
        winrate = self.wr_head(out)

        champ_logits = torch.where(phase.unsqueeze(1) == 1, pick_logits, ban_logits)
        champ_logits = self.__apply_mask(champ_logits, champ_mask)

        role_mask = torch.where(
            side == 1,
            b_role_mask,
            r_role_mask,
        )

        role_logits = self.__apply_mask(role_logits, role_mask)

        return champ_logits, role_logits, winrate

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

        self.step_embedding = nn.Embedding(self.max_steps, self.step_embed_size)
        self.side_embedding = nn.Embedding(2, self.side_embed_size)

    def __apply_mask(self, logits: torch.Tensor, mask: torch.Tensor):

        unavailable = mask == 0

        logits = logits.masked_fill(unavailable, -1.0e9)

        return logits

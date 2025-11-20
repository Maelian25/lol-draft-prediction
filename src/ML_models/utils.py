import torch.nn as nn


# MLP Model blocks
def two_layers_mlp(hidden_dim, dropout, output_size):

    mlp_block = nn.Sequential(
        nn.Linear(hidden_dim // 2, hidden_dim // 4),
        nn.LayerNorm(hidden_dim // 4),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_dim // 4, output_size),
    )
    return mlp_block


def three_layers_mlp(hidden_dim, dropout, output_size):

    mlp_block = nn.Sequential(
        nn.Linear(hidden_dim // 2, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, hidden_dim // 4),
        nn.LayerNorm(hidden_dim // 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 4, output_size),
    )
    return mlp_block


def shared_mlp(input_dim, hidden_dim, dropout):

    mlp_block = nn.Sequential(
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
    return mlp_block

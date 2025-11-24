import torch.nn as nn


def head_mlp(type: str, hidden_dim, output_size):

    match (type):
        case "pick" | "ban":
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_size),
            )
        case "role":
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, output_size),
            )
        case "wr":
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, output_size),
            )
        case _:
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_size),
            )

    return mlp

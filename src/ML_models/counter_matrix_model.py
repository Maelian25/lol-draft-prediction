import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.constants import BT_MODEL, MODELS_PARAMETER_FOLDER
from src.utils.general_helper import save_model


class BTFeature_Embedding_Model(nn.Module):
    """
    Model following Bradley–Terry to get probability of winning
    according to matchups

    Helps build counter matrix
    """

    def __init__(self, input_dim, num_champs, embed_dim=16):
        super().__init__()
        self.feat_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.embed = nn.Embedding(num_champs, embed_dim)
        self.embed_bias = nn.Embedding(num_champs, embed_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_1, x_2, idx_1, idx_2):
        """
        Calculate first the difference of strength based on static features
        Then proceed to calculate the difference of strength based on learnable embeds
        """

        scale = torch.clamp(self.scale, 0.01, 2.0)

        x_diff = x_1 - x_2
        x_mult = x_1 * x_2
        feat_input = torch.cat([x_diff, x_mult], dim=1)
        feat_term = self.feat_mlp(feat_input)

        e_1 = self.embed(idx_1)
        e_2 = self.embed(idx_2)

        b_1 = self.embed_bias(idx_1)
        b_2 = self.embed_bias(idx_2)

        inter_term = (e_1 * (e_2 + b_1 - b_2)).sum(dim=1, keepdim=True)
        logits = feat_term + scale * inter_term

        return logits


class BTFeatureCounter:
    """
    Bradley–Terry training
    Helps build counter matrix
    """

    def __init__(self, input_dim, num_champs, embed_dim=16, device="cpu"):
        self.device = device
        self.num_champs = num_champs
        self.model = nn.Module()
        self.model = BTFeature_Embedding_Model(input_dim, num_champs, embed_dim).to(
            device
        )

    def train(self, X_1, X_2, idx_1, idx_2, target, weight, num_epochs=1000, lr=1e-3):
        """
        Train small dataset (~500 duels per champ pair)

        """
        X_1, X_2, idx_1, idx_2, target, weight = [
            t.to(self.device) for t in [X_1, X_2, idx_1, idx_2, target, weight]
        ]

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(weight=weight)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(X_1, X_2, idx_1, idx_2).view(-1)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss : {loss.item():.6f}")
            if epoch % num_epochs == 0:
                save_model(self.model.state_dict(), MODELS_PARAMETER_FOLDER, BT_MODEL)

    def evaluate(self, champ_features, champ_id_to_idx):
        """
        Evaluate model and return a dataframe containing every counter data
        This should allow counter score calculation within a game
        """
        self.model.eval()
        champs = list(champ_features.keys())
        n = len(champs)
        features_dict = torch.tensor(
            np.stack([champ_features[c] for c in champs]), dtype=torch.float32
        )
        indexes = torch.tensor([champ_id_to_idx[c] for c in champs], dtype=torch.long)
        P = np.zeros((n, n), dtype=np.float32)
        with torch.no_grad():
            for a in range(n):
                features_1 = features_dict[a : a + 1].to(self.device)
                idx_1 = indexes[a : a + 1].to(self.device)
                for b in range(n):
                    features_2 = features_dict[b : b + 1].to(self.device)
                    idx_2 = indexes[b : b + 1].to(self.device)
                    p = torch.sigmoid(
                        self.model(features_1, features_2, idx_1, idx_2)
                    ).item()
                    P[a, b] = p

        return pd.DataFrame(P, index=champs, columns=champs)

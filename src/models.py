import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class BTFeature_Embedding_Model(nn.Module):
    """
    Model following Bradley–Terry to get probability of winning
    according to matchups

    Helps build counter matrix
    """

    def __init__(self, input_dim, num_champs, embed_dim=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.embed = nn.Embedding(num_champs, embed_dim)
        self.scale = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x_1, x_2, idx_1, idx_2):
        feat_term = self.linear(x_1 - x_2)

        e_1 = self.embed(idx_1)
        e_2 = self.embed(idx_2)

        inter_term = (e_1 * e_2).sum(dim=1, keepdim=True)
        logits = feat_term + self.scale * inter_term

        return logits


class BTFeatureCounter:
    """
    Bradley–Terry + Feature + Embedding model
    Helps build counter matrix
    """

    def __init__(
        self, input_dim, num_champs, embed_dim=12, scale_init=0.5, device="cpu"
    ):
        self.device = device
        self.num_champs = num_champs
        self.model = nn.Module()
        self.model = BTFeature_Embedding_Model(input_dim, num_champs, embed_dim).to(
            device
        )

    def train(self, X_1, X_2, idx_1, idx_2, target, weight, num_epochs=1000, lr=1e-3):
        X_1, X_2, idx_1, idx_2, target = [
            t.to(self.device) for t in [X_1, X_2, idx_1, idx_2, target]
        ]
        weight = weight.to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(weight=weight)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            logits = torch.clamp(self.model(X_1, X_2, idx_1, idx_2).view(-1), -20, 20)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss : {loss.item():.6f}")

    def counter_matrix(self, champ_features, champ_id_to_idx):
        self.model.eval()
        champs = list(champ_features.keys())
        n = len(champs)
        X = torch.tensor(
            np.stack([champ_features[c] for c in champs]), dtype=torch.float32
        )
        ids = torch.tensor([champ_id_to_idx[c] for c in champs], dtype=torch.long)
        P = np.zeros((n, n), dtype=np.float32)
        with torch.no_grad():
            for a in range(n):
                xa = X[a : a + 1].to(self.device)
                ida = ids[a : a + 1].to(self.device)
                for b in range(n):
                    xb = X[b : b + 1].to(self.device)
                    idb = ids[b : b + 1].to(self.device)
                    p = torch.sigmoid(self.model(xa, xb, ida, idb)).item()
                    P[a, b] = p
        return pd.DataFrame(P, index=champs, columns=champs)

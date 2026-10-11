"""The IDS model: a small MLP for binary attack/benign classification."""

import os

import torch
import torch.nn as nn

# 128 is the campaign model; other widths are only for the model-size study.
MODEL_WIDTH = int(os.getenv("MODEL_WIDTH", "128"))


class IDSModel(nn.Module):

    def __init__(self, input_dim: int, width: int = MODEL_WIDTH):
        super().__init__()
        h1, h2, h3 = width, width // 2, width // 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(0.3),
            nn.Linear(h1, h2),        nn.ReLU(), nn.BatchNorm1d(h2), nn.Dropout(0.3),
            nn.Linear(h2, h3),        nn.ReLU(), nn.BatchNorm1d(h3), nn.Dropout(0.3),
            nn.Linear(h3, 1),         nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

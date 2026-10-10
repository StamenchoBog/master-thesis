"""The IDS model: a small MLP for binary attack/benign classification."""

import os

import torch
import torch.nn as nn

# First hidden width; the other two are half and a quarter of it. 128 is the model of the
# main campaign. Other widths are only used by the model-size study (protocol Addendum A).
MODEL_WIDTH = int(os.getenv("MODEL_WIDTH", "128"))


class IDSModel(nn.Module):
    """MLP binary classifier: outputs the probability that a network flow is an attack."""

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

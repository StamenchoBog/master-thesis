import torch
import torch.nn as nn


class IDSModel(nn.Module):
    """MLP binary classifier: outputs the probability that a network flow is an attack."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32),  nn.Dropout(0.3),
            nn.Linear(32, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

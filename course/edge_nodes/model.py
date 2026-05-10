import torch
import torch.nn as nn


class IDS_Model(nn.Module):
    """MLP binary classifier: outputs the probability that a network flow is an attack."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        """Build the network as a stack of Linear → ReLU → BatchNorm → Dropout blocks.

        Args:
            input_dim:   Number of input features (determined at runtime from the dataset).
            hidden_dims: Sizes of the hidden layers. Defaults to [128, 64, 32].
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.3)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return a probability in [0, 1] per sample."""
        return self.net(x)

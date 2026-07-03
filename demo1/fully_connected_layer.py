import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, in_features: int = 512, hidden: int = 1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def inputs(self, batch_size: int = 256) -> tuple:
        return (torch.randn(batch_size, self.layers[0].in_features),)
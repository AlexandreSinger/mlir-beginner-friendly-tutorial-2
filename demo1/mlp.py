import torch
import torch.nn as nn


class SampleMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def inputs(self, batch_size: int = 1) -> tuple:
        return (torch.randn(batch_size, self.layers[0].in_features),)
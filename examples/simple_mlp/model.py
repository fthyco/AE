"""Sample only; SAE does not import this module."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

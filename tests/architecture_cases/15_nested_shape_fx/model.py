"""Fixed nested composite shapes for independent FX verification."""
import torch
from torch import nn


class Inner(nn.Module):
    def forward(self, x):
        a = torch.ones(2, 3)
        b = torch.ones(3, 4)
        return torch.matmul(a, b)


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def forward(self, x):
        y = self.inner(x)
        z = torch.ones(2, 6)
        return torch.cat([y, z], dim=1)

"""Module call sequence for independent FX site and edge verification."""
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        a = self.fc1(x)
        b = self.relu(a)
        return self.fc2(b)

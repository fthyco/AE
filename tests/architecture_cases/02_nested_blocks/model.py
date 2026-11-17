from torch import nn
import torch.nn.functional as F
class BlockA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32,64)
        self.norm = nn.LayerNorm(64)
    def forward(self,x):
        return self.norm(F.relu(self.proj(x)))
class BlockB(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64,64)
        self.norm = nn.LayerNorm(64)
    def forward(self,x):
        return self.norm(F.gelu(self.proj(x)))
class BlockC(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64,16)
        self.head = nn.Linear(16,4)
    def forward(self,x):
        h = F.relu(self.proj(x))
        return self.head(h)
class DeepStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = BlockA()
        self.b = BlockB()
        self.c = BlockC()
    def forward(self,x):
        return self.c(self.b(self.a(x)))

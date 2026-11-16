from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32):
        super().__init__()
        self.proj=nn.Linear(input_dim, hidden_dim)
        self.head=nn.Linear(hidden_dim, 4)
    def forward(self,x):
        hidden=F.gelu(self.proj(x))
        return self.head(hidden)

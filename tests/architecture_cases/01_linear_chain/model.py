from torch import nn
import torch.nn.functional as F
class LinearChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32,64)
        self.l2 = nn.Linear(64,96)
        self.l3 = nn.Linear(96,48)
        self.head = nn.Linear(48,5)
    def forward(self,x):
        a = F.relu(self.l1(x))
        b = F.gelu(self.l2(a))
        c = F.relu(self.l3(b))
        return self.head(c)

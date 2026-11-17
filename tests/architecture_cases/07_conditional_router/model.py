from torch import nn
import torch.nn.functional as F
class ConditionalRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a=nn.Linear(16,16)
        self.branch_b=nn.Linear(16,16)
        self.out=nn.Linear(16,3)
    def forward(self,x,use_a):
        if use_a:
            h=F.relu(self.branch_a(x))
        else:
            h=F.relu(self.branch_b(x))
        return self.out(h)

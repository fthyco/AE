from torch import nn
import torch.nn.functional as F
class ReusedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Linear(24,24)
        self.out=nn.Linear(24,2)
    def forward(self,left,right):
        a=F.relu(self.shared(left))
        b=F.relu(self.shared(right))
        joined=a+b
        return self.out(joined)

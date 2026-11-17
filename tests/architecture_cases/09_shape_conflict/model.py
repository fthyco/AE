from torch import nn
class ShapeConflict(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(16,12)
        self.second=nn.Linear(20,3)
    def forward(self,x):
        z=self.first(x)
        return self.second(z)

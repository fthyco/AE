from torch import nn
from missing_file import Encoder
class MissingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder(128)
        self.head=nn.Linear(128,10)
    def forward(self,x):
        z=self.encoder(x)
        return self.head(z)

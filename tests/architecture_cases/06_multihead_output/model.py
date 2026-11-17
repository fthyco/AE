from torch import nn
import torch.nn.functional as F
class SharedStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin=nn.Linear(40,32)
        self.norm=nn.LayerNorm(32)
    def forward(self,x):
        return self.norm(F.relu(self.lin(x)))
class MultiHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=SharedStem()
        self.classifier=nn.Linear(32,7)
        self.regressor=nn.Linear(32,1)
    def forward(self,x):
        h=self.stem(x)
        logits=self.classifier(h)
        score=self.regressor(h)
        return logits,score

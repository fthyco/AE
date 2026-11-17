import torch
from torch import nn
import torch.nn.functional as F
class AttentionCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv=nn.Linear(64,192)
        self.out=nn.Linear(64,64)
    def forward(self,x):
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        scores=torch.matmul(q,k.transpose(-1,-2))
        weights=F.softmax(scores,dim=-1)
        mixed=torch.matmul(weights,v)
        return self.out(mixed)
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(128,64)
        self.core=AttentionCore()
        self.head=nn.Linear(64,128)
    def forward(self,token_ids):
        x=self.embed(token_ids)
        return self.head(self.core(x))

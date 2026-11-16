import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.heads=heads
        self.qkv=nn.Linear(dim,dim*3)
        self.out_proj=nn.Linear(dim,dim)
    def forward(self,x):
        qkv=self.qkv(x)
        q,k,v=qkv.chunk(3,dim=-1)
        scores=torch.matmul(q,k.transpose(-1,-2))
        probs=F.softmax(scores,dim=-1)
        result=torch.matmul(probs,v)
        return self.out_proj(result)

from torch import nn
from .components import Attention

class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.attn=Attention(dim,heads)
        self.norm=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,x):
        hidden=self.norm(self.attn(x)+x)
        return self.ff(hidden)+hidden

class Transformer(nn.Module):
    def __init__(self,vocab_size=64,dim=32,heads=4):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,dim)
        self.block=Block(dim,heads)
        self.head=nn.Linear(dim,vocab_size)
    def forward(self,tokens):
        x=self.embed(tokens)
        return self.head(self.block(x))

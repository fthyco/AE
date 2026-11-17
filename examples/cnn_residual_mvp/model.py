from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    def forward(self,x):
        h=F.relu(self.conv1(x))
        h=self.conv2(h)
        return h+x

class CNN(nn.Module):
    def __init__(self,channels=16):
        super().__init__()
        self.stem=nn.Conv2d(3,channels,kernel_size=3,padding=1)
        self.block=ResidualBlock(channels)
        self.head=nn.Linear(channels,2)
    def forward(self,x):
        h=self.block(self.stem(x))
        pooled=h.mean(dim=(2,3))
        return self.head(pooled)

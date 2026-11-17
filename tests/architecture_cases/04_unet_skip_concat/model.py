from torch import nn
import torch
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(1,16,3,padding=1)
        self.c2=nn.Conv2d(16,32,3,stride=2,padding=1)
    def forward(self,image):
        skip=F.relu(self.c1(image))
        encoded=F.relu(self.c2(skip))
        return encoded,skip
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(48,16,3,padding=1)
        self.c2=nn.Conv2d(16,1,1)
    def forward(self,x,skip):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        fused=torch.cat([up,skip],dim=1)
        return self.c2(F.relu(self.c1(fused)))
class UNetMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
    def forward(self,image):
        encoded,skip=self.encoder(image)
        return self.decoder(encoded,skip)

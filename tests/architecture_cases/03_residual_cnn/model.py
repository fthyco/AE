from torch import nn
import torch.nn.functional as F
class ResUnitA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,8,3,padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,8,3,padding=1)
        self.skip = nn.Conv2d(3,8,1)
    def forward(self,x):
        shortcut = self.skip(x)
        z = F.relu(self.bn1(self.conv1(x)))
        return self.conv2(z) + shortcut
class ResUnitB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8,16,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,16,3,padding=1)
        self.skip = nn.Conv2d(8,16,1,stride=2)
    def forward(self,x):
        shortcut = self.skip(x)
        hidden = F.relu(self.conv1(x))
        return self.conv2(hidden) + shortcut
class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = ResUnitA()
        self.b = ResUnitB()
        self.head = nn.Linear(16,10)
    def forward(self,image):
        features = self.b(self.a(image))
        pooled = features.mean(dim=(2,3))
        return self.head(pooled)

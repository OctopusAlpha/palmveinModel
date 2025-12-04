import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class PalmVeinNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True):
        super(PalmVeinNet, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet18(weights=weights)
        
        # Remove the original FC layer
        # self.base_model.fc = nn.Identity() # Don't do this if we want to use the features before FC easily
        # Instead, we just take the children.
        
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Improved projection head (MLP)
        # ResNet18 output before FC is 512 channels
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        
        # L2 Normalize embedding
        x = F.normalize(x, p=2, dim=1)
        return x

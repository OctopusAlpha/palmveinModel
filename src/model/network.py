import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
from src.model.parts import DOConv2d, SEBlock, CBAM

def replace_conv_with_doconv(module):
    """
    Recursively replace nn.Conv2d with DOConv2d.
    Also inserts SEBlock after BatchNorm if possible (simplistic approach) or manually add SEBlocks.
    For ResNet, it's better to modify the BasicBlock.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Create DOConv2d with same parameters
            doconv = DOConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None
            )
            # Initialize weights (optional but good)
            # For now, just replacing
            setattr(module, name, doconv)
        else:
            replace_conv_with_doconv(child)

class CBAMResNetBasicBlock(nn.Module):
    """
    Modified BasicBlock with DO-Conv and CBAM.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CBAMResNetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # DO-Conv1
        self.conv1 = DOConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # DO-Conv2
        self.conv2 = DOConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        
        # CBAM Block
        self.cbam = CBAM(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply CBAM
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PalmVeinNet(nn.Module):
    def __init__(self, embedding_size=512, num_classes=None, pretrained=True):
        super(PalmVeinNet, self).__init__()
        
        # SOTA Approach: Use ResNet50 (or 101) for stronger features
        # And we use ArcFace, so the output should be features (not normalized here, but usually normalized in ArcFace)
        # But for Metric Learning inference, we output normalized features.
        
        # We will use ResNet50 this time.
        # Check if we should load pretrained.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet50(weights=weights)
        
        # Modified for 1-channel input (grayscale)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Improved projection head (MLP) - for ArcFace we usually just take the Flatten features
        # ResNet50 output before FC is 2048 channels
        self.bn_input = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, embedding_size)
        self.bn_output = nn.BatchNorm1d(embedding_size)
        
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn_input(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_output(x)
        
        if self.num_classes is not None:
            # For classification (softmax) training
            x = self.classifier(x)
            return x
        
        # Note: ArcFace usually takes raw features, but we normalize in ArcFace module or here.
        # Standard practice: Output normalized features for inference.
        # For training with ArcFace, we also normalize.
        # So we can normalize here.
        x = F.normalize(x, p=2, dim=1) 
        return x

class CBAMBottleneckWrapper(nn.Module):
    def __init__(self, bottleneck):
        super(CBAMBottleneckWrapper, self).__init__()
        self.bottleneck = bottleneck
        # Bottleneck expansion is 4. output channels = planes * 4
        self.cbam = CBAM(bottleneck.conv3.out_channels) 
        
    def forward(self, x):
        # Original forward logic of Bottleneck
        identity = x

        out = self.bottleneck.conv1(x)
        out = self.bottleneck.bn1(out)
        out = self.bottleneck.relu(out)

        out = self.bottleneck.conv2(out)
        out = self.bottleneck.bn2(out)
        out = self.bottleneck.relu(out)

        out = self.bottleneck.conv3(out)
        out = self.bottleneck.bn3(out)

        # CBAM injection
        out = self.cbam(out)

        if self.bottleneck.downsample is not None:
            identity = self.bottleneck.downsample(x)

        out += identity
        out = self.bottleneck.relu(out)
        return out

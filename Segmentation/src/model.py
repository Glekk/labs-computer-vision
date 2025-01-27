import torch
from torchvision.models import resnet50
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    '''Dilated Convolutional Block'''
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    '''Atrous Spatial Pyramid Pooling'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = DilatedConvBlock(in_channels, out_channels, 1, 0, 1)
        self.conv3x3_6 = DilatedConvBlock(in_channels, out_channels, 3, 6, 6)
        self.conv3x3_12 = DilatedConvBlock(in_channels, out_channels, 3, 12, 12)
        self.conv3x3_18 = DilatedConvBlock(in_channels, out_channels, 3, 18, 18)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )
        self.conv1x1_out = DilatedConvBlock(out_channels*5, out_channels, 1, 0, 1)

    def forward(self, x):
        x1x1 = self.conv1x1(x)
        x3x3_6 = self.conv3x3_6(x)
        x3x3_12 = self.conv3x3_12(x)
        x3x3_18 = self.conv3x3_18(x)
        avg_pool = self.avg_pool(x)
        avg_pool = nn.functional.interpolate(avg_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1x1, x3x3_6, x3x3_12, x3x3_18, avg_pool], dim=1)
        x = self.conv1x1_out(x)
        return x

class DeepLabV3Plus(nn.Module):
    '''DeepLabV3+ model'''
    def __init__(self, num_classes):
        super().__init__()
        resnet = resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.aspp = ASPP(1024, 256)
        self.conv1x1 = DilatedConvBlock(256, 48, 1, 0, 1)
        self.conv3x3 = DilatedConvBlock(304, 256, 3, 1, 1)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        low_level_features = self.backbone[:-3](x)
        x = self.backbone[:-1](x)
        x = self.aspp(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        low_level_features = self.conv1x1(low_level_features)
        x = torch.cat([x, low_level_features], dim=1)
        x = self.conv3x3(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        return x
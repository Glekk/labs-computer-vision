import torch.nn as nn


class SEBlock(nn.Module):
    '''
    Squeeze and Excitation Block
    '''
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc(out).view(x.size(0), x.size(1), 1, 1)
        return out * x
    

class DepthwiseSeparableBlock(nn.Module):
    '''
    Depthwise Separable Block
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BottleNeckResBlock(nn.Module):
    '''
    Residual Block with Bottleneck
    '''
    def __init__(self, in_channels, mid_channels, stride=1, expansion=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * expansion, kernel_size=1),
        )
        
        if in_channels != mid_channels * expansion:
            self.shortcut = nn.Conv2d(in_channels, mid_channels * expansion, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CNN(nn.Module):
    '''
    Standard CNN
    '''
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 2048), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SECNN(CNN):
    '''
    CNN with Squeeze and Excitation Blocks
    '''
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__(in_channels, num_classes, dropout)
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        self.se5 = SEBlock(512)

    def forward(self, x):
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.block4(x)
        x = self.se4(x)
        x = self.block5(x)
        x = self.se5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class ResBottleneckCNN(nn.Module):
    '''
    CNN with Residual Bottleneck Blocks
    '''
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            BottleNeckResBlock(64, 64),
            BottleNeckResBlock(256, 64),
        )
        self.block3 = nn.Sequential(
            BottleNeckResBlock(256, 128, stride=2),
            BottleNeckResBlock(512, 128),
        )
        self.block4 = nn.Sequential(
            BottleNeckResBlock(512, 256, stride=2),
            BottleNeckResBlock(1024, 256),
        )
        self.block5 = nn.Sequential(
            BottleNeckResBlock(1024, 512, stride=2),
            BottleNeckResBlock(2048, 512),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DepthwiseSeparableCNN(nn.Module):
    '''
    CNN with Depthwise Separable Blocks
    '''
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableBlock(128, 256),
            nn.ReLU(),
            DepthwiseSeparableBlock(256, 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            DepthwiseSeparableBlock(256, 512),
            nn.ReLU(),
            DepthwiseSeparableBlock(512, 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            DepthwiseSeparableBlock(512, 512),
            nn.ReLU(),
            DepthwiseSeparableBlock(512, 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 2048), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
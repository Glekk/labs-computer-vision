import torch.nn as nn


class Generator(nn.Module):
    '''
    Generator class for DCGAN

    Args:
        z_dim(int): Dimension of the noise vector
        channels(int): Number of channels in the image
        features(int): Number of features
    '''
    def __init__(self, z_dim, channels, features):
        super().__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features * 16, 4, 1, 0),
            self._block(features * 16, features * 8, 4, 2, 1),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            nn.ConvTranspose2d(features * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    

class Discriminator(nn.Module):
    '''
    Discriminator class for DCGAN

    Args:
        channels(int): Number of channels in the image
        features(int): Number of features
    '''
    def __init__(self, channels, features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.net(x)
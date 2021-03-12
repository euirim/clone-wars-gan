"""
Adpated from https://github.com/vandit15/Self-Supervised-Gans-Pytorch/blob/master/ops.py
"""
from torch import nn


class ResidualG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualG, self).__init__()
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding=1
        )

    def forward(self, x):
        inpt = x
        x = self.relu(self.batch_norm1(x))
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.conv2(self.relu(x))
        return self.upsample(inpt) + x


class ResidualD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, is_start=False):
        super(ResidualD, self).__init__()
        self.is_start = is_start

        self.conv_short = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        )
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel, stride=stride, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        inpt = x
        if self.is_start:
            conv1 = self.relu(self.conv1(x))
            conv2 = self.relu(self.conv2(conv1))
        else:
            conv1 = self.conv1(self.relu(x))
            conv2 = self.conv2(self.relu(conv1))

        resi = self.conv_short(inpt)
        return resi + conv2

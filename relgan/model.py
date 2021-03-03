"""
Adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/relativistic_gan/relativistic_gan.py
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.init_size = options.img_size // 4
        self.l1 = nn.Linear(options.latent_dim, 128 * self.init_size ** 2)
        self.l1_conv = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2),)
        self.c1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(64, options.num_channels, 3, stride=1, padding=1), nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.l1_conv(out)
        out = self.c1(out)
        out = self.c2(out)
        out = self.c3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(options.num_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = options.img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

"""
Adapted from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
from torch import nn


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(params["nz"], params["ngf"] * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(params["ngf"] * 16),
            nn.ReLU(True),
            # state size. (params["ngf"]*16) x 4 x 4
            nn.ConvTranspose2d(
                params["ngf"] * 16, params["ngf"] * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(params["ngf"] * 8),
            nn.ReLU(True),
            # state size. (params["ngf"]*8) x 8 x 8
            nn.ConvTranspose2d(
                params["ngf"] * 8, params["ngf"] * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(params["ngf"] * 4),
            nn.ReLU(True),
            # state size. (params["ngf"]*4) x 16 x 16
            nn.ConvTranspose2d(
                params["ngf"] * 4, params["ngf"] * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(params["ngf"] * 2),
            nn.ReLU(True),
            # state size. (params["ngf"]*2) x 32 x 32
            nn.ConvTranspose2d(params["ngf"] * 2, params["ngf"], 4, 2, 1, bias=False),
            nn.BatchNorm2d(params["ngf"]),
            nn.ReLU(True),
            # state size. (params["ngf"]) x 64 x 64
            nn.ConvTranspose2d(params["ngf"], params["nc"], 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 128 x 128
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(params["nc"], params["ndf"], 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params["ndf"]) x 64 x 64
            nn.Conv2d(
                params["ndf"], params["ndf"] * 2, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(params["ndf"] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params["ndf"]*2) x 32 x 32
            nn.Conv2d(
                params["ndf"] * 2, params["ndf"] * 4, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(params["ndf"] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params["ndf"]*4) x 16 x 16
            nn.Conv2d(
                params["ndf"] * 4, params["ndf"] * 8, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(params["ndf"] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params["ndf"]*8) x 8 x 8
            nn.Conv2d(
                params["ndf"] * 8,
                params["ndf"] * 16,
                4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(params["ndf"] * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (params["ndf"]*16) x 4 x 4
            nn.Conv2d(params["ndf"] * 16, 1, 4, stride=1, padding=0, bias=False),
            # nn.Sigmoid(),
            # state size. 1
        )

    def forward(self, img):
        return self.main(img)

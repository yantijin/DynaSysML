import torch
import torch.nn as nn

class MLPGen(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(MLPGen, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, inputs):
        return self.l1(inputs)

class ConvGen(nn.Module):
    def __init__(self, z_dim):
        super(ConvGen, self).__init__()
        self.l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=512, kernel_size=4, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.l(z)


class MLPDis(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(MLPDis, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, inputs):
        return self.l1(inputs)


class ConvDis(nn.Module):
    def __init__(self, ):
        super(ConvDis, self).__init__()
        self.l = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x):
        res = self.l(x).squeeze()
        return res
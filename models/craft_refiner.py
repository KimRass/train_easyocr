import torch
import torch.nn as nn

from torch_utilities import (
    initialize_weights
)


class CRAFTRefiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.last_conv = nn.Sequential(
            nn.Conv2d(34, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1)
        )

        initialize_weights(self.last_conv.modules())
        initialize_weights(self.aspp1.modules())
        initialize_weights(self.aspp2.modules())
        initialize_weights(self.aspp3.modules())
        initialize_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        z = torch.cat([y.permute(0, 3, 1, 2), upconv4], dim=1)
        z = self.last_conv(z)

        aspp1 = self.aspp1(z)
        aspp2 = self.aspp2(z)
        aspp3 = self.aspp3(z)
        aspp4 = self.aspp4(z)

        # (batch_size, 1, 128, 128)
        out = aspp1 + aspp2 + aspp3 + aspp4
        # (batch_size, 128, 128, 1)
        return out.permute(0, 2, 3, 1)

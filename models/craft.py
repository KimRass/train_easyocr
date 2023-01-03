import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg16_bn import (
    vgg16_bn
)
from torch_utilities import (
    initialize_weights
)


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.basenet = vgg16_bn(pretrained=False, freeze=False)

        # U-Net
        self.upconv1 = double_conv(in_ch=1024, mid_ch=512, out_ch=256)
        self.upconv2 = double_conv(in_ch=512, mid_ch=256, out_ch=128)
        self.upconv3 = double_conv(in_ch=256, mid_ch=128, out_ch=64)
        self.upconv4 = double_conv(in_ch=128, mid_ch=64, out_ch=32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 2, kernel_size=1),
        )

        initialize_weights(self.upconv1.modules())
        initialize_weights(self.upconv2.modules())
        initialize_weights(self.upconv3.modules())
        initialize_weights(self.upconv4.modules())
        initialize_weights(self.conv_cls.modules())

    def forward(self, x):
        sources = self.basenet(x)

        # U-Net
        z = torch.cat([sources[0], sources[1]], dim=1)
        z = self.upconv1(z)

        z = F.interpolate(input=z, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        z = torch.cat([z, sources[2]], dim=1)
        z = self.upconv2(z)

        z = F.interpolate(input=z, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        z = torch.cat([z, sources[3]], dim=1)
        z = self.upconv3(z)

        z = F.interpolate(input=z, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        z = torch.cat([z, sources[4]], dim=1)
        feature = self.upconv4(z)

        z = self.conv_cls(feature)
        return z.permute(0, 2, 3, 1), feature

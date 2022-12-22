from pathlib import Path
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models

from torch_utilities import (
    copy_state_dict,
    initialize_weights
)


class vgg16_bn(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16_bn(weights=None)

        vgg_pretrained_features.load_state_dict(
            copy_state_dict(
                torch.load(Path(__file__).parent.parent/"pretrained/vgg16_bn.pth")
            )
        )
        vgg_pretrained_features = vgg_pretrained_features.features

        self.slice1 = nn.Sequential()
        for idx in range(12):  # conv2_2
            self.slice1.add_module(name=str(idx), module=vgg_pretrained_features[idx])

        self.slice2 = nn.Sequential()
        for idx in range(12, 19):  # conv3_3
            self.slice2.add_module(name=str(idx), module=vgg_pretrained_features[idx])

        self.slice3 = nn.Sequential()
        for idx in range(19, 29):  # conv4_3
            self.slice3.add_module(name=str(idx), module=vgg_pretrained_features[idx])

        self.slice4 = nn.Sequential()
        for idx in range(29, 39):  # conv5_3
            self.slice4.add_module(name=str(idx), module=vgg_pretrained_features[idx])

        # fc6, fc7 without atrous conv
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        initialize_weights(self.slice5.modules())  # No pretrained model for fc6 and fc7

        # for param in self.slice1.parameters():  # only first conv
        #     param.requires_grad = False

    def forward(self, x):
        z = self.slice1(x)
        z_relu2_2 = z
        z = self.slice2(z)
        z_relu3_2 = z
        z = self.slice3(z)
        z_relu4_3 = z
        z = self.slice4(z)
        z_relu5_3 = z
        z = self.slice5(z)
        z_fc7 = z
        vgg_outputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        return vgg_outputs(z_fc7, z_relu5_3, z_relu4_3, z_relu3_2, z_relu2_2)

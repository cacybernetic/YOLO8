"""SPPF block: Spatial Pyramid Pooling - Fast."""

import torch
import torch.nn as nn

from .conv import Conv


class SPPF(nn.Module):
    """Three chained max pools merged together. Cheap multi-scale pooling."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels,
                        kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels * 4, out_channels,
                        kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

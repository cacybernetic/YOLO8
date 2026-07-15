"""C2f block: the main feature block of YOLOv8 (CSP style)."""

import torch
import torch.nn as nn

from .conv import Conv


class Bottleneck(nn.Module):
    """Two 3x3 convolutions with an optional residual connection."""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels,
                          kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in
        return x


class C2f(nn.Module):
    """Split the features in two, run bottlenecks, then merge everything."""

    def __init__(self, in_channels, out_channels, num_bottlenecks,
                 shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks
        self.conv1 = Conv(in_channels, out_channels,
                          kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([
            Bottleneck(self.mid_channels, self.mid_channels,
                       shortcut=shortcut)
            for _ in range(num_bottlenecks)
        ])
        self.conv2 = Conv((num_bottlenecks + 2) * out_channels // 2,
                          out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Canonical YOLOv8 ordering: [x1, x2, m1(x2), m2(m1(x2)), ...].
        # The bottleneck chain runs on the SECOND chunk and new outputs
        # are appended, matching the official C2f weight layout.
        x = self.conv1(x)
        x1 = x[:, :x.shape[1] // 2, :, :]
        x2 = x[:, x.shape[1] // 2:, :, :]
        outputs = [x1, x2]
        for i in range(self.num_bottlenecks):
            outputs.append(self.m[i](outputs[-1]))
        return self.conv2(torch.cat(outputs, dim=1))

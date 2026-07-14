"""DFL layer: turns a distribution over bins into one distance value."""

import torch
import torch.nn as nn


class DFL(nn.Module):
    """Distribution Focal Loss integration layer.

    The head predicts, for each box side, a probability distribution over
    `ch` bins. This layer computes the expected value of that distribution
    with a fixed 1x1 convolution whose weights are [0, 1, ..., ch - 1].
    """

    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1,
                              kernel_size=1, bias=False).requires_grad_(False)
        with torch.no_grad():
            weights = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
            self.conv.weight.copy_(weights)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)

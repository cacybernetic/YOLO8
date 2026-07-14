"""Simple upsample layer used by the neck."""

import torch.nn as nn


class Upsample(nn.Module):
    """Nearest neighbor upsample by a fixed scale factor."""

    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)

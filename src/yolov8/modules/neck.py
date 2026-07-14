"""YOLOv8 neck: PAN-FPN feature fusion across the three scales."""

import torch
import torch.nn as nn

from .scaling import yolo_params
from .conv import Conv
from .c2f import C2f
from .upsample import Upsample


class Neck(nn.Module):
    """Top-down then bottom-up fusion of the backbone features."""

    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample()
        self.c2f_1 = C2f(int(512 * w * (1 + r)), int(512 * w),
                         num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_2 = C2f(int(768 * w), int(256 * w),
                         num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_3 = C2f(int(768 * w), int(512 * w),
                         num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_4 = C2f(int(512 * w * (1 + r)), int(512 * w * r),
                         num_bottlenecks=int(3 * d), shortcut=False)

        self.cv_1 = Conv(int(256 * w), int(256 * w),
                         kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(int(512 * w), int(512 * w),
                         kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        res_1 = x
        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)
        res_2 = self.c2f_1(x)

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)
        out_1 = self.c2f_2(x)

        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out_2)
        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3

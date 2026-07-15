"""YOLOv8 detection head: anchor-free box and class prediction."""

import math

import torch
import torch.nn as nn

from .scaling import yolo_params
from .conv import Conv
from .dfl import DFL
from .anchors import make_anchors


class Head(nn.Module):
    """Decoupled head with a box branch and a class branch per scale.

    In training mode, forward returns the three raw feature maps.
    In eval mode, it returns a tuple (inference_tensor, raw_outputs) so
    the loss can also be computed on validation without a second forward.
    """

    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes
        self.no = self.coordinates + self.nc
        # Non persistent buffer: follows model.to(device) but stays out of
        # the state_dict (keeps old checkpoints loadable).
        # Values are set by MyYolo after construction.
        self.register_buffer('stride', torch.zeros(3), persistent=False)

        d, w, r = yolo_params(version=version)
        in_channels = [int(256 * w), int(512 * w), int(512 * w * r)]

        # Intermediate widths (Ultralytics Detect convention). The class
        # branch width must NOT depend on the class count alone: with a
        # small nc (1-10 classes) that would squeeze all classification
        # features through a 1-10 channel bottleneck and cap the mAP.
        box_mid = max(16, in_channels[0] // 4, self.coordinates)
        cls_mid = max(in_channels[0], min(self.nc, 100))

        self.box = nn.ModuleList([
            self._branch(c, box_mid, self.coordinates)
            for c in in_channels
        ])
        self.cls = nn.ModuleList([
            self._branch(c, cls_mid, self.nc) for c in in_channels
        ])
        self.dfl = DFL(ch=self.ch)

    @staticmethod
    def _branch(in_channels, mid_channels, out_channels):
        """Build one prediction branch: two Conv blocks + a 1x1 Conv2d."""
        return nn.Sequential(
            Conv(in_channels, mid_channels,
                 kernel_size=3, stride=1, padding=1),
            Conv(mid_channels, mid_channels,
                 kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
        )

    def initialize_biases(self):
        """Set good starting biases for the last box and class convolutions.

        Without this step, the class logits start around sigmoid = 0.5 for
        every anchor and class. The summed BCE then gives a very large loss
        at the start and gradients saturate the clipping for several epochs.
        Convention (same as Ultralytics Detect.bias_init):
          - box branch: bias set to 1.0
          - cls branch: bias set to log(5 / nc / (640 / stride)^2), a small
            prior that matches the expected number of objects per cell.

        Must be called AFTER the strides are set.
        """
        if bool(torch.all(self.stride == 0)):
            raise RuntimeError(
                "initialize_biases() needs calibrated strides "
                "(call _initialize_strides first).")
        with torch.no_grad():
            for box_branch, cls_branch, s in zip(
                    self.box, self.cls, self.stride):
                box_branch[-1].bias.fill_(1.0)
                cls_branch[-1].bias.fill_(
                    math.log(5 / self.nc / (640 / float(s)) ** 2))

    def forward(self, x):
        # x = [out_1, out_2, out_3]. A mutable list is required.
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            x[i] = torch.cat((box, cls), dim=1)

        if self.training:
            return x

        # --- Inference decode ---
        anchors, strides = (
            i.transpose(0, 1) for i in make_anchors(x, self.stride))
        y = torch.cat(
            [i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = y.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        inference_out = torch.cat(
            tensors=(box * strides, cls.sigmoid()), dim=1)
        # Also return the raw tensors for the validation loss.
        return inference_out, x

"""Anchor grid helpers shared by the detection head and the loss."""

import torch


def make_anchors(x, strides, offset=0.5):
    """Build anchor center points and per-anchor stride values.

    Args:
        x: list of feature maps, one per scale, shaped (b, c, h, w).
        strides: iterable of stride values, one per scale.
        offset: sub-cell offset of the anchor centers (0.5 = cell center).

    Returns:
        (anchor_points, stride_tensor):
        anchor_points: (n_anchors, 2) cell-space center coordinates.
        stride_tensor: (n_anchors, 1) stride of each anchor.
    """
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full(
            (h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

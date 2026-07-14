"""Box format helpers shared by the metric modules."""

import numpy as np
import torch


def wh2xy(x):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou_numpy(boxes_a, boxes_b):
    """IoU matrix between two sets of xyxy boxes (numpy).

    Args:
        boxes_a: (N, 4) numpy array
        boxes_b: (M, 4) numpy array

    Returns:
        (N, M) IoU matrix.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]),
                        dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * \
             (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * \
             (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def build_val_targets(images, targets_dict, image_size, device):
    """Rebuild per-image GT tensors (n_gt, 5) = [cls, x1, y1, x2, y2].

    The values are in pixel units of the network input.
    """
    bs = images.size(0)
    idx = targets_dict['idx']
    cls = targets_dict['cls']
    box = targets_dict['box']  # (N, 4) normalized cx cy w h

    per_image = []
    for i in range(bs):
        mask = (idx == i)
        if mask.sum() == 0:
            per_image.append(torch.zeros((0, 5), device=device))
            continue
        c = cls[mask].view(-1, 1)
        b = box[mask]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - w / 2) * image_size
        y1 = (cy - h / 2) * image_size
        x2 = (cx + w / 2) * image_size
        y2 = (cy + h / 2) * image_size
        gt = torch.stack((c.view(-1), x1, y1, x2, y2), dim=1).to(device)
        per_image.append(gt)
    return per_image

"""Deterministic image transforms: letterbox and tensor conversion."""

import cv2
import numpy as np
import torch


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize with letterbox (keeps the aspect ratio), pad with `color`.

    Returns:
        img: padded image (new_shape, new_shape, 3)
        ratio: scale factor
        (pad_w, pad_h): applied padding (left, top)
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def adjust_labels_after_letterbox(labels, ratio, pad, orig_shape,
                                  new_shape):
    """Fix normalized YOLO labels after a letterbox resize.

    Args:
        labels: (N, 5) [cls, cx, cy, w, h] normalized on the source image
        ratio: resize factor
        pad: (pad_left, pad_top)
        orig_shape: source (H, W)
        new_shape: final square size (int)
    """
    if labels.size == 0:
        return labels

    oh, ow = orig_shape
    pad_left, pad_top = pad

    labels = labels.copy()
    labels[:, 1] *= ow
    labels[:, 2] *= oh
    labels[:, 3] *= ow
    labels[:, 4] *= oh

    labels[:, 1] = labels[:, 1] * ratio + pad_left
    labels[:, 2] = labels[:, 2] * ratio + pad_top
    labels[:, 3] = labels[:, 3] * ratio
    labels[:, 4] = labels[:, 4] * ratio

    labels[:, 1:5] /= new_shape
    return labels


def image_to_tensor(img):
    """BGR uint8 HWC image -> RGB float CHW tensor in [0, 1]."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(img))

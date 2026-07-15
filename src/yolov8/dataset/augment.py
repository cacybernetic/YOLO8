"""Data augmentations for object detection (YOLOv8 style)."""

import math
import random

import cv2
import numpy as np

# Default augmentation parameters (YOLOv8 style).
DEFAULT_AUGMENT_PARAMS = {
    'enabled': True,
    # HSV color jitter
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Geometric transforms
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    # Flips
    'flip_lr': 0.5,
    'flip_ud': 0.0,
    # Mosaic (4 images merged). Default YOLOv8 value: 1.0.
    # Turned off near the end of training with `close_mosaic`.
    'mosaic': 1.0,
    # MixUp
    'mixup': 0.0,
    # Cutout
    'cutout': 0.0,
    'cutout_n_max': 4,
    'cutout_size_max': 0.25,
    # Other pixel-level effects
    'blur': 0.0,
    'noise': 0.0,
    'grayscale': 0.0,
}


def merge_augment_params(user_params):
    """Merge user values over the defaults. None values are ignored."""
    params = dict(DEFAULT_AUGMENT_PARAMS)
    if user_params:
        params.update(
            {k: v for k, v in user_params.items() if v is not None})
    return params


def hsv_augment(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """Simple HSV jitter (same behavior as YOLOv5/v8)."""
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_h),
                         cv2.LUT(sat, lut_s),
                         cv2.LUT(val, lut_v)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def horizontal_flip(img, labels):
    img = img[:, ::-1, :].copy()
    if labels.size:
        labels = labels.copy()
        labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels


def vertical_flip(img, labels):
    img = img[::-1, :, :].copy()
    if labels.size:
        labels = labels.copy()
        labels[:, 2] = 1.0 - labels[:, 2]
    return img, labels


def _affine_matrix(w, h, dw, dh, degrees, translate, scale, shear,
                   perspective):
    """Build the composite transform matrix. Returns (M, scale_used)."""
    center = np.eye(3)
    center[0, 2] = -w / 2
    center[1, 2] = -h / 2

    persp = np.eye(3)
    if perspective > 0:
        persp[2, 0] = random.uniform(-perspective, perspective)
        persp[2, 1] = random.uniform(-perspective, perspective)

    rot = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    rot[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=s)

    sh = np.eye(3)
    sh[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    sh[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    trans = np.eye(3)
    trans[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * dw
    trans[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * dh

    return trans @ sh @ rot @ persp @ center, s


def _transform_labels(labels, M, w, h, dw, dh, s, perspective):
    """Apply the matrix M to the boxes and drop degenerate ones."""
    n = labels.shape[0]
    lbl = labels.copy()
    cx = lbl[:, 1] * w
    cy = lbl[:, 2] * h
    bw = lbl[:, 3] * w
    bh = lbl[:, 4] * h
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    corners = np.ones((n * 4, 3), dtype=np.float32)
    corners[:, :2] = np.stack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y1], axis=1),
        np.stack([x2, y2], axis=1),
        np.stack([x1, y2], axis=1),
    ], axis=1).reshape(-1, 2)

    corners = corners @ M.T
    if perspective > 0:
        corners = corners[:, :2] / corners[:, 2:3]
    else:
        corners = corners[:, :2]
    corners = corners.reshape(n, 4, 2)

    new_x1 = corners[:, :, 0].min(axis=1).clip(0, dw)
    new_y1 = corners[:, :, 1].min(axis=1).clip(0, dh)
    new_x2 = corners[:, :, 0].max(axis=1).clip(0, dw)
    new_y2 = corners[:, :, 1].max(axis=1).clip(0, dh)

    # Drop boxes that are too small, too cropped, or too stretched.
    # (aspect ratio < 100 is the YOLOv5/v8 convention; a lower value
    # would remove real thin objects like poles or fences)
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    orig_w = (x2 - x1) * s
    orig_h = (y2 - y1) * s
    ar = np.maximum(new_w / (new_h + 1e-9), new_h / (new_w + 1e-9))
    keep = (new_w > 2) & (new_h > 2) & \
           (new_w * new_h / (orig_w * orig_h + 1e-9) > 0.1) & (ar < 100)

    labels = labels[keep]
    if labels.shape[0] == 0:
        return labels
    new_x1, new_y1 = new_x1[keep], new_y1[keep]
    new_x2, new_y2 = new_x2[keep], new_y2[keep]
    labels[:, 1] = ((new_x1 + new_x2) / 2) / dw
    labels[:, 2] = ((new_y1 + new_y2) / 2) / dh
    labels[:, 3] = (new_x2 - new_x1) / dw
    labels[:, 4] = (new_y2 - new_y1) / dh
    return labels


def random_affine(img, labels, degrees=0.0, translate=0.1, scale=0.5,
                  shear=0.0, perspective=0.0,
                  border_value=(114, 114, 114), dst_size=None):
    """Random affine or perspective transform of the image and labels.

    Args:
        dst_size: output square size. When None, keep the input size.
            The mosaic uses it to crop the 2s x 2s canvas back to s x s
            with the same transform (YOLOv5/v8 convention).
    """
    h, w = img.shape[:2]
    dw, dh = (dst_size, dst_size) if dst_size is not None else (w, h)

    M, s = _affine_matrix(w, h, dw, dh, degrees, translate, scale,
                          shear, perspective)
    if (dw, dh) != (w, h) or (M != np.eye(3)).any():
        if perspective > 0:
            img = cv2.warpPerspective(img, M, dsize=(dw, dh),
                                      borderValue=border_value)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(dw, dh),
                                 borderValue=border_value)

    if labels.shape[0] == 0:
        return img, labels
    labels = _transform_labels(labels, M, w, h, dw, dh, s, perspective)
    return img, labels


def mixup(img1, labels1, img2, labels2, alpha=32.0, beta=32.0):
    """MixUp: img = r * img1 + (1 - r) * img2. Labels are concatenated.

    r follows Beta(alpha, beta); 32/32 gives a balanced mix around 0.5.
    Both images must have the same size.
    """
    r = np.random.beta(alpha, beta)
    img = (img1.astype(np.float32) * r +
           img2.astype(np.float32) * (1 - r)).astype(np.uint8)
    if labels1.size or labels2.size:
        labels = np.concatenate([labels1, labels2], axis=0)
    else:
        labels = np.zeros((0, 5), dtype=np.float32)
    return img, labels


def cutout(img, labels, n_holes_range=(1, 4), size_range=(0.05, 0.25),
           fill=(114, 114, 114)):
    """Cutout: paste random gray patches on the image.

    Labels are NOT changed: a partly hidden box stays valid, which
    pushes the model to use the visible parts.
    """
    h, w = img.shape[:2]
    n = random.randint(*n_holes_range)
    for _ in range(n):
        sz = random.uniform(*size_range)
        ph = int(h * sz)
        pw = int(w * sz)
        if ph < 2 or pw < 2:
            continue
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x1 = max(0, cx - pw // 2)
        y1 = max(0, cy - ph // 2)
        x2 = min(w, x1 + pw)
        y2 = min(h, y1 + ph)
        img[y1:y2, x1:x2] = fill
    return img, labels


def gaussian_blur(img, ksize_range=(3, 7)):
    """Gaussian blur with a random odd kernel size."""
    k = random.randint(*ksize_range)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def gaussian_noise(img, std=0.02):
    """Additive gaussian noise. std is a fraction of 255."""
    noise = np.random.randn(*img.shape).astype(np.float32) * (std * 255)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_grayscale(img):
    """Convert to gray levels (3 channels kept)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class Augmenter:
    """Apply the full augmentation pipeline to one training sample."""

    def __init__(self, params=None, image_size=640):
        self.params = merge_augment_params(params)
        self.image_size = image_size

    def use_mosaic(self):
        """Random draw: should the next sample be built as a mosaic?"""
        p = self.params
        return (p['enabled'] and p['mosaic'] > 0
                and random.random() < p['mosaic'])

    def __call__(self, img, labels, sample_loader, mosaic_used=False):
        """Run the pipeline on an image and its labels.

        Pipeline order (YOLOv8 convention): geometry (which also crops
        a mosaic canvas back to s x s), MixUp, HSV jitter, flips, then
        pixel-level effects. HSV runs after the geometry so it touches
        the final s x s crop instead of the 4x larger mosaic canvas.

        Args:
            img, labels: current sample (letterboxed or mosaic canvas).
            sample_loader: callable returning another random (img,
                labels) pair already passed through the geometry step,
                used by MixUp. May return None.
            mosaic_used: True when `img` is a 2s x 2s mosaic canvas that
                the affine step must crop back to s x s.
        """
        p = self.params
        if not p['enabled']:
            return img, labels

        img, labels = self.geometry(img, labels, mosaic_used)
        img, labels = self._mixup(img, labels, sample_loader)
        img = self._color_jitter(img)
        img, labels = self._flips(img, labels)
        img, labels = self._pixel_effects(img, labels)
        return img, labels

    def _color_jitter(self, img):
        p = self.params
        if any(p[k] > 0 for k in ('hsv_h', 'hsv_s', 'hsv_v')):
            img = hsv_augment(img, h_gain=p['hsv_h'],
                              s_gain=p['hsv_s'], v_gain=p['hsv_v'])
        return img

    def geometry(self, img, labels, mosaic_used=False):
        # Always applied after a mosaic: this step crops the 2s x 2s
        # canvas back to s x s, even with all geometric values at 0.
        # Public: the dataset also uses it to prepare MixUp partners.
        p = self.params
        needed = (p['degrees'] > 0 or p['translate'] > 0 or
                  p['scale'] > 0 or p['shear'] > 0 or
                  p['perspective'] > 0)
        if mosaic_used or needed:
            img, labels = random_affine(
                img, labels,
                degrees=p['degrees'], translate=p['translate'],
                scale=p['scale'], shear=p['shear'],
                perspective=p['perspective'],
                dst_size=self.image_size if mosaic_used else None)
        return img, labels

    def _flips(self, img, labels):
        p = self.params
        if p['flip_lr'] > 0 and random.random() < p['flip_lr']:
            img, labels = horizontal_flip(img, labels)
        if p['flip_ud'] > 0 and random.random() < p['flip_ud']:
            img, labels = vertical_flip(img, labels)
        return img, labels

    def _mixup(self, img, labels, sample_loader):
        # The partner comes from the dataset already passed through the
        # same mosaic + geometry treatment; the flips applied after
        # MixUp cover both images at once (YOLOv8 pipeline order).
        p = self.params
        if p['mixup'] > 0 and random.random() < p['mixup']:
            other = sample_loader()
            if other is not None:
                img2, labels2 = other
                img, labels = mixup(img, labels, img2, labels2)
        return img, labels

    def _pixel_effects(self, img, labels):
        p = self.params
        if p['cutout'] > 0 and random.random() < p['cutout']:
            img, labels = cutout(
                img, labels,
                n_holes_range=(1, max(1, p['cutout_n_max'])),
                size_range=(0.02, p['cutout_size_max']))
        if p['blur'] > 0 and random.random() < p['blur']:
            img = gaussian_blur(img)
        if p['noise'] > 0 and random.random() < p['noise']:
            img = gaussian_noise(img)
        if p['grayscale'] > 0 and random.random() < p['grayscale']:
            img = random_grayscale(img)
        return img, labels

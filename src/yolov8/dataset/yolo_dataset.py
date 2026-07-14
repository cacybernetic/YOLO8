"""Torch Dataset over a YOLO source (folder or zip archive)."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

from .sources import make_source
from .scanner import load_or_scan
from .validation import decode_image_bytes
from .transforms import (letterbox, adjust_labels_after_letterbox,
                         image_to_tensor)
from .augment import Augmenter


def collate_detection_batch(batch):
    """Group a batch and build the target dict used by ComputeLoss."""
    images, labels_list, paths = zip(*batch)
    images = torch.stack(images, dim=0)

    idx_list, cls_list, box_list = [], [], []
    for i, lbl in enumerate(labels_list):
        if lbl.numel() == 0:
            continue
        n = lbl.shape[0]
        idx_list.append(torch.full((n,), i, dtype=torch.float32))
        cls_list.append(lbl[:, 0])
        box_list.append(lbl[:, 1:5])

    if len(idx_list) == 0:
        targets = {
            'idx': torch.zeros(0, dtype=torch.float32),
            'cls': torch.zeros(0, dtype=torch.float32),
            'box': torch.zeros((0, 4), dtype=torch.float32),
        }
    else:
        targets = {
            'idx': torch.cat(idx_list, dim=0),
            'cls': torch.cat(cls_list, dim=0),
            'box': torch.cat(box_list, dim=0),
        }
    return images, targets, list(paths)


def subsample(items, max_count, seed=0):
    """Deterministic random subset of at most `max_count` items."""
    if max_count is None or max_count >= len(items):
        return items
    rng = random.Random(seed)
    picked = rng.sample(range(len(items)), max_count)
    return [items[i] for i in sorted(picked)]


class YoloDataset(Dataset):
    """Object detection dataset with optional augmentations.

    Args:
        path: dataset folder or .zip archive (see sources.py).
        image_size: square input size (default 640).
        augment: enable train-time augmentations.
        augment_params: dict of augmentation values
            (see DEFAULT_AUGMENT_PARAMS). Missing keys use the defaults.
        max_samples: keep at most this many samples (None = all).
        use_cache: read/write the `<name>.cache.json` scan cache.
        validate_images: decode every image during the scan.
        strict: raise on the first invalid sample during the scan.
        seed: seed for the max_samples subset.
    """

    def __init__(self, path, image_size=640, augment=False,
                 augment_params=None, max_samples=None, use_cache=True,
                 validate_images=True, strict=False, seed=0,
                 verbose=True):
        super().__init__()
        self.source = make_source(path)
        self.image_size = image_size
        self.augment = augment
        self.names = self.source.read_names()

        samples = load_or_scan(
            self.source, use_cache=use_cache,
            validate_images=validate_images,
            strict=strict, verbose=verbose)
        if len(samples) == 0:
            raise RuntimeError(
                f"No valid sample after filtering in: {path}")
        self.samples = subsample(samples, max_samples, seed=seed)

        self.augmenter = Augmenter(augment_params, image_size=image_size)

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        """Read one image, apply letterbox, return (img, labels, key).

        Returns None when the image cannot be read anymore (file changed
        or removed after the scan). The caller must handle this case.
        """
        sample = self.samples[idx]
        try:
            data = self.source.read_image_bytes(sample['image'])
        except Exception:
            return None
        img = decode_image_bytes(data)
        if img is None:
            return None

        orig_h, orig_w = img.shape[:2]
        labels = np.array(sample['labels'], dtype=np.float32)
        labels = labels.reshape(-1, 5)
        img, ratio, pad = letterbox(img, new_shape=self.image_size)
        labels = adjust_labels_after_letterbox(
            labels, ratio, pad, (orig_h, orig_w), self.image_size)
        return img, labels, sample['image']

    def _load_sample_retry(self, idx, attempts=10):
        """Load a sample, retry on random indexes when it fails."""
        result = self._load_sample(idx)
        count = 0
        while result is None and count < attempts:
            idx = random.randint(0, len(self) - 1)
            result = self._load_sample(idx)
            count += 1
        return result

    def _mixup_loader(self):
        """Loader callback used by MixUp inside the Augmenter."""
        j = random.randint(0, len(self) - 1)
        result = self._load_sample(j)
        if result is None:
            return None
        img, labels, _ = result
        return img, labels

    def _load_mosaic(self, idx):
        """4-image mosaic (YOLOv5/v8 convention).

        Build a 2s x 2s canvas with a random center in [0.5s, 1.5s]^2,
        paste 4 letterboxed s x s images (the requested one plus 3
        random ones) and fix the labels. The canvas is then cropped
        back to s x s by random_affine inside the Augmenter.
        """
        s = self.image_size
        indices = [idx] + [random.randint(0, len(self) - 1)
                           for _ in range(3)]
        xc = int(random.uniform(0.5 * s, 1.5 * s))
        yc = int(random.uniform(0.5 * s, 1.5 * s))
        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        labels4 = []

        for i, index in enumerate(indices):
            result = self._load_sample_retry(index)
            if result is None:
                continue  # tile stays gray (heavily corrupted dataset)
            img, labels, _ = result
            placed = self._place_mosaic_tile(canvas, img, i, xc, yc, s)
            if labels.size:
                labels4.append(self._shift_labels(labels, img, placed))

        return canvas, self._merge_mosaic_labels(labels4, s)

    @staticmethod
    def _place_mosaic_tile(canvas, img, i, xc, yc, s):
        """Paste tile i on the canvas. Return (padw, padh) offsets."""
        h, w = img.shape[:2]
        if i == 0:    # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a), 0, w,
                                  min(y2a - y1a, h))
        else:         # bottom-right
            x1a, y1a, x2a, y2a = (xc, yc, min(xc + w, 2 * s),
                                  min(2 * s, yc + h))
            x1b, y1b, x2b, y2b = (0, 0, min(w, x2a - x1a),
                                  min(y2a - y1a, h))
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        return x1a - x1b, y1a - y1b

    @staticmethod
    def _shift_labels(labels, img, padding):
        """Move normalized labels to pixel space on the 2s canvas."""
        padw, padh = padding
        h, w = img.shape[:2]
        lb = labels.copy()
        lb[:, 1] = lb[:, 1] * w + padw
        lb[:, 2] = lb[:, 2] * h + padh
        lb[:, 3] *= w
        lb[:, 4] *= h
        return lb

    @staticmethod
    def _merge_mosaic_labels(labels4, s):
        """Clip mosaic labels on the canvas and normalize them on 2s."""
        if not labels4:
            return np.zeros((0, 5), dtype=np.float32)
        labels4 = np.concatenate(labels4, axis=0)
        x1 = np.clip(labels4[:, 1] - labels4[:, 3] / 2, 0, 2 * s)
        y1 = np.clip(labels4[:, 2] - labels4[:, 4] / 2, 0, 2 * s)
        x2 = np.clip(labels4[:, 1] + labels4[:, 3] / 2, 0, 2 * s)
        y2 = np.clip(labels4[:, 2] + labels4[:, 4] / 2, 0, 2 * s)
        keep = ((x2 - x1) > 2) & ((y2 - y1) > 2)
        labels4 = labels4[keep]
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        labels4[:, 1] = ((x1 + x2) / 2) / (2 * s)
        labels4[:, 2] = ((y1 + y2) / 2) / (2 * s)
        labels4[:, 3] = (x2 - x1) / (2 * s)
        labels4[:, 4] = (y2 - y1) / (2 * s)
        return labels4

    def __getitem__(self, idx):
        use_mosaic = self.augment and self.augmenter.use_mosaic()

        if use_mosaic:
            img, labels = self._load_mosaic(idx)
            img_path = self.samples[idx]['image']
        else:
            img, labels, img_path = self._load_plain(idx)

        if self.augment:
            img, labels = self.augmenter(
                img, labels, self._mixup_loader, mosaic_used=use_mosaic)

        img = image_to_tensor(img)
        labels = torch.from_numpy(labels).float()  # (N, 5)
        return img, labels, img_path

    def _load_plain(self, idx):
        """Load one letterboxed sample with the right failure policy."""
        result = self._load_sample(idx)
        if result is not None:
            return result

        if not self.augment:
            # Eval mode: fail loudly. A silent replacement would make
            # the evaluation non deterministic and count GT twice.
            raise RuntimeError(
                f"Unreadable image during evaluation: "
                f"{self.samples[idx]['image']}. Fix or remove it.")

        # Train mode: fall back on another random sample. The scan
        # already validated the images, so this only covers files
        # changed or deleted after startup.
        logger.warning(
            f"Unreadable image skipped: {self.samples[idx]['image']} "
            f"(falling back on a random sample)")
        result = self._load_sample_retry(idx)
        if result is None:
            raise RuntimeError(
                "Cannot load a valid sample after several retries. "
                "Check the dataset files.")
        return result

    # Kept as a static method for backward compatibility.
    collate_fn = staticmethod(collate_detection_batch)

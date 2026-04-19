"""
Dataset loader pour la détection d'objets au format YOLO.

Structure attendue :
    dataset_dir/
      train/
        images/<*.png|*.jpg|*.jpeg>
        labels/<*.txt>   # une ligne par boite: "class cx cy w h" (normalisé [0,1])
      test/
        images/
        labels/

Les GT en retour de __getitem__ sont au format (N, 5) = [class, cx, cy, w, h] normalisé.
La collate_fn produit le dict attendu par ComputeLoss : {'idx', 'cls', 'box'}.
"""

import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize avec letterbox (préserve le ratio), remplit avec `color`.

    Returns:
        img: image padée (new_shape, new_shape, 3)
        ratio: facteur de scaling
        (pad_w, pad_h): padding appliqué (left, top)
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


def adjust_labels_after_letterbox(labels, ratio, pad, orig_shape, new_shape):
    """Ajuste les labels YOLO normalisés après letterbox.

    Args:
        labels: (N, 5) [cls, cx, cy, w, h] normalisés sur l'image originale
        ratio: facteur de resize
        pad: (pad_left, pad_top)
        orig_shape: (H, W) originale
        new_shape: taille finale (int)
    """
    if labels.size == 0:
        return labels

    oh, ow = orig_shape
    pad_left, pad_top = pad

    labels = labels.copy()
    # dénormalisation sur image originale
    labels[:, 1] *= ow  # cx
    labels[:, 2] *= oh  # cy
    labels[:, 3] *= ow  # w
    labels[:, 4] *= oh  # h

    # resize + shift
    labels[:, 1] = labels[:, 1] * ratio + pad_left
    labels[:, 2] = labels[:, 2] * ratio + pad_top
    labels[:, 3] = labels[:, 3] * ratio
    labels[:, 4] = labels[:, 4] * ratio

    # renormalisation sur image finale
    labels[:, 1] /= new_shape
    labels[:, 2] /= new_shape
    labels[:, 3] /= new_shape
    labels[:, 4] /= new_shape

    return labels


def horizontal_flip(img, labels):
    img = img[:, ::-1, :].copy()
    if labels.size:
        labels = labels.copy()
        labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels


def hsv_augment(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """Augmentation HSV simple (conforme à YOLOv5/8)."""
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLODataset(Dataset):
    """Dataset YOLO (détection d'objets)."""

    def __init__(self, root_dir, split='train', image_size=640, augment=False):
        """
        Args:
            root_dir: dossier racine contenant `train/` et `test/`
            split: 'train' ou 'test'
            image_size: taille d'entrée carrée (par défaut 640)
            augment: active les augmentations (flip, HSV). À False pour la val/test.
        """
        super().__init__()
        self.root = Path(root_dir) / split
        self.image_size = image_size
        self.augment = augment

        images_dir = self.root / 'images'
        labels_dir = self.root / 'labels'
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Dossier images introuvable: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Dossier labels introuvable: {labels_dir}")

        self.image_paths = sorted([
            p for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ])

        self.label_paths = []
        for img_path in self.image_paths:
            label_path = labels_dir / (img_path.stem + '.txt')
            self.label_paths.append(label_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def load_labels(self, label_path):
        """Charge un fichier label YOLO -> (N, 5)."""
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        labels = np.array([[float(x) for x in line.split()] for line in lines],
                          dtype=np.float32)
        # On ne garde que les 5 premières colonnes (cls, cx, cy, w, h)
        return labels[:, :5]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Impossible de lire l'image: {img_path}")
        orig_h, orig_w = img.shape[:2]

        labels = self.load_labels(label_path)

        # Augmentations avant letterbox
        if self.augment:
            img = hsv_augment(img)

        # Letterbox
        img, ratio, pad = letterbox(img, new_shape=self.image_size)
        labels = adjust_labels_after_letterbox(
            labels, ratio, pad, (orig_h, orig_w), self.image_size
        )

        # Flip horizontal
        if self.augment and random.random() < 0.5:
            img, labels = horizontal_flip(img, labels)

        # BGR -> RGB, HWC -> CHW, normalisation [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(np.ascontiguousarray(img))

        labels = torch.from_numpy(labels).float()  # (N, 5) [cls, cx, cy, w, h]
        return img, labels, str(img_path)

    @staticmethod
    def collate_fn(batch):
        """Regroupe un batch et construit le dict attendu par ComputeLoss."""
        images, labels_list, paths = zip(*batch)
        images = torch.stack(images, dim=0)

        # Construction idx / cls / box
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

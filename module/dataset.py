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

import math
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


def vertical_flip(img, labels):
    img = img[::-1, :, :].copy()
    if labels.size:
        labels = labels.copy()
        labels[:, 2] = 1.0 - labels[:, 2]
    return img, labels


def random_affine(img, labels, degrees=0.0, translate=0.1, scale=0.5,
                  shear=0.0, perspective=0.0, border_value=(114, 114, 114)):
    """Transformation affine/perspective aléatoire (translation, scale, rotation, shear).

    Les labels sont en format YOLO normalisé [cls, cx, cy, w, h] et sont réajustés.
    Les boites qui sortent du cadre ou deviennent trop petites sont filtrées.
    """
    h, w = img.shape[:2]

    # === Matrice de transformation composite ===
    C = np.eye(3)
    C[0, 2] = -w / 2  # centre sur (0, 0) avant rotation/scale
    C[1, 2] = -h / 2

    P = np.eye(3)
    if perspective > 0:
        P[2, 0] = random.uniform(-perspective, perspective)
        P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    M = T @ S @ R @ P @ C
    # Identité → pas de transformation, skip pour économiser
    if (M != np.eye(3)).any():
        if perspective > 0:
            img = cv2.warpPerspective(img, M, dsize=(w, h),
                                      borderValue=border_value)
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(w, h),
                                 borderValue=border_value)

    # === Ajustement des labels ===
    n = labels.shape[0]
    if n == 0:
        return img, labels

    # YOLO normalisé -> xyxy pixels
    lbl = labels.copy()
    cx = lbl[:, 1] * w
    cy = lbl[:, 2] * h
    bw = lbl[:, 3] * w
    bh = lbl[:, 4] * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Transformer les 4 coins de chaque boite
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

    # Nouvelle bbox englobante des 4 coins transformés
    new_x1 = corners[:, :, 0].min(axis=1)
    new_y1 = corners[:, :, 1].min(axis=1)
    new_x2 = corners[:, :, 0].max(axis=1)
    new_y2 = corners[:, :, 1].max(axis=1)

    # Clip dans l'image
    new_x1 = new_x1.clip(0, w)
    new_y1 = new_y1.clip(0, h)
    new_x2 = new_x2.clip(0, w)
    new_y2 = new_y2.clip(0, h)

    # Filtrage: boites dégénérées ou trop petites, ou ratio d'aspect aberrant
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    orig_w = (x2 - x1) * s
    orig_h = (y2 - y1) * s
    ar = np.maximum(new_w / (new_h + 1e-9), new_h / (new_w + 1e-9))
    keep = (new_w > 2) & (new_h > 2) & \
           (new_w * new_h / (orig_w * orig_h + 1e-9) > 0.1) & \
           (ar < 20)

    labels = labels[keep]
    if labels.shape[0] == 0:
        return img, labels

    new_x1 = new_x1[keep]; new_y1 = new_y1[keep]
    new_x2 = new_x2[keep]; new_y2 = new_y2[keep]

    labels[:, 1] = ((new_x1 + new_x2) / 2) / w  # cx
    labels[:, 2] = ((new_y1 + new_y2) / 2) / h  # cy
    labels[:, 3] = (new_x2 - new_x1) / w        # w
    labels[:, 4] = (new_y2 - new_y1) / h        # h
    return img, labels


def mixup(img1, labels1, img2, labels2, alpha=32.0, beta=32.0):
    """MixUp: img = r·img1 + (1-r)·img2. Les labels sont concaténés.

    r ~ Beta(alpha, beta) (32/32 ~= 0.5, produit un mix équilibré).
    Les deux images doivent avoir la même taille.
    """
    r = np.random.beta(alpha, beta)
    img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
    labels = np.concatenate([labels1, labels2], axis=0) if (labels1.size or labels2.size) \
        else np.zeros((0, 5), dtype=np.float32)
    return img, labels


def cutout(img, labels, n_holes_range=(1, 4), size_range=(0.05, 0.25),
           fill=(114, 114, 114)):
    """Cutout: patches rectangulaires aléatoires.

    Les labels ne sont PAS modifiés (la boite reste valide même partiellement occultée,
    ce qui force le modèle à se baser sur les parties visibles).
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
    """Flou gaussien avec kernel de taille aléatoire (impair)."""
    k = random.randint(*ksize_range)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


def gaussian_noise(img, std=0.02):
    """Bruit gaussien additif. std est en fraction de 255."""
    noise = np.random.randn(*img.shape).astype(np.float32) * (std * 255)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def random_grayscale(img):
    """Convertit en niveaux de gris (3 canaux)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# Paramètres d'augmentation par défaut (style YOLOv8)
DEFAULT_AUGMENT_PARAMS = {
    # HSV (color jitter)
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    # Transformations géométriques
    'degrees': 0.0,           # rotation ± degrés
    'translate': 0.1,         # translation ± fraction
    'scale': 0.5,             # scale ± (0.5 -> entre 0.5x et 1.5x)
    'shear': 0.0,             # shear ± degrés
    'perspective': 0.0,       # perspective ± (0.0 désactivé)
    # Flips
    'flip_lr': 0.5,           # probabilité flip horizontal
    'flip_ud': 0.0,           # probabilité flip vertical (0 par défaut)
    # MixUp
    'mixup': 0.0,             # probabilité d'appliquer MixUp
    # Cutout
    'cutout': 0.0,            # probabilité d'appliquer Cutout
    'cutout_n_max': 4,
    'cutout_size_max': 0.25,
    # Autres
    'blur': 0.0,              # probabilité flou gaussien
    'noise': 0.0,             # probabilité bruit gaussien
    'grayscale': 0.0,         # probabilité conversion gris
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class YOLODataset(Dataset):
    """Dataset YOLO (détection d'objets) avec augmentations configurables."""

    def __init__(self, root_dir, split='train', image_size=640,
                 augment=False, augment_params=None):
        """
        Args:
            root_dir: dossier racine contenant `train/` et `test/`
            split: 'train' ou 'test'
            image_size: taille d'entrée carrée (par défaut 640)
            augment: active les augmentations. À False pour la val/test.
            augment_params: dict d'hyperparamètres d'augmentation (cf. DEFAULT_AUGMENT_PARAMS).
                            Si None, utilise les défauts. Les clés manquantes sont complétées.
        """
        super().__init__()
        self.root = Path(root_dir) / split
        self.image_size = image_size
        self.augment = augment

        # Fusion avec les défauts
        p = dict(DEFAULT_AUGMENT_PARAMS)
        if augment_params:
            p.update({k: v for k, v in augment_params.items() if v is not None})
        self.aug = p

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
        return labels[:, :5]

    def _load_and_preprocess(self, idx):
        """Lit une image + ses labels et applique letterbox (labels YOLO normalisés)."""
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Impossible de lire l'image: {img_path}")
        orig_h, orig_w = img.shape[:2]

        labels = self.load_labels(label_path)
        img, ratio, pad = letterbox(img, new_shape=self.image_size)
        labels = adjust_labels_after_letterbox(
            labels, ratio, pad, (orig_h, orig_w), self.image_size
        )
        return img, labels, str(img_path)

    def __getitem__(self, idx):
        img, labels, img_path = self._load_and_preprocess(idx)

        if self.augment:
            p = self.aug

            # 1) HSV color jitter
            if any(p[k] > 0 for k in ('hsv_h', 'hsv_s', 'hsv_v')):
                img = hsv_augment(img, h_gain=p['hsv_h'],
                                  s_gain=p['hsv_s'], v_gain=p['hsv_v'])

            # 2) Random affine (translation, scale, rotation, shear, perspective)
            if (p['degrees'] > 0 or p['translate'] > 0 or p['scale'] > 0 or
                    p['shear'] > 0 or p['perspective'] > 0):
                img, labels = random_affine(
                    img, labels,
                    degrees=p['degrees'], translate=p['translate'],
                    scale=p['scale'], shear=p['shear'],
                    perspective=p['perspective'],
                )

            # 3) Flips
            if p['flip_lr'] > 0 and random.random() < p['flip_lr']:
                img, labels = horizontal_flip(img, labels)
            if p['flip_ud'] > 0 and random.random() < p['flip_ud']:
                img, labels = vertical_flip(img, labels)

            # 4) MixUp avec un autre échantillon (déjà letterboxé pour matcher la taille)
            if p['mixup'] > 0 and random.random() < p['mixup']:
                j = random.randint(0, len(self) - 1)
                img2, labels2, _ = self._load_and_preprocess(j)
                if p['flip_lr'] > 0 and random.random() < p['flip_lr']:
                    img2, labels2 = horizontal_flip(img2, labels2)
                img, labels = mixup(img, labels, img2, labels2)

            # 5) Cutout (ne modifie pas les labels)
            if p['cutout'] > 0 and random.random() < p['cutout']:
                img, labels = cutout(
                    img, labels,
                    n_holes_range=(1, max(1, p['cutout_n_max'])),
                    size_range=(0.02, p['cutout_size_max']),
                )

            # 6) Flou gaussien
            if p['blur'] > 0 and random.random() < p['blur']:
                img = gaussian_blur(img)

            # 7) Bruit gaussien
            if p['noise'] > 0 and random.random() < p['noise']:
                img = gaussian_noise(img)

            # 8) Grayscale aléatoire
            if p['grayscale'] > 0 and random.random() < p['grayscale']:
                img = random_grayscale(img)

        # BGR -> RGB, HWC -> CHW, normalisation [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = torch.from_numpy(np.ascontiguousarray(img))

        labels = torch.from_numpy(labels).float()  # (N, 5) [cls, cx, cy, w, h]
        return img, labels, img_path

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

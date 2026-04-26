"""
Script d'inférence autonome pour un modèle YOLOv8 exporté en ONNX.

Dépendances minimales (pas de PyTorch requis):
    pip install numpy onnxruntime Pillow

Usage:
    python predict.py --model weights/best.onnx --nc 36 --image samples/image.png --output result.jpg

Ce script reproduit en numpy pur toutes les étapes du pipeline d'inférence:
    1. Chargement de l'image avec PIL
    2. Prétraitement (letterbox + normalisation + transposition CHW)
    3. Inférence ONNX via onnxruntime
    4. Décodage de la sortie + NMS par classe
    5. Reprojection des boites dans l'image originale
    6. Rendu des boites "futuristes" (coins L + cadre + jauge de confiance)
    7. Sauvegarde ou affichage du résultat
"""

import argparse
import colorsys
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration loguru locale (script standalone, pas d'accès au module)
# ---------------------------------------------------------------------------
def _setup_logging(level: str = "INFO"):
    """Configure loguru avec un format compact pour ce script standalone.

    On retire le handler par défaut et on ajoute le nôtre vers stderr,
    avec couleurs si TTY. Cohérent avec module/utils.py:setup_logging().
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format=("<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | <level>{message}</level>"),
        level=level,
        colorize=True,
    )


# ---------------------------------------------------------------------------
# Noms de classes par défaut (COCO — 80 classes)
# ---------------------------------------------------------------------------
# Utilisé par défaut pour étiqueter les prédictions. Si `--nc` correspond à la
# longueur de cette liste (80), les noms ici sont utilisés. Sinon, des noms
# génériques "class_i" sont générés, sauf si --names est fourni.

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new_size: int = 640,
              color=(114, 114, 114)):
    """Redimensionne en conservant le ratio d'aspect et remplit avec `color`.

    Mathématiquement: on cherche le plus grand facteur r tel que
    (h*r, w*r) <= (new_size, new_size), puis on centre l'image redimensionnée
    dans un carré new_size x new_size rempli de `color` (gris 114 par défaut,
    convention YOLOv5/v8).

    Args:
        img: image BGR ou RGB (H, W, 3) uint8
        new_size: taille cible carrée
        color: couleur de remplissage (B, G, R) ou (R, G, B) selon l'ordre utilisé

    Returns:
        padded: image (new_size, new_size, 3) uint8
        ratio: facteur d'échelle appliqué
        (pad_left, pad_top): padding appliqué (pour reprojection inverse)
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))

    # Redimensionnement bilinéaire via PIL (plus rapide et meilleure qualité
    # que numpy nu pour du resize). On convertit en PIL, resize, puis retour en np.
    if (w, h) != (new_unpad_w, new_unpad_h):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((new_unpad_w, new_unpad_h), Image.BILINEAR)
        img_resized = np.asarray(pil_img)
    else:
        img_resized = img

    # Calcul du padding à répartir de chaque côté
    dw = new_size - new_unpad_w
    dh = new_size - new_unpad_h
    pad_left = dw // 2
    pad_right = dw - pad_left
    pad_top = dh // 2
    pad_bottom = dh - pad_top

    # Construction du canvas gris et placement de l'image redimensionnée
    padded = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    padded[pad_top:pad_top + new_unpad_h,
           pad_left:pad_left + new_unpad_w] = img_resized
    return padded, r, (pad_left, pad_top)


def preprocess(img_rgb: np.ndarray, input_size: int = 640):
    """Pipeline complet de prétraitement : letterbox + normalisation + CHW batch.

    Args:
        img_rgb: image RGB (H, W, 3) uint8 telle que chargée par PIL
        input_size: taille carrée d'entrée du modèle

    Returns:
        tensor: batch (1, 3, input_size, input_size) float32 dans [0, 1]
        ratio: facteur letterbox
        pad: (pad_left, pad_top)
        orig_shape: (orig_h, orig_w) pour reprojection
    """
    orig_h, orig_w = img_rgb.shape[:2]
    padded, ratio, pad = letterbox(img_rgb, new_size=input_size)

    # Normalisation [0, 1] + HWC -> CHW + ajout dimension batch
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
    tensor = np.ascontiguousarray(tensor)
    return tensor, ratio, pad, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Décodage ONNX + NMS (tout en numpy)
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) -> (x1, y1, x2, y2). Vectorisé, shape (N, 4)."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calcule la matrice IoU entre N boites A et M boites B (format xyxy).

    Returns:
        iou: (N, M)
    """
    # Aire de chaque boite
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Intersection: max(x1) à min(x2), max(y1) à min(y2)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray,
              iou_threshold: float = 0.45) -> np.ndarray:
    """Non-Maximum Suppression gloutonne en numpy pur.

    Algorithme (classique):
      1. Trier les boites par score décroissant
      2. Prendre la boite de plus haut score, l'ajouter au résultat
      3. Supprimer toutes les boites dont l'IoU avec elle dépasse le seuil
      4. Répéter jusqu'à épuisement

    Args:
        boxes_xyxy: (N, 4)
        scores: (N,)
        iou_threshold: seuil IoU au-delà duquel deux boites sont considérées
                       comme redondantes

    Returns:
        indices des boites conservées, dans l'ordre d'ajout
    """
    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=np.int64)

    # Pré-calcul des aires et tri décroissant par score
    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        # IoU entre la boite i et toutes les boites restantes
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

        # On ne garde que les boites avec IoU <= seuil
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def postprocess(output: np.ndarray, num_classes: int,
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                max_det: int = 300) -> np.ndarray:
    """Décode la sortie brute du modèle ONNX en détections filtrées.

    Format d'entrée (sortie YOLOv8 ONNX):
        output: (1, 4 + num_classes, num_anchors)
            [:, 0:4, :]  -> cx, cy, w, h (en coords image letterbox)
            [:, 4:, :]   -> scores sigmoïdés par classe [0, 1]

    Pipeline:
        1. Transposer pour avoir (num_anchors, 4 + num_classes)
        2. Pour chaque anchor: prendre la classe de plus haut score
        3. Filtrer par seuil de confiance
        4. Convertir cxcywh -> xyxy
        5. NMS par classe (offset par classe pour isoler les classes)
        6. Limiter à max_det détections

    Returns:
        (N, 6) array: [x1, y1, x2, y2, conf, cls_id]
    """
    # (1, 4+nc, na) -> (na, 4+nc) car batch=1
    preds = output[0].transpose(1, 0)

    boxes_cxcywh = preds[:, :4]
    cls_scores = preds[:, 4:4 + num_classes]

    # Pour chaque anchor, on prend la classe de plus haut score (approche
    # "best class only" standard). Alternative: garder toutes les classes
    # au-dessus du seuil, mais c'est plus coûteux et YOLOv8 utilise cette
    # version simple en inférence.
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

    # Filtrage par seuil de confiance — beaucoup d'anchors sont écartés ici
    # (typiquement on passe de 8400 à quelques dizaines).
    mask = confs > conf_threshold
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes_cxcywh = boxes_cxcywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    # Conversion centre-taille -> coins
    boxes_xyxy = xywh_to_xyxy(boxes_cxcywh)

    # NMS par classe via l'astuce du "class offset":
    # on décale les boites de chaque classe d'une grande constante pour que
    # deux boites de classes différentes aient toujours IoU=0. Cela permet
    # d'utiliser un seul appel NMS global au lieu d'un NMS par classe.
    max_wh = 7680  # suffisamment grand pour que les classes ne se chevauchent jamais
    offsets = cls_ids[:, None].astype(np.float32) * max_wh
    boxes_offset = boxes_xyxy + offsets

    keep = nms_numpy(boxes_offset, confs, iou_threshold=iou_threshold)
    keep = keep[:max_det]

    detections = np.concatenate([
        boxes_xyxy[keep],
        confs[keep, None],
        cls_ids[keep, None].astype(np.float32),
    ], axis=1)
    return detections


def scale_boxes_to_original(boxes_xyxy: np.ndarray, ratio: float,
                            pad: tuple, orig_shape: tuple) -> np.ndarray:
    """Inverse le letterbox pour ramener les boites dans l'image originale.

    Étapes:
      1. Retirer le padding (pad_left, pad_top)
      2. Diviser par le ratio de resize
      3. Clamper aux bornes de l'image originale
    """
    pad_left, pad_top = pad
    oh, ow = orig_shape
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= pad_left
    out[:, [1, 3]] -= pad_top
    out[:, :4] /= ratio
    out[:, 0] = np.clip(out[:, 0], 0, ow - 1)
    out[:, 1] = np.clip(out[:, 1], 0, oh - 1)
    out[:, 2] = np.clip(out[:, 2], 0, ow - 1)
    out[:, 3] = np.clip(out[:, 3], 0, oh - 1)
    return out


# ---------------------------------------------------------------------------
# Rendu des boites (style futuriste)
# ---------------------------------------------------------------------------

def build_color_palette(n: int):
    """Palette HSV équirépartie, couleurs stables entre exécutions."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))  # RGB pour PIL
    return colors


def _get_font(size: int):
    """Tente de charger une police TTF système, sinon fallback par défaut."""
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arial.ttf",         # Windows
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_futuristic_box(draw: ImageDraw.ImageDraw, overlay_draw: ImageDraw.ImageDraw,
                        x1, y1, x2, y2, color_rgb, conf, label,
                        font, thickness=2, corner_ratio=0.22, gauge_width=6):
    """Dessine une boite futuriste. Le cadre, coins, tabs et jauge sont dessinés
    sur `overlay_draw` (pour le blend semi-transparent), le label sur `draw`
    (pour rester net).

    Éléments:
      - Cadre rectangulaire fin (1px)
      - 4 coins en L (épais)
      - Petits tabs carrés aux 4 sommets
      - Jauge verticale à droite remplie proportionnellement à la confiance
      - Label en haut à gauche avec ombre portée
    """
    # Arrondis entiers pour PIL
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    # === 1) Cadre fin 1px ===
    overlay_draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=1)

    # === 2) Coins en L épais ===
    cl = max(8, int(min(w, h) * corner_ratio))
    # Top-left
    overlay_draw.line([(x1, y1), (x1 + cl, y1)], fill=color_rgb, width=thickness)
    overlay_draw.line([(x1, y1), (x1, y1 + cl)], fill=color_rgb, width=thickness)
    # Top-right
    overlay_draw.line([(x2, y1), (x2 - cl, y1)], fill=color_rgb, width=thickness)
    overlay_draw.line([(x2, y1), (x2, y1 + cl)], fill=color_rgb, width=thickness)
    # Bottom-left
    overlay_draw.line([(x1, y2), (x1 + cl, y2)], fill=color_rgb, width=thickness)
    overlay_draw.line([(x1, y2), (x1, y2 - cl)], fill=color_rgb, width=thickness)
    # Bottom-right
    overlay_draw.line([(x2, y2), (x2 - cl, y2)], fill=color_rgb, width=thickness)
    overlay_draw.line([(x2, y2), (x2, y2 - cl)], fill=color_rgb, width=thickness)

    # === 3) Tabs carrés aux sommets ===
    tab = max(2, thickness)
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        overlay_draw.rectangle(
            [cx - tab, cy - tab, cx + tab, cy + tab],
            fill=color_rgb
        )

    # === 4) Jauge verticale de confiance ===
    offset = max(4, thickness + 1)
    gx1 = x2 + offset
    gx2 = gx1 + gauge_width
    # Si la jauge dépasse, on la bascule à l'intérieur de la boite
    img_w, img_h = overlay_draw.im.size
    if gx2 >= img_w:
        gx2 = max(0, x2 - offset)
        gx1 = max(0, gx2 - gauge_width)
    gy1 = max(0, y1)
    gy2 = min(img_h - 1, y2)
    if gx2 > gx1 and gy2 > gy1:
        # Partie vide de la jauge : teinte légère (via alpha dans la couleur)
        overlay_draw.rectangle([gx1, gy1, gx2, gy2],
                               fill=color_rgb + (80,), outline=color_rgb)
        # Partie remplie depuis le bas
        fill_h = int((gy2 - gy1) * max(0.0, min(1.0, conf)))
        if fill_h > 0:
            overlay_draw.rectangle(
                [gx1, gy2 - fill_h, gx2, gy2], fill=color_rgb
            )

    # === 5) Label avec ombre (dessiné sur la couche opaque pour rester lisible) ===
    label_text = f"{label}: {conf:.2f}"
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    label_x = x1
    label_y = y1 - th - 4
    if label_y < 0:
        label_y = y1 + 4
    # Ombre noire décalée 1px pour lisibilité sur fond clair
    draw.text((label_x + 1, label_y + 1), label_text, fill=(0, 0, 0), font=font)
    draw.text((label_x, label_y), label_text, fill=color_rgb, font=font)


def render_detections(img_pil: Image.Image, detections: np.ndarray,
                      class_names: list, colors: list,
                      thickness=2, font_size=16, opacity=0.75) -> Image.Image:
    """Applique le rendu des détections sur l'image PIL.

    Technique d'opacité: on dessine sur une couche RGBA transparente,
    qu'on blend ensuite sur l'image d'origine. Le label est dessiné
    séparément en pleine opacité pour rester net.
    """
    base = img_pil.convert("RGBA")
    # Couche overlay transparente pour les éléments graphiques
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")

    # Couche label (opaque, sera collée après blend)
    label_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_layer, "RGBA")

    font = _get_font(font_size)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

        draw_futuristic_box(
            label_draw, overlay_draw,
            x1, y1, x2, y2,
            color_rgb=color, conf=float(conf), label=name,
            font=font, thickness=thickness,
        )

    # Application de l'opacité globale sur l'overlay graphique
    # (on multiplie le canal alpha par `opacity`)
    overlay_arr = np.asarray(overlay).copy()
    overlay_arr[..., 3] = (overlay_arr[..., 3].astype(np.float32) * opacity).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")

    # Composition: base + overlay (graphisme semi-transparent) + labels (opaque)
    result = Image.alpha_composite(base, overlay)
    result = Image.alpha_composite(result, label_layer)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(args):
    # --- Validation des chemins ---
    model_path = Path(args.model)
    image_path = Path(args.image)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle ONNX introuvable: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    # --- Chargement du modèle ONNX ---
    # CUDAExecutionProvider est tenté automatiquement si onnxruntime-gpu est installé,
    # sinon on tombe sur le CPU. `get_available_providers` nous le dit.
    available = ort.get_available_providers()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']

    logger.info(f"ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(str(model_path), providers=providers)

    # Inspection des I/O pour déduire la taille d'entrée
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_name = input_info.name

    # Le modèle peut avoir des dimensions dynamiques (ex: 'batch'). Dans ce cas
    # on utilise la taille passée par l'utilisateur ou le défaut.
    input_shape = input_info.shape
    if len(input_shape) == 4 and isinstance(input_shape[-1], int):
        model_input_size = input_shape[-1]
    else:
        model_input_size = args.input_size

    logger.info(f"Modèle: {model_path.name}")
    logger.info(f"  input:  {input_info.name} shape={input_info.shape}")
    logger.info(f"  output: {output_info.name} shape={output_info.shape}")
    logger.info(f"  taille d'entrée effective: {model_input_size}")

    # --- Chargement de l'image avec PIL ---
    img_pil = Image.open(image_path).convert("RGB")
    img_rgb = np.asarray(img_pil)
    orig_h, orig_w = img_rgb.shape[:2]
    logger.info(f"Image: {image_path.name}  ({orig_w}x{orig_h})")

    # --- Prétraitement ---
    t0 = time.perf_counter()
    tensor, ratio, pad, orig_shape = preprocess(img_rgb, input_size=model_input_size)
    t_pre = time.perf_counter() - t0

    # --- Inférence ---
    t0 = time.perf_counter()
    output = session.run(None, {input_name: tensor})[0]
    t_inf = time.perf_counter() - t0

    # --- Postprocessing (décodage + NMS) ---
    t0 = time.perf_counter()
    detections = postprocess(
        output,
        num_classes=args.nc,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    # Reprojection dans l'espace image originale
    if len(detections) > 0:
        detections[:, :4] = scale_boxes_to_original(
            detections[:, :4], ratio, pad, orig_shape
        )
    t_post = time.perf_counter() - t0

    logger.info(f"Timings: preprocess={t_pre*1000:.1f}ms  "
                f"inference={t_inf*1000:.1f}ms  "
                f"postprocess={t_post*1000:.1f}ms")
    logger.info(f"{len(detections)} détection(s) au-dessus du seuil conf={args.conf}")

    # --- Préparation des noms de classes et couleurs ---
    # Priorité: --names (explicite) > CLASS_NAMES (si --nc correspond) > générique
    if args.names:
        # Fichier .txt avec un nom par ligne (surcharge CLASS_NAMES)
        with open(args.names, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
        if len(class_names) != args.nc:
            logger.warning(f"{args.names} contient {len(class_names)} noms "
                           f"mais --nc={args.nc}. Complétion par noms génériques.")
            class_names = (class_names + [f"class_{i}" for i in range(args.nc)])[:args.nc]
    elif args.nc == len(CLASS_NAMES):
        # Le nombre de classes correspond à la constante globale (typiquement COCO à 80).
        # On utilise les noms natifs, bien plus lisibles que "class_0, class_1, ..."
        class_names = list(CLASS_NAMES)
        logger.info(f"Utilisation de CLASS_NAMES (COCO, {len(CLASS_NAMES)} classes)")
    else:
        # Nombre de classes différent de CLASS_NAMES et pas de fichier --names fourni.
        class_names = [f"class_{i}" for i in range(args.nc)]
        logger.info(f"--nc={args.nc} ne correspond pas à CLASS_NAMES "
                    f"({len(CLASS_NAMES)}). Noms génériques utilisés. "
                    f"Fournissez --names classes.txt pour personnaliser.")
    colors = build_color_palette(args.nc)

    # --- Log console des détections ---
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        logger.info(f"  [{i}] {name:<20} conf={conf:.3f}  "
                    f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # --- Rendu ---
    result = render_detections(
        img_pil, detections, class_names, colors,
        thickness=args.thickness,
        font_size=args.font_size,
        opacity=args.opacity,
    )

    # --- Sauvegarde ou affichage ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
        logger.success(f"Image annotée sauvegardée: {out_path}")

    if args.show:
        try:
            result.show(title=image_path.name)
        except Exception as e:
            logger.warning(f"Impossible d'afficher l'image ({e})")

    return detections, result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inférence YOLOv8 autonome depuis un modèle ONNX "
                    "(dépendances: numpy, onnxruntime, Pillow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Arguments obligatoires
    parser.add_argument('--model', type=str, required=True,
                        help="Chemin du fichier .onnx")
    parser.add_argument('--image', type=str, required=True,
                        help="Chemin de l'image à analyser")
    parser.add_argument('--nc', type=int, required=True,
                        help="Nombre de classes du modèle")

    # Arguments optionnels de sortie
    parser.add_argument('--output', type=str, default=None,
                        help="Chemin de sauvegarde de l'image annotée (optionnel)")
    parser.add_argument('--show', action='store_true',
                        help="Affiche l'image annotée après inférence")

    # Seuils
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Seuil de confiance minimum")
    parser.add_argument('--iou', type=float, default=0.45,
                        help="Seuil IoU pour NMS")

    # Rendu
    parser.add_argument('--thickness', type=int, default=2,
                        help="Épaisseur des lignes")
    parser.add_argument('--font-size', type=int, default=16,
                        help="Taille du texte des labels")
    parser.add_argument('--opacity', type=float, default=0.75,
                        help="Opacité du graphisme des boites [0, 1]")

    # Noms de classes
    parser.add_argument('--names', type=str, default=None,
                        help="Fichier .txt avec un nom de classe par ligne")

    # Taille d'entrée (utile si le modèle ONNX a des dimensions dynamiques)
    parser.add_argument('--input-size', type=int, default=640,
                        help="Taille d'entrée du modèle (utilisée si ONNX dynamique)")

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Niveau de log loguru")

    args = parser.parse_args()
    _setup_logging(level=args.log_level)
    run(args)


if __name__ == '__main__':
    main()

"""
Script d'inférence YOLOv8 sur une image.

Usage :
    python -m yolov8.infer --config configs/infer.yaml --image path/vers/image.jpg
    python -m yolov8.infer --config configs/infer.yaml --image img.jpg --save out.jpg
    python -m yolov8.infer --config configs/infer.yaml --image img.jpg --no-show

Le script :
  1. Charge le checkpoint spécifié dans la config
  2. Charge l'image fournie en ligne de commande
  3. Applique letterbox + normalisation
  4. Effectue un forward + NMS
  5. Reprojette les boites dans l'espace image d'origine
  6. Dessine les boites et affiche l'image (cv2.imshow)
"""

import argparse
import colorsys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from yolov8.config import load_infer_config, InferConfig
from yolov8.dataset import letterbox
from yolov8.metrics import non_max_suppression
from yolov8.model import MyYolo
from yolov8.utils import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_color_palette(n):
    """Génère `n` couleurs BGR visuellement distinctes (via HSV equi-espacé)."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def scale_boxes_to_original(boxes_xyxy, ratio, pad, orig_shape):
    """Reprojette des boites (x1, y1, x2, y2) de l'espace letterbox vers l'image d'origine.

    Args:
        boxes_xyxy: (N, 4) en coords pixel dans l'image letterbox
        ratio: facteur de resize appliqué
        pad: (pad_left, pad_top)
        orig_shape: (H, W) image originale

    Returns:
        (N, 4) boites clamped dans les limites de l'image originale.
    """
    pad_left, pad_top = pad
    oh, ow = orig_shape

    boxes = boxes_xyxy.clone()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= ratio

    # Clamp dans l'image d'origine
    boxes[:, 0].clamp_(0, ow - 1)
    boxes[:, 1].clamp_(0, oh - 1)
    boxes[:, 2].clamp_(0, ow - 1)
    boxes[:, 3].clamp_(0, oh - 1)
    return boxes


def _draw_futuristic_box(img, x1, y1, x2, y2, color, conf, label,
                         thickness=2, font_scale=0.5,
                         corner_ratio=0.22, gauge_width=None,
                         opacity=0.75):
    """Dessine une boite futuriste sur `img` (modifiée en place).

    Éléments dessinés:
      - cadre rectangulaire fin (1px) faisant le tour complet de la boite
      - 4 coins en L plus épais par-dessus
      - petits tabs carrés aux 4 sommets (effet "target lock")
      - jauge verticale à droite de la boite, remplie proportionnellement à `conf`
      - label au-dessus du coin supérieur gauche, avec ombre portée

    Toute la boite (cadre + coins + tabs + jauge) est fondue avec l'image d'origine
    via `opacity`. Le texte du label reste à pleine opacité pour rester lisible.
    """
    h_img, w_img = img.shape[:2]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return img

    conf = max(0.0, min(1.0, float(conf)))
    lt = cv2.LINE_AA

    # Tout le "graphisme" de la boite est dessiné sur `overlay`, puis fondu dans
    # `img` avec `opacity`. Le texte, lui, est dessiné ensuite directement sur
    # `img` pour rester parfaitement net.
    overlay = img.copy()

    # === 1) Cadre rectangulaire fin (1px) ===
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=1, lineType=lt)

    # === 2) Coins en L (plus épais, par-dessus le cadre) ===
    cl = int(max(8, min(w, h) * corner_ratio))
    # Top-left
    cv2.line(overlay, (x1, y1), (x1 + cl, y1), color, thickness, lt)
    cv2.line(overlay, (x1, y1), (x1, y1 + cl), color, thickness, lt)
    # Top-right
    cv2.line(overlay, (x2, y1), (x2 - cl, y1), color, thickness, lt)
    cv2.line(overlay, (x2, y1), (x2, y1 + cl), color, thickness, lt)
    # Bottom-left
    cv2.line(overlay, (x1, y2), (x1 + cl, y2), color, thickness, lt)
    cv2.line(overlay, (x1, y2), (x1, y2 - cl), color, thickness, lt)
    # Bottom-right
    cv2.line(overlay, (x2, y2), (x2 - cl, y2), color, thickness, lt)
    cv2.line(overlay, (x2, y2), (x2, y2 - cl), color, thickness, lt)

    # === 3) Petits tabs carrés aux 4 sommets ===
    tab = max(2, thickness)
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.rectangle(overlay, (cx - tab, cy - tab), (cx + tab, cy + tab),
                      color, thickness=cv2.FILLED)

    # === 4) Jauge verticale de confiance ===
    gw = gauge_width if gauge_width is not None else max(4, thickness * 2)
    offset = max(4, thickness + 1)
    gx1 = x2 + offset
    gx2 = gx1 + gw
    # Si la jauge dépasse à droite, on la place à l'intérieur de la boite
    if gx2 >= w_img:
        gx2 = max(0, x2 - offset)
        gx1 = max(0, gx2 - gw)
    gy1 = max(0, y1)
    gy2 = min(h_img - 1, y2)
    if gx2 > gx1 and gy2 > gy1:
        # Fond de la jauge: teinte colorée légère (blend local sur overlay)
        roi_ov = overlay[gy1:gy2 + 1, gx1:gx2 + 1]
        tint = np.full_like(roi_ov, color, dtype=np.uint8)
        alpha_bg = 0.35
        overlay[gy1:gy2 + 1, gx1:gx2 + 1] = cv2.addWeighted(
            tint, alpha_bg, roi_ov, 1 - alpha_bg, 0
        )
        # Partie remplie depuis le bas, proportionnelle à la confiance
        fill_h = int((gy2 - gy1) * conf)
        fill_top = gy2 - fill_h
        if fill_h > 0:
            cv2.rectangle(overlay, (gx1, fill_top), (gx2, gy2),
                          color, thickness=cv2.FILLED)
        # Bordure fine de la jauge
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, 1, lt)

    # === 5) Fondu global: toute la boite devient semi-transparente ===
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, dst=img)

    # === 6) Label + confiance (dessiné APRÈS le blend, à pleine opacité) ===
    label_text = f"{label}: {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

    label_x = x1
    label_y = y1 - 6
    if label_y - th < 0:
        label_y = y1 + th + 4
    if label_x + tw >= w_img:
        label_x = max(0, w_img - tw - 2)

    # Ombre noire décalée d'1px pour la lisibilité sur fond clair
    cv2.putText(img, label_text, (label_x + 1, label_y + 1),
                font, font_scale, (0, 0, 0), thickness=2, lineType=lt)
    cv2.putText(img, label_text, (label_x, label_y),
                font, font_scale, color, thickness=1, lineType=lt)
    return img


def draw_detections(img, detections, class_names, colors,
                    line_thickness=2, font_scale=0.5, opacity=0.75):
    """Dessine les boites futuristes + labels sur `img` (modifie en place et retourne).

    Args:
        img: image BGR (H, W, 3) uint8
        detections: tenseur (N, 6) [x1, y1, x2, y2, conf, cls] en coords image originale
        class_names: liste de noms de classes
        colors: liste de couleurs BGR, une par classe
        opacity: opacité globale des éléments graphiques de la boite [0, 1]
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = colors[cls_id % len(colors)]
        cls_name = str(class_names[cls_id]) if cls_id < len(class_names) else f"class_{cls_id}"

        _draw_futuristic_box(
            img, x1, y1, x2, y2, color, float(conf), cls_name,
            thickness=line_thickness, font_scale=font_scale,
            opacity=opacity,
        )
    return img


# ---------------------------------------------------------------------------
# Pipeline d'inférence
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(cfg: InferConfig, image_path: str,
                  save_path=None, show=None):
    # Résolution des arguments ligne de commande > config
    if save_path is None:
        save_path = cfg.save_path
    if show is None:
        show = cfg.show

    # --- Device ---
    device_str = cfg.device
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA indisponible, bascule sur CPU.")
        device_str = 'cpu'
    device = torch.device(device_str)
    logger.info(f"device={device}")

    # --- Modèle ---
    model = MyYolo(version=cfg.version, num_classes=cfg.num_classes,
                   input_size=cfg.image_size).to(device)
    model.head.stride = model.head.stride.to(device)

    weights_path = Path(cfg.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights_path}")
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Poids chargés: {weights_path}")

    # --- Image ---
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise RuntimeError(f"Impossible de décoder l'image: {image_path}")
    orig_h, orig_w = orig.shape[:2]
    logger.info(f"Image: {image_path.name} ({orig_w}x{orig_h})")

    # --- Préprocessing ---
    img, ratio, pad = letterbox(orig, new_shape=cfg.image_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0).to(device)

    # --- Forward + NMS ---
    out = model(tensor)
    inference_out = out[0] if isinstance(out, tuple) else out
    preds = non_max_suppression(
        inference_out,
        confidence_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
    )
    detections = preds[0]  # (N, 6) dans l'espace letterbox

    # --- Reprojection dans l'image d'origine ---
    if detections.shape[0] > 0:
        detections[:, :4] = scale_boxes_to_original(
            detections[:, :4], ratio, pad, (orig_h, orig_w)
        )
    logger.info(f"{detections.shape[0]} détection(s) au-dessus du seuil "
                f"conf={cfg.conf_threshold}")

    # --- Noms de classes + palette ---
    if cfg.class_names and len(cfg.class_names) == cfg.num_classes:
        # Cast explicite en str: un YAML peut contenir des ints (ex: ids) ou autres scalaires
        class_names = [str(n) for n in cfg.class_names]
    else:
        if cfg.class_names is not None:
            logger.warning(f"class_names a {len(cfg.class_names)} entrées mais "
                           f"num_classes={cfg.num_classes}. Noms génériques utilisés.")
        class_names = [f"class_{i}" for i in range(cfg.num_classes)]
    colors = build_color_palette(cfg.num_classes)

    # --- Rendu ---
    annotated = orig.copy()
    draw_detections(
        annotated, detections.cpu(), class_names, colors,
        line_thickness=cfg.line_thickness,
        font_scale=cfg.font_scale,
        opacity=cfg.box_opacity,
    )

    # Log console des détections
    for i, det in enumerate(detections.cpu()):
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        name = str(class_names[cls_id]) if cls_id < len(class_names) else f"class_{cls_id}"
        logger.info(f"  [{i}] {name:<20} conf={conf:.3f} "
                    f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # --- Sauvegarde ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        logger.success(f"Image annotée écrite: {save_path}")

    # --- Affichage ---
    if show:
        window = f"YOLOv8 - {image_path.name}"
        try:
            cv2.imshow(window, annotated)
            logger.info("Appuyez sur une touche dans la fenêtre pour fermer.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            logger.warning(f"Affichage impossible ({e}). "
                           f"Utilisez --save out.jpg pour récupérer le résultat.")

    return detections, annotated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inférence YOLOv8 sur une image.",
    )
    parser.add_argument('--config', type=str, required=True,
                        help="Chemin vers infer.yaml")
    parser.add_argument('--image', type=str, required=True,
                        help="Chemin vers l'image sur laquelle effectuer l'inférence")
    parser.add_argument('--save', type=str, default=None,
                        help="(optionnel) chemin de sauvegarde de l'image annotée "
                             "(override la valeur du YAML)")
    parser.add_argument('--no-show', action='store_true',
                        help="Désactive l'affichage (utile sur serveur headless)")
    parser.add_argument('--conf', type=float, default=None,
                        help="(optionnel) override du seuil de confiance")
    parser.add_argument('--iou', type=float, default=None,
                        help="(optionnel) override du seuil IoU pour NMS")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Niveau de log loguru')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    cfg = load_infer_config(args.config)

    # Overrides CLI
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.iou is not None:
        cfg.iou_threshold = args.iou

    show = None if not args.no_show else False
    run_inference(cfg, args.image, save_path=args.save, show=show)


if __name__ == '__main__':
    main()

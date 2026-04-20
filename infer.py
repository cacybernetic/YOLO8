"""
Script d'inférence YOLOv8 sur une image.

Usage :
    python -m module.infer --config module/infer.yaml --image path/vers/image.jpg
    python -m module.infer --config module/infer.yaml --image img.jpg --save out.jpg
    python -m module.infer --config module/infer.yaml --image img.jpg --no-show

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

from module.config import load_infer_config, InferConfig
from module.dataset import letterbox
from module.metrics import non_max_suppression
from module.model import MyYolo


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


def draw_detections(img, detections, class_names, colors,
                    line_thickness=2, font_scale=0.5):
    """Dessine les boites + labels sur `img` (modifie en place et retourne).

    Args:
        img: image BGR (H, W, 3) uint8
        detections: tenseur (N, 6) [x1, y1, x2, y2, conf, cls] en coords image originale
        class_names: liste de noms de classes
        colors: liste de couleurs BGR, une par classe
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = colors[cls_id % len(colors)]
        cls_name = str(class_names[cls_id]) if cls_id < len(class_names) else f"class_{cls_id}"
        label = f"{cls_name} {conf:.2f}"

        # Boite
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness)

        # Fond du label
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness=1)
        ytop = max(y1 - th - baseline - 2, 0)
        cv2.rectangle(img, (x1, ytop), (x1 + tw + 2, ytop + th + baseline + 2),
                      color, thickness=cv2.FILLED)

        # Texte (noir ou blanc selon luminance de la couleur)
        luminance = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 1, ytop + th + 1),
                    font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)
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
        print(f"[warn] CUDA indisponible, bascule sur CPU.")
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"[setup] device={device}")

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
    print(f"[weights] Chargé: {weights_path}")

    # --- Image ---
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise RuntimeError(f"Impossible de décoder l'image: {image_path}")
    orig_h, orig_w = orig.shape[:2]
    print(f"[image] {image_path.name} ({orig_w}x{orig_h})")

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
    print(f"[infer] {detections.shape[0]} détection(s) au-dessus du seuil "
          f"conf={cfg.conf_threshold}")

    # --- Noms de classes + palette ---
    if cfg.class_names and len(cfg.class_names) == cfg.num_classes:
        # Cast explicite en str: un YAML peut contenir des ints (ex: ids) ou autres scalaires
        class_names = [str(n) for n in cfg.class_names]
    else:
        if cfg.class_names is not None:
            print(f"[warn] class_names a {len(cfg.class_names)} entrées mais "
                  f"num_classes={cfg.num_classes}. Noms génériques utilisés.")
        class_names = [f"class_{i}" for i in range(cfg.num_classes)]
    colors = build_color_palette(cfg.num_classes)

    # --- Rendu ---
    annotated = orig.copy()
    draw_detections(
        annotated, detections.cpu(), class_names, colors,
        line_thickness=cfg.line_thickness,
        font_scale=cfg.font_scale,
    )

    # Log console des détections
    for i, det in enumerate(detections.cpu()):
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        name = str(class_names[cls_id]) if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  [{i}] {name:<20} conf={conf:.3f} "
              f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    # --- Sauvegarde ---
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), annotated)
        print(f"[save] Image annotée écrite: {save_path}")

    # --- Affichage ---
    if show:
        window = f"YOLOv8 - {image_path.name}"
        try:
            cv2.imshow(window, annotated)
            print("[show] Appuyez sur une touche dans la fenêtre pour fermer.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[warn] Affichage impossible ({e}). "
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
    args = parser.parse_args()

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

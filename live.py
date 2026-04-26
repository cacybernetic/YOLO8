"""
Inférence YOLOv8 ONNX en temps réel sur source vidéo (webcam ou fichier).

Dépendances:
    pip install numpy onnxruntime opencv-python

Usage:
    # Webcam (index 0)
    python live.py --model weights/best.onnx --nc 80 --source 0

    # Fichier vidéo
    python live.py --model weights/best.onnx --nc 80 --source path/to/video.mp4

    # Sauvegarde du flux annoté
    python live.py --model weights/best.onnx --nc 80 --source 0 --output out.mp4

Le script reproduit le pipeline de predict.py (letterbox, NMS, reprojection),
mais affiche le résultat en temps réel via cv2.imshow et calcule un FPS lissé.
La sortie ONNX attendue: (1, 4 + nc, num_anchors).
"""

import argparse
import colorsys
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration loguru locale (script standalone, pas d'accès au module)
# ---------------------------------------------------------------------------
def _setup_logging(level: str = "INFO"):
    """Configure loguru avec un format compact pour ce script standalone.

    Cohérent avec module/utils.py:setup_logging() — même format que les
    autres scripts du projet pour une expérience uniforme.
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
# Noms de classes par défaut (COCO 80) — modifiable selon le modèle utilisé
# ---------------------------------------------------------------------------
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
# Prétraitement (letterbox + normalisation)
# ---------------------------------------------------------------------------

def letterbox(img_bgr: np.ndarray, new_size: int = 640,
              color=(114, 114, 114)):
    """Resize avec préservation du ratio, padding gris pour atteindre new_size carré.

    Sur une boucle vidéo, on évite de réallouer un canvas neuf à chaque frame
    si possible — ici on accepte le coût car l'image source peut changer de
    taille (frames d'un fichier). cv2.resize est bien plus rapide qu'un
    resize PIL ou numpy nu.
    """
    h, w = img_bgr.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))

    if (w, h) != (new_unpad_w, new_unpad_h):
        img_resized = cv2.resize(img_bgr, (new_unpad_w, new_unpad_h),
                                 interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_bgr

    dw = new_size - new_unpad_w
    dh = new_size - new_unpad_h
    pad_left, pad_top = dw // 2, dh // 2

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    canvas[pad_top:pad_top + new_unpad_h,
           pad_left:pad_left + new_unpad_w] = img_resized
    return canvas, r, (pad_left, pad_top)


def preprocess(frame_bgr: np.ndarray, input_size: int):
    """BGR -> letterbox -> RGB -> CHW float32 [0, 1] -> batch."""
    padded, ratio, pad = letterbox(frame_bgr, new_size=input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(tensor), ratio, pad


# ---------------------------------------------------------------------------
# Postprocessing (décodage + NMS numpy pur)
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms_per_class(boxes_xyxy: np.ndarray, scores: np.ndarray,
                  cls_ids: np.ndarray, iou_threshold: float) -> np.ndarray:
    """NMS par classe via cv2.dnn.NMSBoxes (très rapide, bon pour le temps réel).

    cv2.dnn.NMSBoxes fait du NMS par classe nativement quand on offset les
    boites par leur classe. On pourrait aussi appeler cv2.dnn.NMSBoxes en
    boucle par classe, mais l'astuce de l'offset est plus efficace.
    """
    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=np.int64)

    # Format attendu par cv2.dnn.NMSBoxes: (x, y, w, h)
    boxes_wh = np.empty_like(boxes_xyxy)
    boxes_wh[:, 0] = boxes_xyxy[:, 0]
    boxes_wh[:, 1] = boxes_xyxy[:, 1]
    boxes_wh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_wh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    # Décalage par classe pour que des boites de classes différentes ne
    # puissent pas s'éliminer entre elles dans le NMS global.
    max_wh = 7680
    offset = cls_ids.astype(np.float32) * max_wh
    boxes_offset = boxes_wh.copy()
    boxes_offset[:, 0] += offset
    boxes_offset[:, 1] += offset

    indices = cv2.dnn.NMSBoxes(
        boxes_offset.tolist(),
        scores.astype(np.float32).tolist(),
        score_threshold=0.0,        # déjà filtré en amont
        nms_threshold=iou_threshold,
    )
    if indices is None or len(indices) == 0:
        return np.array([], dtype=np.int64)
    # cv2.dnn.NMSBoxes renvoie une shape variable selon la version
    return np.asarray(indices).flatten().astype(np.int64)


def postprocess(output: np.ndarray, num_classes: int,
                conf_threshold: float, iou_threshold: float,
                max_det: int = 300) -> np.ndarray:
    """Décode (1, 4+nc, na) -> (N, 6) [x1, y1, x2, y2, conf, cls]."""
    preds = output[0].transpose(1, 0)  # (na, 4+nc)

    boxes_cxcywh = preds[:, :4]
    cls_scores = preds[:, 4:4 + num_classes]

    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

    mask = confs > conf_threshold
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes_cxcywh = boxes_cxcywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]
    boxes_xyxy = xywh_to_xyxy(boxes_cxcywh)

    keep = nms_per_class(boxes_xyxy, confs, cls_ids, iou_threshold)
    keep = keep[:max_det]

    if keep.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    return np.concatenate([
        boxes_xyxy[keep],
        confs[keep, None],
        cls_ids[keep, None].astype(np.float32),
    ], axis=1)


def scale_boxes_to_original(boxes_xyxy: np.ndarray, ratio: float,
                            pad: tuple, orig_shape: tuple) -> np.ndarray:
    pad_left, pad_top = pad
    oh, ow = orig_shape
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= pad_left
    out[:, [1, 3]] -= pad_top
    out[:, :4] /= ratio
    np.clip(out[:, 0], 0, ow - 1, out=out[:, 0])
    np.clip(out[:, 1], 0, oh - 1, out=out[:, 1])
    np.clip(out[:, 2], 0, ow - 1, out=out[:, 2])
    np.clip(out[:, 3], 0, oh - 1, out=out[:, 3])
    return out


# ---------------------------------------------------------------------------
# Rendu en temps réel (style futuriste, optimisé pour la vidéo)
# ---------------------------------------------------------------------------

def build_color_palette(n: int):
    """Palette HSV équirépartie en BGR (OpenCV)."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def draw_futuristic_box(canvas: np.ndarray, x1, y1, x2, y2,
                        color, conf, label, thickness=2,
                        corner_ratio=0.22, gauge_width=6,
                        font_scale=0.45):
    """Dessine en place sur `canvas`. Style: cadre fin + coins L + tabs + jauge.

    Différence avec la version PIL de predict.py: on dessine directement
    sur le canvas (pas d'overlay avec alpha) car en temps réel chaque
    addWeighted coûterait cher. Le rendu est moins "soft" mais reste lisible.
    """
    h_img, w_img = canvas.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    lt = cv2.LINE_AA

    # 1) Cadre fin
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, lt)

    # 2) Coins en L épais
    cl = max(8, int(min(w, h) * corner_ratio))
    cv2.line(canvas, (x1, y1), (x1 + cl, y1), color, thickness, lt)
    cv2.line(canvas, (x1, y1), (x1, y1 + cl), color, thickness, lt)
    cv2.line(canvas, (x2, y1), (x2 - cl, y1), color, thickness, lt)
    cv2.line(canvas, (x2, y1), (x2, y1 + cl), color, thickness, lt)
    cv2.line(canvas, (x1, y2), (x1 + cl, y2), color, thickness, lt)
    cv2.line(canvas, (x1, y2), (x1, y2 - cl), color, thickness, lt)
    cv2.line(canvas, (x2, y2), (x2 - cl, y2), color, thickness, lt)
    cv2.line(canvas, (x2, y2), (x2, y2 - cl), color, thickness, lt)

    # 3) Tabs aux sommets
    tab = max(2, thickness)
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.rectangle(canvas, (cx - tab, cy - tab), (cx + tab, cy + tab),
                      color, cv2.FILLED)

    # 4) Jauge verticale de confiance à droite
    offset = max(4, thickness + 1)
    gx1 = x2 + offset
    gx2 = gx1 + gauge_width
    if gx2 >= w_img:
        gx2 = max(0, x2 - offset)
        gx1 = max(0, gx2 - gauge_width)
    gy1, gy2 = max(0, y1), min(h_img - 1, y2)
    if gx2 > gx1 and gy2 > gy1:
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), color, 1, lt)
        fill_h = int((gy2 - gy1) * max(0.0, min(1.0, conf)))
        if fill_h > 0:
            cv2.rectangle(canvas, (gx1, gy2 - fill_h), (gx2, gy2),
                          color, cv2.FILLED)

    # 5) Label avec ombre
    text = f"{label}: {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, 1)
    label_x, label_y = x1, y1 - 4
    if label_y - th < 0:
        label_y = y1 + th + 4
    cv2.putText(canvas, text, (label_x + 1, label_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, lt)
    cv2.putText(canvas, text, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, lt)


def render_detections(canvas: np.ndarray, detections: np.ndarray,
                      class_names: list, colors: list,
                      thickness: int = 2):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)
        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        draw_futuristic_box(canvas, x1, y1, x2, y2, color, float(conf), name,
                            thickness=thickness)


def draw_hud(canvas: np.ndarray, fps: float, n_det: int):
    """Petit overlay info en haut à gauche."""
    text_lines = [
        f"FPS: {fps:5.1f}",
        f"Detections: {n_det}",
        "Q/ESC: quit",
    ]
    y = 22
    for line in text_lines:
        cv2.putText(canvas, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


# ---------------------------------------------------------------------------
# Source vidéo (webcam ou fichier)
# ---------------------------------------------------------------------------

def open_capture(source: str) -> cv2.VideoCapture:
    """Ouvre une source vidéo. Si c'est un nombre, c'est un index de webcam,
    sinon un chemin de fichier (ou URL RTSP/HTTP)."""
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        kind = f"webcam #{idx}"
    except ValueError:
        cap = cv2.VideoCapture(source)
        kind = f"file '{source}'"
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source: {kind}")
    logger.info(f"Source ouverte: {kind}")
    return cap


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def run(args):
    # --- Chargement du modèle ONNX ---
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle ONNX introuvable: {model_path}")

    available = ort.get_available_providers()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
        if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']
    logger.info(f"providers={providers}")

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Détection auto de la taille d'entrée si statique
    if len(input_shape) == 4 and isinstance(input_shape[-1], int):
        input_size = input_shape[-1]
    else:
        input_size = args.input_size
    logger.info(f"Modèle: {model_path.name}  input_size={input_size}")

    # --- Source vidéo ---
    cap = open_capture(args.source)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Source: {src_w}x{src_h} @ {src_fps:.1f} fps")

    # --- VideoWriter (optionnel) ---
    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # mp4v est largement compatible. Pour h264 il faut openh264 installé.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (src_w, src_h))
        if not writer.isOpened():
            logger.warning(f"VideoWriter n'a pas pu s'ouvrir pour {out_path}")
            writer = None
        else:
            logger.info(f"Enregistrement: {out_path}")

    # --- Noms de classes + couleurs ---
    if args.names:
        with open(args.names, 'r') as f:
            class_names = [l.strip() for l in f if l.strip()]
        if len(class_names) != args.nc:
            logger.warning(f"{args.names} contient {len(class_names)} noms "
                           f"mais --nc={args.nc}. Complétion par noms génériques.")
            class_names = (class_names + [f"class_{i}" for i in range(args.nc)])[:args.nc]
    elif args.nc == len(CLASS_NAMES):
        class_names = list(CLASS_NAMES)
        logger.info("CLASS_NAMES (COCO, 80) utilisé")
    else:
        class_names = [f"class_{i}" for i in range(args.nc)]
        logger.info(f"Noms génériques (--nc={args.nc} ne matche pas COCO)")
    colors = build_color_palette(args.nc)

    # --- Fenêtre d'affichage ---
    window_name = "YOLOv8 ONNX live"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS lissé sur les N dernières frames pour éviter le scintillement
    fps_window = deque(maxlen=30)
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok or frame is None:
                logger.info("Fin du flux ou frame illisible.")
                break
            orig_h, orig_w = frame.shape[:2]

            # Pipeline: preprocess -> inference -> postprocess -> render
            tensor, ratio, pad = preprocess(frame, input_size=input_size)
            output = session.run(None, {input_name: tensor})[0]
            detections = postprocess(
                output,
                num_classes=args.nc,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
            if len(detections) > 0:
                detections[:, :4] = scale_boxes_to_original(
                    detections[:, :4], ratio, pad, (orig_h, orig_w)
                )
            render_detections(frame, detections, class_names, colors,
                              thickness=args.thickness)

            # FPS
            dt = time.perf_counter() - t0
            fps_window.append(1.0 / max(dt, 1e-6))
            fps = sum(fps_window) / len(fps_window)
            draw_hud(frame, fps, len(detections))

            # Sortie / affichage
            if writer is not None:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow(window_name, frame)
                # Clé: q ou ESC pour quitter
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    logger.info("Quitté par l'utilisateur.")
                    break

            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    logger.success(f"{frame_count} frames en {elapsed:.1f}s "
                   f"(moyenne {avg_fps:.1f} FPS)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inférence YOLOv8 ONNX en temps réel (webcam ou fichier vidéo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True, type=str,
                        help="Chemin du fichier .onnx")
    parser.add_argument('--nc', required=True, type=int,
                        help="Nombre de classes du modèle")
    parser.add_argument('--source', required=True, type=str,
                        help="Source vidéo: index webcam (ex: 0) ou chemin fichier")

    parser.add_argument('--output', type=str, default=None,
                        help="Sauvegarde du flux annoté en .mp4")
    parser.add_argument('--no-show', action='store_true',
                        help="Désactive l'affichage temps réel (utile en headless)")

    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--thickness', type=int, default=2)
    parser.add_argument('--input-size', type=int, default=640,
                        help="Taille d'entrée du modèle (utile si ONNX dynamique)")
    parser.add_argument('--names', type=str, default=None,
                        help="Fichier .txt avec un nom par ligne")
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Niveau de log loguru")
    args = parser.parse_args()

    _setup_logging(level=args.log_level)
    run(args)


if __name__ == '__main__':
    main()

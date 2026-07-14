"""Standalone ONNX inference script for YOLOv8.

This file is fully self contained: it only needs numpy, opencv-python
and onnxruntime. You can copy it anywhere (outside this repository)
and it keeps working.

Usage:
    ylinfer --model weights/best.onnx --nc 10 --image photo.jpg \
            --output result.jpg
    python -m yolov8.entrypoints.inference --model best.onnx --nc 80 \
            --image photo.jpg --show

The model input is a (1, 3, S, S) float32 tensor in [0, 1] (RGB).
The model output is (1, 4 + nc, num_anchors): rows 0..3 hold
(cx, cy, w, h) in letterbox pixel space, the next rows hold the
class scores after sigmoid.
"""

import argparse
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pre and post processing
# ---------------------------------------------------------------------------

def letterbox(img, new_size=640, color=(114, 114, 114)):
    """Resize keeping the aspect ratio, pad to a square."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, r, (pad_x, pad_y)


def preprocess(img_bgr, input_size):
    """BGR image -> (tensor, ratio, padding) for the network."""
    boxed, ratio, pad = letterbox(img_bgr, input_size)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], ratio, pad


def nms_numpy(boxes, scores, iou_threshold):
    """Plain numpy NMS. boxes: (N, 4) xyxy, scores: (N,)."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * \
            np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * \
            (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * \
            (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(output, num_classes, conf_threshold, iou_threshold,
                ratio, pad, orig_shape, max_det=300):
    """Model output -> list of (x1, y1, x2, y2, conf, cls) rows."""
    pred = output[0]                      # (4 + nc, anchors)
    boxes_cxcywh = pred[:4].T             # (anchors, 4)
    scores_all = pred[4:4 + num_classes].T

    conf = scores_all.max(axis=1)
    cls = scores_all.argmax(axis=1)
    keep = conf > conf_threshold
    boxes_cxcywh = boxes_cxcywh[keep]
    conf, cls = conf[keep], cls[keep]
    if boxes_cxcywh.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Per class NMS through the coordinate offset trick.
    offset = cls[:, None].astype(np.float32) * 7680.0
    keep_idx = nms_numpy(boxes + offset, conf, iou_threshold)[:max_det]
    boxes, conf, cls = boxes[keep_idx], conf[keep_idx], cls[keep_idx]

    # Back to the original image space.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    oh, ow = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh - 1)

    out = np.concatenate(
        [boxes, conf[:, None], cls[:, None].astype(np.float32)],
        axis=1)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def class_color(c):
    rng = np.random.RandomState(int(c) + 7)
    return tuple(int(v) for v in rng.randint(64, 255, size=3))


def draw_detections(img, detections, names):
    for x1, y1, x2, y2, conf, cls in detections:
        c = int(cls)
        color = class_color(c)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, 2)
        name = names[c] if c < len(names) else str(c)
        label = f"{name} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)
    return img


COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def load_names(args):
    if args.names:
        with open(args.names, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    if args.nc == 80:
        return COCO_NAMES
    return [f"class_{i}" for i in range(args.nc)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone YOLOv8 ONNX inference on one image.")
    parser.add_argument('--model', required=True,
                        help='Path to the .onnx model')
    parser.add_argument('--image', required=True,
                        help='Path to the input image')
    parser.add_argument('--nc', type=int, required=True,
                        help='Number of classes of the model')
    parser.add_argument('--output', default=None,
                        help='Save the annotated image here')
    parser.add_argument('--names', default=None,
                        help='Text file with one class name per line')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--size', type=int, default=640,
                        help='Network input size')
    parser.add_argument('--show', action='store_true',
                        help='Display the annotated image')
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is required: pip install onnxruntime")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Cannot read image: {args.image}")
        sys.exit(1)

    session = ort.InferenceSession(
        args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    tensor, ratio, pad = preprocess(img, args.size)
    output = session.run(None, {input_name: tensor})[0]
    detections = postprocess(output, args.nc, args.conf, args.iou,
                             ratio, pad, img.shape[:2])
    names = load_names(args)

    print(f"{len(detections)} detection(s)")
    for x1, y1, x2, y2, conf, cls in detections:
        name = names[int(cls)] if int(cls) < len(names) else int(cls)
        print(f"  {name:<20} conf={conf:.3f} "
              f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    annotated = draw_detections(img.copy(), detections, names)
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved: {args.output}")
    if args.show:
        cv2.imshow('yolov8', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

"""Validation of label texts and image bytes."""

import cv2
import numpy as np


def parse_label_text(text, num_classes=None):
    """Validate and parse a YOLO label text in a single pass.

    A line is valid when it has 5 numbers: class cx cy w h, with the
    class a non negative integer (below `num_classes` when given),
    cx and cy inside [0, 1], and w and h inside (0, 1].

    An out-of-range class id would otherwise survive until the loss
    assigner and crash mid-training with an obscure CUDA device-side
    assert, so it must be rejected here.

    Returns:
        (reason, had_bad_lines, labels)
        reason: None when usable (at least one valid line, or an empty
            file which means "image without object"), else one of
            {'empty_label', 'bad_format', 'bad_values', 'bad_class'}.
        had_bad_lines: True when some lines were dropped but the file
            stays usable.
        labels: list of [cls, cx, cy, w, h] rows, or None when invalid.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 0:
        return None, False, []

    valid = []
    n_bad = 0
    worst_reason = None
    for line in lines:
        row, reason = _parse_line(line, num_classes)
        if row is None:
            n_bad += 1
            if worst_reason is None or reason in ('bad_values',
                                                  'bad_class'):
                worst_reason = reason
            continue
        valid.append(row)

    if len(valid) == 0:
        return worst_reason, False, None
    return None, (n_bad > 0), valid


def _parse_line(line, num_classes=None):
    """Parse one label line. Return (row, None) or (None, reason)."""
    parts = line.split()
    if len(parts) != 5:
        return None, 'bad_format'
    try:
        row = [float(x) for x in parts]
    except ValueError:
        return None, 'bad_format'
    cls, cx, cy, w, h = row
    if cls < 0 or cls != int(cls):
        return None, 'bad_values'
    if num_classes is not None and int(cls) >= num_classes:
        return None, 'bad_class'
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None, 'bad_values'
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None, 'bad_values'
    return row, None


def decode_image_bytes(data):
    """Decode image bytes with OpenCV. Return the image or None."""
    if data is None or len(data) == 0:
        return None
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None or img.shape[0] < 2 or img.shape[1] < 2:
        return None
    return img


def check_image_bytes(data):
    """Return True when the bytes decode to a usable image."""
    return decode_image_bytes(data) is not None

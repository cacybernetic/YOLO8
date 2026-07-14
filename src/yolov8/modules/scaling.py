"""Depth, width and ratio factors for each YOLOv8 model size."""


def yolo_params(version):
    """Return (depth, width, ratio) for a model version.

    Args:
        version: one letter among 'n', 's', 'm', 'l', 'x'.

    Returns:
        Tuple (d, w, r) used to scale the network.
    """
    if version == 'n':
        return 1 / 3, 1 / 4, 2.0
    if version == 's':
        return 1 / 3, 1 / 2, 2.0
    if version == 'm':
        return 2 / 3, 3 / 4, 1.5
    if version == 'l':
        return 1.0, 1.0, 1.0
    if version == 'x':
        return 1.0, 1.25, 1.0
    raise ValueError(f"Unknown YOLO version: {version}")

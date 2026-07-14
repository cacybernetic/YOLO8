"""Model tests: output shapes, strides, parameter counts, speed."""

import time

import pytest
import torch

from yolov8.model import MyYolo
from yolov8.modules import yolo_params


def test_yolo_params_known_versions():
    assert yolo_params('n') == (1 / 3, 1 / 4, 2.0)
    assert yolo_params('x') == (1.0, 1.25, 1.0)
    with pytest.raises(ValueError):
        yolo_params('z')


def test_strides_are_calibrated():
    model = MyYolo(version='n', num_classes=3, input_size=128)
    assert model.head.stride.tolist() == [8.0, 16.0, 32.0]


def test_training_output_shapes():
    model = MyYolo(version='n', num_classes=3, input_size=128)
    model.train()
    outputs = model(torch.zeros(2, 3, 128, 128))
    assert len(outputs) == 3
    no = model.head.no  # 64 box channels + 3 classes
    assert outputs[0].shape == (2, no, 16, 16)
    assert outputs[1].shape == (2, no, 8, 8)
    assert outputs[2].shape == (2, no, 4, 4)


def test_eval_output_shapes():
    model = MyYolo(version='n', num_classes=3, input_size=128)
    model.eval()
    inference, raw = model(torch.zeros(1, 3, 128, 128))
    n_anchors = 16 * 16 + 8 * 8 + 4 * 4
    assert inference.shape == (1, 4 + 3, n_anchors)
    assert len(raw) == 3
    # Class scores are sigmoid outputs, inside [0, 1].
    scores = inference[:, 4:, :]
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_parameter_count_nano():
    model = MyYolo(version='n', num_classes=80)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    # YOLOv8n has close to 3 million parameters.
    assert 2.5 < n_params < 4.0


def test_forward_speed_cpu():
    model = MyYolo(version='n', num_classes=3, input_size=128)
    model.eval()
    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        model(x)  # warmup
        start = time.time()
        model(x)
        elapsed = time.time() - start
    # Very loose bound: one small forward must stay under 5 seconds.
    assert elapsed < 5.0

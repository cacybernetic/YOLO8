"""Loss function tests: finite values, empty targets, IoU sanity."""

import torch

from yolov8.lossfn import ComputeLoss, compute_iou
from yolov8.model import MyYolo


def _make_loss_and_outputs(num_classes=3, batch_size=2, size=128):
    model = MyYolo(version='n', num_classes=num_classes,
                   input_size=size)
    loss_fn = ComputeLoss(model, {'box': 7.5, 'cls': 0.5, 'dfl': 1.5})
    model.train()
    outputs = model(torch.zeros(batch_size, 3, size, size))
    return loss_fn, outputs


def test_compute_iou_identity():
    box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    iou = compute_iou(box, box)
    assert float(iou) > 0.99


def test_compute_iou_disjoint():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 150.0, 150.0]])
    iou = compute_iou(a, b)
    # CIoU of disjoint boxes is negative (distance penalty).
    assert float(iou) < 0.0


def test_loss_with_targets_is_finite_and_positive():
    loss_fn, outputs = _make_loss_and_outputs()
    targets = {
        'idx': torch.tensor([0.0, 1.0]),
        'cls': torch.tensor([0.0, 2.0]),
        'box': torch.tensor([[0.5, 0.5, 0.4, 0.4],
                             [0.3, 0.3, 0.2, 0.2]]),
    }
    loss_box, loss_cls, loss_dfl = loss_fn(outputs, targets)
    for value in (loss_box, loss_cls, loss_dfl):
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0


def test_loss_with_empty_targets():
    loss_fn, outputs = _make_loss_and_outputs()
    targets = {
        'idx': torch.zeros(0),
        'cls': torch.zeros(0),
        'box': torch.zeros((0, 4)),
    }
    loss_box, loss_cls, loss_dfl = loss_fn(outputs, targets)
    assert float(loss_box) == 0.0
    assert float(loss_dfl) == 0.0
    assert torch.isfinite(loss_cls)


def test_loss_accepts_eval_tuple():
    loss_fn, _ = _make_loss_and_outputs()
    model = MyYolo(version='n', num_classes=3, input_size=128)
    loss_fn2 = ComputeLoss(model, {'box': 7.5, 'cls': 0.5, 'dfl': 1.5})
    model.eval()
    out = model(torch.zeros(1, 3, 128, 128))
    targets = {
        'idx': torch.tensor([0.0]),
        'cls': torch.tensor([1.0]),
        'box': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }
    losses = loss_fn2(out, targets)  # tuple (inference, raw) accepted
    assert all(torch.isfinite(v) for v in losses)

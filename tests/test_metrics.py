"""Metric accuracy tests on small hand-checked cases."""

import numpy as np
import torch

from yolov8.metrics import (MetricAccumulator, non_max_suppression,
                            wh2xy, box_iou_numpy,
                            match_predictions_to_gt,
                            compute_ap_per_class,
                            build_confusion_matrix)


def test_wh2xy():
    box = torch.tensor([[10.0, 10.0, 4.0, 6.0]])
    out = wh2xy(box)
    assert out.tolist() == [[8.0, 7.0, 12.0, 13.0]]


def test_box_iou_numpy_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 5.0]])
    iou = box_iou_numpy(a, b)
    assert abs(float(iou[0, 0]) - 0.5) < 1e-6


def test_perfect_predictions_give_map_one():
    accumulator = MetricAccumulator(device=torch.device('cpu'))
    # Predictions exactly on the ground truths, right classes.
    preds = [torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0],
                           [60.0, 60.0, 90.0, 90.0, 0.8, 1.0]])]
    gts = [torch.tensor([[0.0, 10.0, 10.0, 50.0, 50.0],
                         [1.0, 60.0, 60.0, 90.0, 90.0]])]
    accumulator.update(preds, gts)
    results = accumulator.compute()
    assert results['map50'] > 0.99
    assert results['map'] > 0.99


def test_wrong_class_gives_zero_map():
    accumulator = MetricAccumulator(device=torch.device('cpu'))
    preds = [torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 1.0]])]
    gts = [torch.tensor([[0.0, 10.0, 10.0, 50.0, 50.0]])]
    accumulator.update(preds, gts)
    results = accumulator.compute()
    assert results['map50'] == 0.0


def test_accumulator_state_roundtrip():
    acc1 = MetricAccumulator(device=torch.device('cpu'))
    preds = [torch.tensor([[10.0, 10.0, 50.0, 50.0, 0.9, 0.0]])]
    gts = [torch.tensor([[0.0, 10.0, 10.0, 50.0, 50.0]])]
    acc1.update(preds, gts)
    state = acc1.state_dict()

    acc2 = MetricAccumulator(device=torch.device('cpu'))
    acc2.load_state_dict(state)
    assert acc2.compute() == acc1.compute()


def test_nms_removes_duplicates():
    # Raw output layout: (bs, 4 + nc, n_anchors), boxes as cx cy w h.
    nc = 2
    out = torch.zeros(1, 4 + nc, 3)
    # Two overlapping boxes of class 0 and one far box of class 1.
    out[0, :4, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0])
    out[0, 4, 0] = 0.9
    out[0, :4, 1] = torch.tensor([51.0, 51.0, 20.0, 20.0])
    out[0, 4, 1] = 0.8
    out[0, :4, 2] = torch.tensor([200.0, 200.0, 20.0, 20.0])
    out[0, 5, 2] = 0.7
    preds = non_max_suppression(out, confidence_threshold=0.25,
                                iou_threshold=0.5)
    assert preds[0].shape[0] == 2  # duplicate suppressed


def test_match_predictions_to_gt_thresholds():
    preds = np.array([[0.0, 0.0, 10.0, 10.0]])
    gts = np.array([[0.0, 0.0, 10.0, 8.0]])  # IoU = 0.8
    tp = match_predictions_to_gt(
        preds, np.array([0]), np.array([0.9]), gts, np.array([0]),
        np.array([0.5, 0.9]))
    assert tp[0, 0]          # IoU 0.8 >= 0.5 -> TP
    assert not tp[0, 1]      # IoU 0.8 < 0.9 -> FP


def test_compute_ap_per_class_perfect():
    tp = np.ones((2, 10), dtype=bool)
    conf = np.array([0.9, 0.8])
    pred_cls = np.array([0, 0])
    target_cls = np.array([0, 0])
    per_class = compute_ap_per_class(tp, conf, pred_cls, target_cls,
                                     num_classes=1)
    assert per_class[0]['ap50'] > 0.99
    assert per_class[0]['n_gt'] == 2


def test_confusion_matrix_counts():
    preds = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1]),
              np.array([0.9]))]
    gts = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0]))]
    cm = build_confusion_matrix(preds, gts, num_classes=2,
                                iou_threshold=0.5, conf_threshold=0.25)
    # Class 0 ground truth predicted as class 1.
    assert cm[1, 0] == 1
    assert cm.sum() == 1

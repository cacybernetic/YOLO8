"""Detailed evaluation metrics: per class AP, confusion matrix, tables.

References:
  - Padilla et al., "A Survey on Performance Metrics for Object-Detection
    Algorithms" (2020)
  - COCO evaluation protocol (Lin et al., 2014)
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .boxes import box_iou_numpy


def match_predictions_to_gt(preds_xyxy, pred_cls, pred_conf,
                            gts_xyxy, gt_cls, iou_thresholds):
    """For each prediction, tell if it is a TP at each IoU threshold.

    Standard COCO greedy matching:
      1. Sort the predictions by decreasing confidence.
      2. For each prediction, find the best free ground truth of the same
         class whose IoU is above the threshold.
      3. Found: TP and the ground truth is marked as used. Else: FP.

    A ground truth that no prediction matches becomes a FN.

    Returns:
        tp_matrix: (n_pred, n_iou) bool array
    """
    n_pred = preds_xyxy.shape[0]
    n_iou = len(iou_thresholds)
    tp_matrix = np.zeros((n_pred, n_iou), dtype=bool)

    if n_pred == 0 or gts_xyxy.shape[0] == 0:
        return tp_matrix

    iou_mat = box_iou_numpy(preds_xyxy, gts_xyxy)
    cls_match = pred_cls[:, None] == gt_cls[None, :]
    iou_mat = np.where(cls_match, iou_mat, 0.0)

    order = np.argsort(-pred_conf)
    for k, iou_thr in enumerate(iou_thresholds):
        gt_matched = np.zeros(gts_xyxy.shape[0], dtype=bool)
        for i in order:
            ious_i = np.where(gt_matched, -1.0, iou_mat[i])
            j_best = int(np.argmax(ious_i))
            if ious_i[j_best] >= iou_thr:
                tp_matrix[i, k] = True
                gt_matched[j_best] = True
    return tp_matrix


def compute_ap_per_class(tp_matrix, conf, pred_cls, target_cls,
                         num_classes, eps=1e-9):
    """Compute AP, P, R and F1 curves per class (COCO 101 points).

    Returns:
        dict[cls_id] -> {
            'ap50', 'ap5095', 'p_curve', 'r_curve', 'f1_curve',
            'n_gt', 'n_pred'
        }
    """
    order = np.argsort(-conf)
    tp_matrix = tp_matrix[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    px = np.linspace(0, 1, 1000)
    results: Dict[int, dict] = {}

    for c in range(num_classes):
        mask_pred = pred_cls == c
        n_pred_c = int(mask_pred.sum())
        n_gt_c = int(np.sum(target_cls == c))

        if n_gt_c == 0 and n_pred_c == 0:
            continue
        if n_gt_c == 0 or n_pred_c == 0:
            # Only FP, or only FN: AP is zero, flat curves.
            results[c] = {
                'ap50': 0.0, 'ap5095': 0.0,
                'p_curve': np.zeros_like(px),
                'r_curve': np.zeros_like(px),
                'f1_curve': np.zeros_like(px),
                'n_gt': n_gt_c, 'n_pred': n_pred_c,
            }
            continue

        results[c] = _class_ap(
            tp_matrix[mask_pred], conf[mask_pred],
            n_gt_c, n_pred_c, px, eps)

    return results


def _class_ap(tp_c, conf_c, n_gt_c, n_pred_c, px, eps):
    """AP and curves for one class. tp_c: (n_pred_c, n_iou)."""
    tp_cum = np.cumsum(tp_c, axis=0).astype(np.float64)
    fp_cum = np.cumsum(~tp_c, axis=0).astype(np.float64)
    recall_cum = tp_cum / (n_gt_c + eps)
    precision_cum = tp_cum / (tp_cum + fp_cum + eps)

    n_iou = tp_c.shape[1]
    ap_per_iou = np.zeros(n_iou)
    for k in range(n_iou):
        r = recall_cum[:, k]
        p = precision_cum[:, k]
        p_envelope = np.flip(np.maximum.accumulate(np.flip(p)))
        x_eval = np.linspace(0, 1, 101)
        # COCO interpolation: below the first reached recall, the
        # precision is the envelope start; above the last reached
        # recall, the precision is zero.
        p_interp = np.interp(x_eval, r, p_envelope,
                             left=float(p_envelope[0]), right=0.0)
        ap_per_iou[k] = p_interp.mean()

    # Curves on the confidence grid, at IoU 0.5 (k = 0).
    p_curve = np.interp(-px, -conf_c, precision_cum[:, 0], left=1)
    r_curve = np.interp(-px, -conf_c, recall_cum[:, 0], left=0)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    return {
        'ap50': float(ap_per_iou[0]),
        'ap5095': float(ap_per_iou.mean()),
        'p_curve': p_curve,
        'r_curve': r_curve,
        'f1_curve': f1_curve,
        'n_gt': n_gt_c,
        'n_pred': n_pred_c,
    }


def find_best_f1_threshold(per_class, px) -> Tuple[float, float]:
    """Return (best confidence threshold, macro F1 at that threshold).

    The threshold maximizes the mean of the per-class F1 curves. This is
    the Ultralytics convention (see plots/F1_curve.png).
    """
    if not per_class:
        return 0.0, 0.0
    f1_stack = np.stack(
        [v['f1_curve'] for v in per_class.values()], axis=0)
    f1_macro = f1_stack.mean(axis=0)
    best_idx = int(np.argmax(f1_macro))
    return float(px[best_idx]), float(f1_macro[best_idx])


def metrics_at_threshold(per_class, px, conf_threshold):
    """P, R, F1 per class at a given confidence threshold."""
    idx = int(np.argmin(np.abs(px - conf_threshold)))
    out = {}
    for c, d in per_class.items():
        out[c] = {
            'precision': float(d['p_curve'][idx]),
            'recall': float(d['r_curve'][idx]),
            'f1': float(d['f1_curve'][idx]),
        }
    return out


def build_confusion_matrix(all_preds, all_gts, num_classes,
                           iou_threshold=0.45, conf_threshold=0.25):
    """Build a (N+1) x (N+1) confusion matrix.

    Rows are predicted classes, columns are true classes; the extra index
    N is the background. cm[N, j] counts missed ground truths (FN) and
    cm[i, N] counts predictions without any ground truth (FP).

    Args:
        all_preds: per image (preds_xyxy, pred_cls, pred_conf)
        all_gts: per image (gts_xyxy, gt_cls)
    """
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    bg = num_classes

    for (preds_xyxy, pred_cls, pred_conf), (gts_xyxy, gt_cls) in \
            zip(all_preds, all_gts):
        keep = pred_conf >= conf_threshold
        preds_xyxy = preds_xyxy[keep]
        pred_cls_kept = pred_cls[keep]
        conf_kept = pred_conf[keep]

        n_pred = preds_xyxy.shape[0]
        n_gt = gts_xyxy.shape[0]

        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            for c in pred_cls_kept.astype(int):
                cm[c, bg] += 1
            continue
        if n_pred == 0:
            for c in gt_cls.astype(int):
                cm[bg, c] += 1
            continue

        _fill_confusion(cm, preds_xyxy, pred_cls_kept, conf_kept,
                        gts_xyxy, gt_cls, iou_threshold, bg)

    return cm


def _fill_confusion(cm, preds_xyxy, pred_cls, pred_conf,
                    gts_xyxy, gt_cls, iou_threshold, bg):
    """Greedy IoU matching, class is free so we can see confusions."""
    n_pred = preds_xyxy.shape[0]
    n_gt = gts_xyxy.shape[0]
    iou_mat = box_iou_numpy(preds_xyxy, gts_xyxy)
    order = np.argsort(-pred_conf)

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    for i in order:
        ious_i = np.where(gt_matched, -1.0, iou_mat[i])
        j_best = int(np.argmax(ious_i))
        if ious_i[j_best] >= iou_threshold:
            cm[int(pred_cls[i]), int(gt_cls[j_best])] += 1
            gt_matched[j_best] = True
            pred_matched[i] = True

    for j in np.where(~gt_matched)[0]:
        cm[bg, int(gt_cls[j])] += 1
    for i in np.where(~pred_matched)[0]:
        cm[int(pred_cls[i]), bg] += 1


def build_per_class_table(per_class, metrics_at_thr, class_names,
                          cls_loss_per_class=None) -> pd.DataFrame:
    """Per class metric table.

    Columns: class_id, class_name, n_gt, n_pred, precision, recall, f1,
             ap50, ap50_95 and optionally cls_loss.
    """
    rows = []
    for c, d in sorted(per_class.items()):
        m = metrics_at_thr.get(c, {})
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        row = {
            'class_id': c,
            'class_name': name,
            'n_gt': d['n_gt'],
            'n_pred': d['n_pred'],
            'precision': round(m.get('precision', 0.0), 4),
            'recall': round(m.get('recall', 0.0), 4),
            'f1': round(m.get('f1', 0.0), 4),
            'ap50': round(d['ap50'], 4),
            'ap50_95': round(d['ap5095'], 4),
        }
        if cls_loss_per_class is not None:
            row['cls_loss'] = round(cls_loss_per_class.get(c, 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_global_table(per_class, metrics_at_thr, losses, iou_mean,
                       best_conf_threshold, best_f1, counts) -> pd.DataFrame:
    """Global metric table.

    Macro values are simple means over the classes. Micro values are
    computed from the global TP / FP / FN counts.

    Args:
        counts: dict with n_gt, n_pred, n_tp, n_fp, n_fn.
    """
    if per_class:
        map50 = float(np.mean([d['ap50'] for d in per_class.values()]))
        map5095 = float(np.mean(
            [d['ap5095'] for d in per_class.values()]))
        p_macro = float(np.mean(
            [metrics_at_thr.get(c, {}).get('precision', 0.0)
             for c in per_class]))
        r_macro = float(np.mean(
            [metrics_at_thr.get(c, {}).get('recall', 0.0)
             for c in per_class]))
        f1_macro = float(np.mean(
            [metrics_at_thr.get(c, {}).get('f1', 0.0)
             for c in per_class]))
    else:
        map50 = map5095 = p_macro = r_macro = f1_macro = 0.0

    eps = 1e-9
    p_micro = counts['n_tp'] / (counts['n_tp'] + counts['n_fp'] + eps)
    r_micro = counts['n_tp'] / (counts['n_tp'] + counts['n_fn'] + eps)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + eps)

    rows = [
        ('mAP50',           round(map50, 4)),
        ('mAP50-95',        round(map5095, 4)),
        ('precision_macro', round(p_macro, 4)),
        ('recall_macro',    round(r_macro, 4)),
        ('f1_macro',        round(f1_macro, 4)),
        ('precision_micro', round(p_micro, 4)),
        ('recall_micro',    round(r_micro, 4)),
        ('f1_micro',        round(f1_micro, 4)),
        ('iou_mean',        round(iou_mean, 4)),
        ('box_loss',        round(losses.get('box', 0.0), 4)),
        ('cls_loss',        round(losses.get('cls', 0.0), 4)),
        ('dfl_loss',        round(losses.get('dfl', 0.0), 4)),
        ('total_loss',      round(losses.get('total', 0.0), 4)),
        ('best_conf_threshold', round(best_conf_threshold, 4)),
        ('best_f1_macro',   round(best_f1, 4)),
        ('n_total_gt',      int(counts['n_gt'])),
        ('n_total_pred',    int(counts['n_pred'])),
        ('n_total_tp',      int(counts['n_tp'])),
        ('n_total_fp',      int(counts['n_fp'])),
        ('n_total_fn',      int(counts['n_fn'])),
    ]
    return pd.DataFrame(rows, columns=['metric', 'value'])

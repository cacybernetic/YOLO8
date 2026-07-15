"""Fast mAP computation used for the per-epoch validation."""

import numpy as np
import torch


def average_precision_101(recall, precision):
    """COCO 101-point interpolated AP for one class at one IoU.

    Single implementation shared by the per-epoch validation and the
    full evaluation CLI, so `map50` means the same thing in
    history.csv and in results.csv.

    Args:
        recall: cumulative recall values, sorted by descending conf.
        precision: matching cumulative precision values.
    """
    p_envelope = np.flip(np.maximum.accumulate(np.flip(precision)))
    x_eval = np.linspace(0, 1, 101)
    # Below the first reached recall, the precision is the envelope
    # start; above the last reached recall, the precision is zero.
    p_interp = np.interp(x_eval, recall, p_envelope,
                         left=float(p_envelope[0]), right=0.0)
    return float(p_interp.mean())


def compute_metric(output, target, iou_v):
    """Return a (n_pred, n_iou) bool array: TP flag at each IoU threshold.

    Args:
        output: (n_pred, 6) [x1, y1, x2, y2, conf, cls]
        target: (n_gt, 5)   [cls, x1, y1, x2, y2]
        iou_v:  (n_iou,) IoU thresholds
    """
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)) \
        .clamp(0).prod(2)
    iou = intersection / (
        (a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = np.zeros((output.shape[0], iou_v.shape[0])).astype(bool)
    for i in range(len(iou_v)):
        x = torch.where(
            (iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
            ).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[
                    np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[
                    np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def _smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Compute AP@0.5 and mAP@0.5:0.95.

    Args:
        tp: (N, n_iou) bool, TP flag per prediction per IoU threshold
        conf: (N,) confidence score
        pred_cls: (N,) predicted class
        target_cls: (M,) ground truth classes

    Returns:
        tp, fp, precision_mean, recall_mean, map50, mean_ap
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px = np.linspace(0, 1, 1000)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]
        no = i.sum()
        if no == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j] = average_precision_101(
                recall[:, j], precision[:, j])

    f1 = 2 * p * r / (p + r + eps)
    i = _smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp_out = (r * nt).round()
    fp_out = (tp_out / (p + eps) - tp_out).round()
    ap50, ap_all = ap[:, 0], ap.mean(1)
    return tp_out, fp_out, p.mean(), r.mean(), ap50.mean(), ap_all.mean()


class MetricAccumulator:
    """Collect predictions over a dataset and compute the final metrics.

    The internal stats can be exported and restored with state_dict and
    load_state_dict, so an evaluation pass can resume after a crash.
    """

    def __init__(self, device, iou_thresholds=None):
        self.device = device
        self.iou_v = iou_thresholds if iou_thresholds is not None else \
            torch.linspace(0.5, 0.95, 10)
        self.stats = []  # per image: (correct, conf, pred_cls, target_cls)

    def update(self, preds, targets_per_image):
        """
        Args:
            preds: list of NMS outputs, one per image, each (n_pred, 6)
            targets_per_image: list of GT tensors, one per image,
                each (n_gt, 5) [cls, x1, y1, x2, y2] in pixel units
        """
        iou_v = self.iou_v.to(self.device)
        for pred, target in zip(preds, targets_per_image):
            nl = target.shape[0]
            if pred.shape[0] == 0:
                if nl:
                    self.stats.append((
                        torch.zeros(0, self.iou_v.numel(),
                                    dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        target[:, 0].cpu(),
                    ))
                continue

            if nl:
                correct = compute_metric(
                    pred, target.to(self.device), iou_v)
            else:
                correct = torch.zeros(
                    pred.shape[0], self.iou_v.numel(),
                    dtype=torch.bool, device=pred.device)

            self.stats.append((
                correct.cpu(),
                pred[:, 4].cpu(),
                pred[:, 5].cpu(),
                target[:, 0].cpu() if nl else torch.zeros(0),
            ))

    def compute(self):
        """Return a dict with the final metric values."""
        if len(self.stats) == 0:
            return dict(precision=0.0, recall=0.0, map50=0.0, map=0.0)

        tp = np.concatenate([s[0].numpy() for s in self.stats], axis=0)
        conf = np.concatenate([s[1].numpy() for s in self.stats], axis=0)
        pred_cls = np.concatenate(
            [s[2].numpy() for s in self.stats], axis=0)
        target_cls = np.concatenate(
            [s[3].numpy() for s in self.stats], axis=0)

        if tp.shape[0] == 0 or target_cls.shape[0] == 0:
            return dict(precision=0.0, recall=0.0, map50=0.0, map=0.0)

        _, _, mp, mr, map50, mean_ap = compute_ap(
            tp, conf, pred_cls, target_cls)
        return dict(precision=float(mp), recall=float(mr),
                    map50=float(map50), map=float(mean_ap))

    def state_dict(self):
        """Export the collected stats (used by mid-pass checkpoints)."""
        return {'stats': self.stats}

    def load_state_dict(self, state):
        """Restore stats saved by state_dict."""
        self.stats = list(state.get('stats', []))

    def reset(self):
        self.stats = []

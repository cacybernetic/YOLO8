"""
Métriques pour la détection d'objets : mAP@0.5, mAP@0.5:0.95, Precision, Recall.

S'appuie sur NMS + matching IoU multi-seuils (évaluation COCO-style).
"""

from time import time

import numpy as np
import torch
import torchvision


# ---------------------------------------------------------------------------
# Utilitaires de conversion
# ---------------------------------------------------------------------------

def wh2xy(x):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.7):
    """NMS par classe. `outputs`: (bs, 4+nc, n_anchors) - sortie brute du modèle en eval."""
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]
    nc = outputs.shape[1] - 4
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold  # candidats

    start = time()
    limit = 0.5 + 0.05 * bs
    output = [torch.zeros((0, 6), device=outputs.device)] * bs

    for index, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[index]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # offset par classe
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)[:max_det]
        output[index] = x[indices]
        if (time() - start) > limit:
            break
    return output


# ---------------------------------------------------------------------------
# Matching TP/FP pour mAP
# ---------------------------------------------------------------------------

def compute_metric(output, target, iou_v):
    """
    Retourne un tableau (n_pred, n_iou_thresh) indiquant TP à chaque seuil IoU.

    Args:
        output: (n_pred, 6) [x1, y1, x2, y2, conf, cls]
        target: (n_gt, 5)   [cls, x1, y1, x2, y2]
        iou_v:  (n_iou,) seuils IoU
    """
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = np.zeros((output.shape[0], iou_v.shape[0])).astype(bool)
    for i in range(len(iou_v)):
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
            ).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


# ---------------------------------------------------------------------------
# Agrégation (mAP, P, R)
# ---------------------------------------------------------------------------

def _smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """Calcule AP@0.5 et mAP@0.5:0.95.

    Args:
        tp: (N, n_iou) bool (TP par prediction par seuil IoU)
        conf: (N,) score de confiance
        pred_cls: (N,) classe prédite
        target_cls: (M,) classes GT
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
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            x = np.linspace(0, 1, 101)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    f1 = 2 * p * r / (p + r + eps)
    i = _smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp_out = (r * nt).round()
    fp_out = (tp_out / (p + eps) - tp_out).round()
    ap50, ap_all = ap[:, 0], ap.mean(1)
    return tp_out, fp_out, p.mean(), r.mean(), ap50.mean(), ap_all.mean()


# ---------------------------------------------------------------------------
# Évaluateur de haut niveau
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """Accumule les prédictions sur tout un dataset et calcule les métriques."""

    def __init__(self, device, iou_thresholds=None):
        self.device = device
        self.iou_v = iou_thresholds if iou_thresholds is not None else \
            torch.linspace(0.5, 0.95, 10)
        self.stats = []  # liste de (correct, conf, pred_cls, target_cls) par image

    def update(self, preds, targets_per_image):
        """
        Args:
            preds: liste des sorties NMS, une par image, chacune (n_pred, 6)
            targets_per_image: liste de tenseurs GT, un par image, chacun (n_gt, 5)
                               format [cls, x1, y1, x2, y2] en coords pixel
        """
        iou_v = self.iou_v.to(self.device)
        for pred, target in zip(preds, targets_per_image):
            nl = target.shape[0]
            if pred.shape[0] == 0:
                if nl:
                    self.stats.append((
                        torch.zeros(0, self.iou_v.numel(), dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        target[:, 0].cpu(),
                    ))
                continue

            if nl:
                correct = compute_metric(pred, target.to(self.device), iou_v)
            else:
                correct = torch.zeros(pred.shape[0], self.iou_v.numel(),
                                      dtype=torch.bool, device=pred.device)

            self.stats.append((
                correct.cpu(),
                pred[:, 4].cpu(),
                pred[:, 5].cpu(),
                target[:, 0].cpu() if nl else torch.zeros(0),
            ))

    def compute(self):
        """Retourne un dict avec les métriques finales."""
        if len(self.stats) == 0:
            return dict(precision=0.0, recall=0.0, map50=0.0, map=0.0)

        tp = np.concatenate([s[0].numpy() for s in self.stats], axis=0)
        conf = np.concatenate([s[1].numpy() for s in self.stats], axis=0)
        pred_cls = np.concatenate([s[2].numpy() for s in self.stats], axis=0)
        target_cls = np.concatenate([s[3].numpy() for s in self.stats], axis=0)

        if tp.shape[0] == 0 or target_cls.shape[0] == 0:
            return dict(precision=0.0, recall=0.0, map50=0.0, map=0.0)

        _, _, mp, mr, map50, mean_ap = compute_ap(tp, conf, pred_cls, target_cls)
        return dict(precision=float(mp), recall=float(mr),
                    map50=float(map50), map=float(mean_ap))

    def reset(self):
        self.stats = []

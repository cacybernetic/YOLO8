"""
Implémentation de la fonction de perte YOLOv8.

Corrections par rapport à la version initiale :
  - `Assigner`: le tenseur `ind` est désormais créé sur le bon device (bug qui
    provoquait une erreur d'indexation quand les prédictions étaient sur GPU)
  - `Assigner`: cast explicite `gt_labels` -> long pour éviter les warnings
  - `ComputeLoss`: accepte directement la sortie de la tête (liste de 3 tenseurs
    en mode train, tuple (inference, raw) en mode eval) et choisit la bonne branche
  - `ComputeLoss`: gestion propre du cas "batch sans aucune boite"
  - Utilise les `stride` calibrés du modèle (nécessite MyYolo avec _initialize_strides)
"""

import math

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


# ---------------------------------------------------------------------------
# IoU / CIoU
# ---------------------------------------------------------------------------

def compute_iou(box1, box2, eps=1e-7):
    """Retourne la CIoU entre box1 et box2 (format xyxy)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def make_anchors(x, strides, offset=0.5):
    """Génère les centres d'ancres et les strides par ancre."""
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner (YOLOv8)
# ---------------------------------------------------------------------------

class Assigner(nn.Module):
    """Task-Aligned One-stage Object Detection Assigner (TAL).

    Papier: https://arxiv.org/abs/2108.07755 et l'implémentation officielle YOLOv8.
    """

    def __init__(self, nc=80, top_k=13, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores:  (b, n_anchors, nc) scores de classification (sigmoid)
            pd_bboxes:  (b, n_anchors, 4) boites prédites xyxy (espace image)
            anc_points: (n_anchors, 2) centres d'ancres (espace image)
            gt_labels:  (b, max_num_obj, 1)
            gt_bboxes:  (b, max_num_obj, 4) xyxy (espace image)
            mask_gt:    (b, max_num_obj, 1) masque des GT valides
        """
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)
        device = pd_bboxes.device

        if num_max_boxes == 0:
            return (torch.zeros_like(pd_bboxes, device=device),
                    torch.zeros_like(pd_scores, device=device),
                    torch.zeros_like(pd_scores[..., 0], device=device))

        num_anchors = anc_points.shape[0]
        shape = gt_bboxes.shape

        # Masque "ancre à l'intérieur de la GT"
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        mask_in_gts = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.view(shape[0], shape[1], num_anchors, -1).amin(3).gt_(self.eps)

        na = pd_bboxes.shape[-2]
        gt_mask = (mask_in_gts * mask_gt).bool()  # (b, max_obj, na)

        overlaps = torch.zeros([batch_size, num_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na],
                                  dtype=pd_scores.dtype, device=device)

        # *** FIX: ind doit être sur le même device que pd_scores/gt_labels ***
        ind = torch.zeros([2, batch_size, num_max_boxes],
                          dtype=torch.long, device=device)
        ind[0] = torch.arange(end=batch_size, device=device).view(-1, 1).expand(-1, num_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()  # cast explicite
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Top-K
        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=device)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        # Résolution des conflits (ancre assignée à plusieurs GT)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)

        # Cibles
        batch_index = torch.arange(end=batch_size, dtype=torch.int64, device=device)[..., None]
        target_index = target_gt_idx + batch_index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_index]
        target_labels.clamp_(0)

        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.nc),
            dtype=torch.int64, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalisation
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps /
                             (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()


# ---------------------------------------------------------------------------
# Loss Box + DFL
# ---------------------------------------------------------------------------

class BoxLoss(nn.Module):
    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch  # reg_max - 1

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU (CIoU) loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.dfl_ch + 1),
            target[fg_mask]
        )
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        return loss_box, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        left_loss = cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Compute Loss (orchestrateur)
# ---------------------------------------------------------------------------

class ComputeLoss:
    """Fonction de perte complète YOLOv8 (box CIoU + DFL + BCE classif)."""

    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device
        m = model.head

        # Garde-fou: strides doivent être calibrés
        if torch.all(m.stride == 0):
            raise RuntimeError(
                "model.head.stride n'est pas initialisé. "
                "Utilisez MyYolo (qui le calibre automatiquement) ou appelez "
                "model._initialize_strides(input_size) avant de créer la loss."
            )

        self.params = params
        self.stride = m.stride.to(device)
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)
        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: liste de 3 tenseurs [bs, no, h, w] (mode train).
                     Si c'est un tuple (inf, raw) (mode eval), on prend `raw`.
            targets: dict {'idx': (N,), 'cls': (N,), 'box': (N, 4)}
                     box en format YOLO (cx, cy, w, h) normalisé [0,1]
        """
        # Si la tête a renvoyé (inference, raw), on prend `raw`
        if isinstance(outputs, tuple):
            outputs = outputs[1]

        x = torch.cat([i.view(outputs[0].shape[0], self.no, -1) for i in outputs], dim=2)
        pred_distri, pred_scores = x.split(split_size=(self.reg_max * 4, self.nc), dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   # (b, n_anchors, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   # (b, n_anchors, 4*reg_max)

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(outputs[0].shape[2:], device=self.device, dtype=data_type) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(outputs, self.stride, offset=0.5)

        # Construction des GT
        idx = targets['idx'].view(-1, 1).to(self.device)
        cls = targets['cls'].view(-1, 1).to(self.device)
        box = targets['box'].to(self.device)  # (N, 4) cx, cy, w, h normalisé

        targets_cat = torch.cat((idx, cls, box), dim=1)

        if targets_cat.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 5, device=self.device, dtype=data_type)
        else:
            i = targets_cat[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, int(counts.max()), 5,
                             device=self.device, dtype=data_type)
            for j in range(batch_size):
                matches = (i == j)
                n = matches.sum()
                if n:
                    gt[j, :n] = targets_cat[matches, 1:].to(data_type)
            # cx, cy, w, h normalisés -> xyxy en espace image
            xywh = gt[..., 1:5].clone()
            xywh.mul_(input_size[[1, 0, 1, 0]])  # multiplie par (W, H, W, H)
            y = torch.empty_like(xywh)
            dw = xywh[..., 2] / 2
            dh = xywh[..., 3] / 2
            y[..., 0] = xywh[..., 0] - dw
            y[..., 1] = xywh[..., 1] - dh
            y[..., 2] = xywh[..., 0] + dw
            y[..., 3] = xywh[..., 1] + dh
            gt[..., 1:5] = y

        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Décodage des prédictions
        pred_bboxes = self.box_decode(anchor_points, pred_distri)  # (b, na, 4) xyxy

        # Assignation
        assigned = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt
        )
        target_bboxes, target_scores, fg_mask = assigned

        target_scores_sum = max(target_scores.sum().item(), 1)

        # Classification loss (BCE)
        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum

        # Box + DFL loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            loss_box, loss_dfl = self.box_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss_box = loss_box * self.params['box']
        loss_cls = loss_cls * self.params['cls']
        loss_dfl = loss_dfl * self.params['dfl']

        return loss_box, loss_cls, loss_dfl

"""YOLOv8 loss function: TAL assigner + CIoU + DFL + BCE."""

import math

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from yolov8.modules.anchors import make_anchors  # noqa: F401 (re-export)


# ---------------------------------------------------------------------------
# IoU / CIoU
# ---------------------------------------------------------------------------

def compute_iou(box1, box2, eps=1e-7):
    """Return the CIoU between box1 and box2 (xyxy format)."""
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
    v = (4 / math.pi ** 2) * \
        (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner (YOLOv8)
# ---------------------------------------------------------------------------

class Assigner(nn.Module):
    """Task-Aligned assigner (TAL).

    Paper: https://arxiv.org/abs/2108.07755 and the official YOLOv8 code.
    """

    def __init__(self, nc=80, top_k=13, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points,
                gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores:  (b, n_anchors, nc) class scores after sigmoid
            pd_bboxes:  (b, n_anchors, 4) predicted boxes xyxy (image space)
            anc_points: (n_anchors, 2) anchor centers (image space)
            gt_labels:  (b, max_num_obj, 1)
            gt_bboxes:  (b, max_num_obj, 4) xyxy (image space)
            mask_gt:    (b, max_num_obj, 1) mask of valid ground truths
        """
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)
        device = pd_bboxes.device

        if num_max_boxes == 0:
            return (torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros(pd_scores.shape[:2], dtype=torch.bool,
                                device=device))

        num_anchors = anc_points.shape[0]
        shape = gt_bboxes.shape

        # Mask "anchor center is inside the ground truth box".
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        mask_in_gts = torch.cat(
            (anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = mask_in_gts.view(
            shape[0], shape[1], num_anchors, -1).amin(3).gt_(self.eps)

        na = pd_bboxes.shape[-2]
        gt_mask = (mask_in_gts * mask_gt).bool()  # (b, max_obj, na)

        overlaps = torch.zeros([batch_size, num_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na],
                                  dtype=pd_scores.dtype, device=device)

        # `ind` must live on the same device as pd_scores and gt_labels.
        ind = torch.zeros([2, batch_size, num_max_boxes],
                          dtype=torch.long, device=device)
        ind[0] = torch.arange(end=batch_size, device=device) \
            .view(-1, 1).expand(-1, num_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long()
        bbox_scores[gt_mask] = pd_scores[ind[0], :, ind[1]][gt_mask]

        pd_boxes = pd_bboxes.unsqueeze(1) \
            .expand(-1, num_max_boxes, -1, -1)[gt_mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[gt_mask]
        overlaps[gt_mask] = compute_iou(
            gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Keep only the top-k anchors per ground truth.
        top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_k_metrics, top_k_indices = torch.topk(
            align_metric, self.top_k, dim=-1, largest=True)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        mask_top_k = torch.zeros(
            align_metric.shape, dtype=torch.int8, device=device)
        ones = torch.ones_like(
            top_k_indices[:, :, :1], dtype=torch.int8, device=device)
        for k in range(self.top_k):
            mask_top_k.scatter_add_(
                -1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k.masked_fill_(mask_top_k > 1, 0)
        mask_top_k = mask_top_k.to(align_metric.dtype)
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        # Solve conflicts: one anchor assigned to several ground truths.
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1) \
                .expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(
                mask_pos.shape, dtype=mask_pos.dtype, device=device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(
                mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)

        # Build the targets.
        batch_index = torch.arange(
            end=batch_size, dtype=torch.int64, device=device)[..., None]
        target_index = target_gt_idx + batch_index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]
        target_bboxes = gt_bboxes.view(
            -1, gt_bboxes.shape[-1])[target_index]
        target_labels.clamp_(0)

        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.nc),
            dtype=torch.int64, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize the class targets with the alignment metric.
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps /
                             (pos_align_metrics + self.eps)) \
            .amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()


# ---------------------------------------------------------------------------
# Box + DFL loss
# ---------------------------------------------------------------------------

class BoxLoss(nn.Module):
    """CIoU loss on the boxes plus the DFL loss on the distributions."""

    def __init__(self, dfl_ch):
        super().__init__()
        self.dfl_ch = dfl_ch  # reg_max - 1

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU (CIoU) loss.
        weight = torch.masked_select(
            target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss.
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.dfl_ch - 0.01)
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.dfl_ch + 1),
            target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        return loss_box, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        left_loss = cross_entropy(
            pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = cross_entropy(
            pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


# ---------------------------------------------------------------------------
# Full loss
# ---------------------------------------------------------------------------

class ComputeLoss:
    """Complete YOLOv8 loss (CIoU box + DFL + BCE classification)."""

    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device
        m = model.head

        # Safety check: the strides must be calibrated.
        if torch.all(m.stride == 0):
            raise RuntimeError(
                "model.head.stride is not set. Use YOLO (which sets it "
                "automatically) or call model._initialize_strides first.")

        self.params = params
        self.stride = m.stride.to(device)
        # Scalar stride of the first scale, cached once so the target
        # packing never has to touch a GPU tensor.
        self.stride0 = float(m.stride[0])
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)
        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        """Turn the DFL distributions into xyxy boxes (stride units)."""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def _build_gt(self, targets, batch_size, image_wh):
        """Pack the flat target dict into a (b, max_obj, 5) fp32 tensor.

        The packing runs on the device the targets already live on
        (usually the CPU, straight from the collate function) with pure
        tensor ops — no per-image Python loop, no GPU synchronization.
        The result is moved to the compute device in one transfer.

        Args:
            targets: {'idx': (N,), 'cls': (N,), 'box': (N, 4)} with
                boxes as normalized (cx, cy, w, h).
            batch_size: number of images in the batch.
            image_wh: (width, height) of the network input in pixels.
        """
        idx = targets['idx'].view(-1).long()
        cls = targets['cls'].view(-1, 1).float()
        box = targets['box'].float()  # (N, 4) normalized cx cy w h

        if idx.numel() == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device,
                               dtype=torch.float32)

        if int(cls.max()) >= self.nc:
            raise ValueError(
                f"Target class id {int(cls.max())} is out of range for "
                f"a {self.nc}-class model. Check the dataset labels "
                f"and data.yaml.")

        pack_device = idx.device
        rows = torch.cat((cls, box), dim=1)          # (N, 5)
        counts = torch.bincount(idx, minlength=batch_size)
        max_n = int(counts.max())

        # Row -> (image, slot) destination, computed without a loop:
        # after a stable sort by image index, the slot of a row is its
        # rank minus the offset of its image.
        order = torch.argsort(idx, stable=True)
        idx_sorted = idx[order]
        offsets = torch.zeros(batch_size, dtype=torch.long,
                              device=pack_device)
        offsets[1:] = counts.cumsum(0)[:-1]
        slots = torch.arange(idx.numel(), device=pack_device) \
            - offsets[idx_sorted]

        gt = torch.zeros(batch_size, max_n, 5, dtype=torch.float32,
                         device=pack_device)
        gt[idx_sorted, slots] = rows[order]

        # Normalized cx cy w h -> xyxy in image space.
        w, h = image_wh
        scale = torch.tensor([w, h, w, h], dtype=torch.float32,
                             device=pack_device)
        xywh = gt[..., 1:5] * scale
        half = xywh[..., 2:4] / 2
        gt[..., 1:3] = xywh[..., 0:2] - half
        gt[..., 3:5] = xywh[..., 0:2] + half
        return gt.to(self.device, non_blocking=True)

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: list of 3 tensors [bs, no, h, w] (train mode).
                     If it is a tuple (inference, raw), `raw` is used.
            targets: dict {'idx': (N,), 'cls': (N,), 'box': (N, 4)}
                     boxes in YOLO format (cx, cy, w, h) normalized [0, 1]
        """
        if isinstance(outputs, tuple):
            outputs = outputs[1]

        x = torch.cat(
            [i.view(outputs[0].shape[0], self.no, -1) for i in outputs],
            dim=2)
        pred_distri, pred_scores = x.split(
            split_size=(self.reg_max * 4, self.nc), dim=1)
        # The loss runs in float32 even under autocast: fp16 would
        # quantize image-space coordinates (0.5 px steps above 512) and
        # degrade the CIoU / DFL numerics. Autograd casts the gradients
        # back to the model dtype automatically.
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().float()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous().float()

        batch_size = pred_scores.shape[0]
        feat_h, feat_w = outputs[0].shape[2:]
        image_wh = (feat_w * self.stride0, feat_h * self.stride0)

        anchor_points, stride_tensor = make_anchors(
            outputs, self.stride, offset=0.5)
        anchor_points = anchor_points.float()
        stride_tensor = stride_tensor.float()

        gt = self._build_gt(targets, batch_size, image_wh)
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode the predictions and run the assigner.
        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        assigned = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask = assigned

        # clamp(min=1) on the tensor avoids a GPU sync (.item()) per step
        # while keeping the same result as max(x, 1).
        target_scores_sum = target_scores.sum().clamp(min=1)

        # Classification loss (BCE).
        loss_cls = self.cls_loss(
            pred_scores, target_scores.to(pred_scores.dtype)).sum() \
            / target_scores_sum

        # Box + DFL loss. Computed unconditionally: with an empty
        # foreground mask every masked select is empty and both sums
        # are zero, so no `fg_mask.sum()` GPU sync is needed per step.
        target_bboxes = target_bboxes / stride_tensor
        loss_box, loss_dfl = self.box_loss(
            pred_distri, pred_bboxes, anchor_points,
            target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss_box = loss_box * self.params['box']
        loss_cls = loss_cls * self.params['cls']
        loss_dfl = loss_dfl * self.params['dfl']

        return loss_box, loss_cls, loss_dfl

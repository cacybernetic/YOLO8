"""Non-Maximum Suppression on the raw model output."""

from time import time

import torch
import torchvision
from loguru import logger

from .boxes import wh2xy


def non_max_suppression(outputs, confidence_threshold=0.001,
                        iou_threshold=0.7, max_det=300, time_limit=None):
    """Per-class NMS. `outputs`: (bs, 4 + nc, n_anchors), raw eval output.

    Args:
        time_limit: time budget in seconds (None = no limit, default).
            WARNING: when the budget is over, the remaining images of the
            batch get zero predictions. Never enable it during an
            evaluation (mAP would be silently truncated). It is only
            useful for real time inference with a bounded latency.
    """
    max_wh = 7680
    max_nms = 30000

    bs = outputs.shape[0]
    nc = outputs.shape[1] - 4
    xc = outputs[:, 4:4 + nc].amax(1) > confidence_threshold

    start = time()
    output = [torch.zeros((0, 6), device=outputs.device)] * bs

    for index, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[index]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > confidence_threshold) \
                .nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > confidence_threshold]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # class offset trick
        boxes, scores = x[:, :4] + c, x[:, 4]
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)[:max_det]
        output[index] = x[indices]
        if time_limit is not None and (time() - start) > time_limit:
            logger.warning(
                f"NMS: time budget over ({time_limit:.2f}s), "
                f"{bs - index - 1} image(s) of the batch skipped")
            break
    return output

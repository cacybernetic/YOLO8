"""Exponential Moving Average of the model weights."""

import math
from copy import deepcopy

import torch


class ModelEMA:
    """Smoothed copy of the model weights (Ultralytics convention).

    Update rule: ema = d * ema + (1 - d) * model, where the decay `d`
    grows toward `decay` following d(u) = decay * (1 - exp(-u / tau)).
    Early updates follow the model closely, then the smoothing settles.
    The EMA weights are used for validation and for best.pt: they are
    much more stable than the raw weights (often +1 to +2 mAP).
    """

    def __init__(self, model, decay=0.9999, tau=2000.0, updates=0):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = int(updates)
        self._decay = float(decay)
        self._tau = float(tau)

    def decay(self):
        return self._decay * (1 - math.exp(-self.updates / self._tau))

    @torch.no_grad()
    def update(self, model):
        """Call after every optimizer.step()."""
        self.updates += 1
        d = self.decay()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                # Integer buffers (num_batches_tracked...): direct copy.
                v.copy_(msd[k])

    def load_state_dict(self, state_dict, updates=None):
        self.ema.load_state_dict(state_dict)
        if updates is not None:
            self.updates = int(updates)

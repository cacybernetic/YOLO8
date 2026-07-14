"""Learning rate schedulers with full YOLOv8 warmup.

The schedulers are stateless: the learning rate only depends on the
global step, so a training resume just needs the step counter.
"""

import math


class BaseLR:
    """Per-step scheduler with the full warmup.

    During warmup (Ultralytics convention):
      - the weight LR goes up linearly from 0 to max_lr;
      - the bias LR goes DOWN from warmup_bias_lr (0.1) to max_lr, so
        biases can move fast early without hurting the weights;
      - the momentum goes up from warmup_momentum (0.8) to momentum.

    Small dataset guard: the warmup is capped so it never consumes the
    whole step budget.
    """

    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs,
                 num_steps, momentum=0.937, warmup_momentum=0.8,
                 warmup_bias_lr=0.1):
        total_steps = max(int(round(total_epochs * num_steps)), 1)
        warmup_steps = int(max(warmup_epochs * num_steps, 100))
        self.warmup_steps = max(min(warmup_steps, total_steps - 1), 0)
        self.decay_steps = max(total_steps - self.warmup_steps, 1)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.momentum = momentum
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr

    def _decayed_lr(self, s):
        """LR after warmup; s inside [0, decay_steps]."""
        raise NotImplementedError

    def lr_at(self, global_step):
        """Post-warmup LR value at a step (used for logging)."""
        if global_step < self.warmup_steps:
            return self.max_lr * (global_step + 1) / self.warmup_steps
        s = min(global_step - self.warmup_steps, self.decay_steps)
        return self._decayed_lr(s)

    def step(self, global_step, optimizer):
        """Set the LR (and momentum) of every group for this step."""
        if global_step < self.warmup_steps:
            self._warmup_step(global_step, optimizer)
            return
        s = min(global_step - self.warmup_steps, self.decay_steps)
        lr = self._decayed_lr(s)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            if 'momentum' in pg:
                pg['momentum'] = self.momentum

    def _warmup_step(self, global_step, optimizer):
        f = (global_step + 1) / self.warmup_steps
        momentum = (self.warmup_momentum +
                    (self.momentum - self.warmup_momentum) * f)
        for pg in optimizer.param_groups:
            start_lr = self.warmup_bias_lr if pg.get('is_bias') else 0.0
            pg['lr'] = start_lr + (self.max_lr - start_lr) * f
            if 'momentum' in pg:
                pg['momentum'] = momentum


class CosineLR(BaseLR):
    def _decayed_lr(self, s):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * s / self.decay_steps))


class LinearLR(BaseLR):
    def _decayed_lr(self, s):
        return self.max_lr + (self.min_lr - self.max_lr) * \
            (s / self.decay_steps)


def build_scheduler(name, max_lr, min_lr, warmup_epochs, total_epochs,
                    num_steps, momentum=0.937, warmup_momentum=0.8,
                    warmup_bias_lr=0.1):
    name = name.lower()
    kwargs = dict(momentum=momentum, warmup_momentum=warmup_momentum,
                  warmup_bias_lr=warmup_bias_lr)
    if name == 'cosine':
        return CosineLR(max_lr, min_lr, warmup_epochs, total_epochs,
                        num_steps, **kwargs)
    if name == 'linear':
        return LinearLR(max_lr, min_lr, warmup_epochs, total_epochs,
                        num_steps, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")

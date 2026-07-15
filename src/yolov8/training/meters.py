"""Loss meters: running averages that can pause and resume."""

import torch

LOSS_KEYS = ('box', 'cls', 'dfl', 'total')


class LossMeters:
    """Weighted running sums of the four loss values.

    The sums live on the compute device, so update() does not force a
    GPU synchronization. The sync happens in averages() and
    state_dict(). The training loop calls averages() at every step to
    feed the progress bar, so it does sync once per step.
    """

    def __init__(self, device):
        self.device = device
        self.sums = torch.zeros(len(LOSS_KEYS), device=device)
        self.count = 0

    def update(self, loss_box, loss_cls, loss_dfl, loss_total,
               batch_size):
        """Add one batch of loss values, weighted by the batch size."""
        values = torch.stack((
            loss_box.detach(), loss_cls.detach(),
            loss_dfl.detach(), loss_total.detach()))
        self.sums += values * batch_size
        self.count += batch_size

    def averages(self):
        """Return {'box': ..., 'cls': ..., 'dfl': ..., 'total': ...}."""
        n = max(self.count, 1)
        values = (self.sums / n).tolist()
        return dict(zip(LOSS_KEYS, values))

    def state_dict(self):
        return {'sums': self.sums.detach().cpu().tolist(),
                'count': self.count}

    def load_state_dict(self, state):
        self.sums = torch.tensor(
            state['sums'], dtype=torch.float32, device=self.device)
        self.count = int(state['count'])

    def reset(self):
        self.sums.zero_()
        self.count = 0

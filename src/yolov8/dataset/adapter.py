"""Resumable DataLoader adapter for fault tolerant training.

The adapter is an nn.Module that wraps a torch DataLoader. Its state
(epoch number, seed, and the count of batches already consumed in the
current epoch) lives in buffers, so `state_dict()` and
`load_state_dict()` work out of the box and the state can be stored in
a training checkpoint.

The shuffle order only depends on (seed + epoch), so it can be rebuilt
after a restart. On resume, iteration skips the batches that were
already consumed: every sample is still seen exactly once per epoch.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DataLoaderAdapter(nn.Module):
    """DataLoader wrapper with a resumable position.

    Usage:
        adapter = DataLoaderAdapter(dataset, batch_size=16, shuffle=True)
        for batch in adapter:   # position advances batch by batch
            ...
        # After the loop the epoch counter advances and position resets.

    A checkpoint can store `adapter.state_dict()` at any time. After
    `load_state_dict()`, the next iteration continues from the saved
    position of the saved epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=0, collate_fn=None, pin_memory=False,
                 drop_last=False, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = int(num_workers)
        self.collate_fn = collate_fn
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.register_buffer(
            '_epoch', torch.zeros((), dtype=torch.long))
        self.register_buffer(
            '_position', torch.zeros((), dtype=torch.long))
        self.register_buffer(
            '_seed', torch.tensor(int(seed), dtype=torch.long))

    @property
    def epoch(self):
        return int(self._epoch)

    @property
    def position(self):
        """Number of batches already consumed in the current epoch."""
        return int(self._position)

    def _build_order(self):
        """Sample order for the current epoch (depends on seed+epoch)."""
        n = len(self.dataset)
        if not self.shuffle:
            return list(range(n))
        g = torch.Generator()
        g.manual_seed(int(self._seed) + int(self._epoch))
        return torch.randperm(n, generator=g).tolist()

    def _batches(self, order):
        """Cut the epoch order into batch index lists."""
        bs = self.batch_size
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        if self.drop_last and chunks and len(chunks[-1]) < bs:
            chunks.pop()
        return chunks

    def __len__(self):
        """Number of batches in one full epoch."""
        return len(self._batches(self._build_order()))

    def remaining_batches(self):
        """Number of batches left in the current epoch."""
        return max(len(self) - self.position, 0)

    def __iter__(self):
        """Yield the remaining batches of the current epoch.

        The position buffer is updated before each yield, so a
        checkpoint taken while a batch is processed counts that batch
        as consumed. When the epoch completes, the epoch counter
        advances and the position resets to zero.
        """
        chunks = self._batches(self._build_order())
        pending = chunks[self.position:]
        loader = DataLoader(
            self.dataset, batch_sampler=pending,
            num_workers=self.num_workers, collate_fn=self.collate_fn,
            pin_memory=self.pin_memory)
        for batch in loader:
            self._position += 1
            yield batch
        self._epoch += 1
        self._position.zero_()

    def start_new_epoch(self):
        """Force the start of a fresh epoch (drop a partial position)."""
        self._position.zero_()

    def forward(self):
        """nn.Module interface; the adapter is not a network layer."""
        raise RuntimeError(
            "DataLoaderAdapter is not callable; iterate over it.")

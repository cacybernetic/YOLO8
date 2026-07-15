"""Resumable DataLoader adapter for fault tolerant training.

The adapter wraps a torch DataLoader. Its state (epoch number, seed,
and the count of batches already consumed in the current epoch) is
exposed through `state_dict()` / `load_state_dict()` so it can be
stored in a training checkpoint.

The shuffle order only depends on (seed + epoch), so it can be rebuilt
after a restart. On resume, iteration skips the batches that were
already consumed: every sample is still seen exactly once per epoch.

With `persistent=True` and `num_workers > 0`, one DataLoader (and its
worker processes) is kept alive across epochs instead of being
re-spawned every epoch. The per-epoch batch order is produced by a
small dynamic batch sampler that reads the adapter state at the start
of each iteration. Call `invalidate_workers()` after mutating the
dataset in the main process (for example when `close_mosaic` turns the
mosaic off): persistent workers hold their own dataset copy and must
be restarted to see the change.
"""

import torch
from torch.utils.data import DataLoader


class _PendingBatchSampler:
    """Batch sampler that yields the REMAINING batches of the epoch.

    It reads the adapter's epoch and position lazily at the start of
    each iteration, so one DataLoader object (with persistent workers)
    can serve every epoch and every mid-epoch resume.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def __iter__(self):
        return iter(self.adapter._pending_batches())

    def __len__(self):
        return self.adapter.remaining_batches()


class DataLoaderAdapter:
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
                 drop_last=False, seed=0, persistent=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = int(num_workers)
        self.collate_fn = collate_fn
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.persistent = bool(persistent) and self.num_workers > 0
        self._epoch = 0
        self._position = 0
        self._seed = int(seed)
        self._dl = None

    @property
    def epoch(self):
        return self._epoch

    @property
    def position(self):
        """Number of batches already consumed in the current epoch."""
        return self._position

    def _build_order(self):
        """Sample order for the current epoch (depends on seed+epoch)."""
        n = len(self.dataset)
        if not self.shuffle:
            return list(range(n))
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        return torch.randperm(n, generator=g).tolist()

    def _batches(self, order):
        """Cut the epoch order into batch index lists."""
        bs = self.batch_size
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        if self.drop_last and chunks and len(chunks[-1]) < bs:
            chunks.pop()
        return chunks

    def _pending_batches(self):
        """Remaining batch index lists of the current epoch."""
        return self._batches(self._build_order())[self._position:]

    def __len__(self):
        """Number of batches in one full epoch."""
        return len(self._batches(self._build_order()))

    def remaining_batches(self):
        """Number of batches left in the current epoch."""
        return max(len(self) - self.position, 0)

    def _loader(self):
        """The wrapped DataLoader (cached when workers are persistent)."""
        if self._dl is not None:
            return self._dl
        kwargs = dict(num_workers=self.num_workers,
                      collate_fn=self.collate_fn,
                      pin_memory=self.pin_memory)
        if self.persistent:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(self.dataset,
                            batch_sampler=_PendingBatchSampler(self),
                            **kwargs)
        if self.persistent:
            self._dl = loader
        return loader

    def invalidate_workers(self):
        """Drop the cached DataLoader so workers restart on next iter.

        Needed after any main-process mutation of the dataset (for
        example `close_mosaic`): persistent workers keep their own
        copy of the dataset and would otherwise never see it.
        """
        self._dl = None

    def __iter__(self):
        """Yield the remaining batches of the current epoch.

        The position advances before each yield, so a checkpoint taken
        while a batch is processed counts that batch as consumed. When
        the epoch completes, the epoch counter advances and the
        position resets to zero.
        """
        for batch in self._loader():
            self._position += 1
            yield batch
        self._epoch += 1
        self._position = 0

    def start_new_epoch(self):
        """Force the start of a fresh epoch (drop a partial position)."""
        self._position = 0

    def state_dict(self):
        """Checkpoint-friendly state (same keys as the old versions)."""
        return {'_epoch': self._epoch, '_position': self._position,
                '_seed': self._seed}

    def load_state_dict(self, state):
        """Restore a state saved by this class or by older versions
        (which stored 0-d tensors instead of plain ints)."""
        self._epoch = self._to_int(state['_epoch'])
        self._position = self._to_int(state['_position'])
        self._seed = self._to_int(state['_seed'])

    @staticmethod
    def _to_int(value):
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

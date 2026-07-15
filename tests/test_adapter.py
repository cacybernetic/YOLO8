"""DataLoaderAdapter tests: one pass per sample, mid-epoch resume."""

import torch
from torch.utils.data import Dataset

from yolov8.dataset import DataLoaderAdapter


class IndexDataset(Dataset):
    """Returns its own index; makes coverage checks trivial."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor(idx)


def collect(batches):
    out = []
    for b in batches:
        out.extend(int(v) for v in b)
    return out


def test_every_sample_seen_once_per_epoch():
    adapter = DataLoaderAdapter(IndexDataset(10), batch_size=3,
                                shuffle=True, seed=0)
    seen = collect(iter(adapter))
    assert sorted(seen) == list(range(10))
    assert adapter.epoch == 1
    assert adapter.position == 0


def test_shuffle_changes_between_epochs_but_is_deterministic():
    a1 = DataLoaderAdapter(IndexDataset(20), batch_size=5,
                           shuffle=True, seed=7)
    epoch0 = collect(iter(a1))
    epoch1 = collect(iter(a1))
    assert epoch0 != epoch1  # a new order every epoch

    a2 = DataLoaderAdapter(IndexDataset(20), batch_size=5,
                           shuffle=True, seed=7)
    assert collect(iter(a2)) == epoch0  # same seed -> same order


def test_resume_mid_epoch_no_duplicates():
    adapter = DataLoaderAdapter(IndexDataset(12), batch_size=3,
                                shuffle=True, seed=3)
    it = iter(adapter)
    first = collect([next(it), next(it)])  # 2 of 4 batches
    state = adapter.state_dict()

    # A new adapter (fresh process) resumes from the saved state.
    resumed = DataLoaderAdapter(IndexDataset(12), batch_size=3,
                                shuffle=True, seed=999)
    resumed.load_state_dict(state)
    assert resumed.position == 2
    rest = collect(iter(resumed))

    assert sorted(first + rest) == list(range(12))
    assert len(first + rest) == 12  # nothing seen twice


def test_drop_last():
    adapter = DataLoaderAdapter(IndexDataset(10), batch_size=4,
                                shuffle=False, drop_last=True)
    assert len(adapter) == 2
    seen = collect(iter(adapter))
    assert len(seen) == 8


def test_state_dict_contains_buffers():
    adapter = DataLoaderAdapter(IndexDataset(4), batch_size=2)
    state = adapter.state_dict()
    assert set(state) == {'_epoch', '_position', '_seed'}


def test_load_state_dict_accepts_old_tensor_format():
    """Checkpoints from the nn.Module-based adapter stored 0-d tensors."""
    adapter = DataLoaderAdapter(IndexDataset(6), batch_size=2)
    adapter.load_state_dict({
        '_epoch': torch.tensor(3), '_position': torch.tensor(1),
        '_seed': torch.tensor(42)})
    assert adapter.epoch == 3
    assert adapter.position == 1


def test_persistent_workers_cover_every_epoch():
    adapter = DataLoaderAdapter(IndexDataset(10), batch_size=3,
                                shuffle=True, num_workers=2, seed=0,
                                persistent=True)
    for _ in range(2):
        seen = collect(iter(adapter))
        assert sorted(seen) == list(range(10))
    assert adapter._dl is not None      # loader kept alive
    adapter.invalidate_workers()
    assert adapter._dl is None
    seen = collect(iter(adapter))       # still works after restart
    assert sorted(seen) == list(range(10))

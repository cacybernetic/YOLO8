"""Build and read HDF5 files with ready-to-train samples.

The HDF5 file stores samples that are already letterboxed (and possibly
augmented). Training with `use_hdf5: true` then skips most of the image
work and reads tensors straight from the file.

Layout of the file:
    images:        (N, S, S, 3) uint8, BGR
    labels:        (M, 5) float32, all boxes of all samples
    label_offsets: (N + 1,) int64; boxes of sample i are
                   labels[label_offsets[i]:label_offsets[i + 1]]
    attrs: image_size (int), names (JSON list or empty string)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm

from .transforms import image_to_tensor


def _open_h5(path, mode='r'):
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 datasets. "
            "Install it with: pip install h5py") from e
    return h5py.File(path, mode)


class Hdf5Builder:
    """Write samples into a growing HDF5 file."""

    def __init__(self, output_path, image_size, names=None):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self._file = _open_h5(self.path, 'w')
        s = image_size
        self._images = self._file.create_dataset(
            'images', shape=(0, s, s, 3), maxshape=(None, s, s, 3),
            dtype='uint8', chunks=(1, s, s, 3))
        self._labels = self._file.create_dataset(
            'labels', shape=(0, 5), maxshape=(None, 5), dtype='float32')
        self._offsets = [0]
        self._file.attrs['image_size'] = image_size
        self._file.attrs['names'] = json.dumps(names) if names else ''

    def add(self, img, labels):
        """Append one sample. img: (S, S, 3) uint8, labels: (n, 5)."""
        n = self._images.shape[0]
        self._images.resize(n + 1, axis=0)
        self._images[n] = img
        labels = np.asarray(labels, dtype=np.float32).reshape(-1, 5)
        m = self._labels.shape[0]
        self._labels.resize(m + labels.shape[0], axis=0)
        if labels.shape[0]:
            self._labels[m:] = labels
        self._offsets.append(m + labels.shape[0])

    def close(self):
        self._file.create_dataset(
            'label_offsets',
            data=np.asarray(self._offsets, dtype=np.int64))
        self._file.close()
        size_mb = self.path.stat().st_size / 1e6
        logger.info(f"HDF5 file written: {self.path} ({size_mb:.1f} MB)")


def build_hdf5(dataset, output_path, augmented_copies=0, verbose=True):
    """Build an HDF5 file from a YoloDataset.

    Base samples are stored without augmentation. When
    `augmented_copies` > 0, that many augmented copies of each sample
    are added too (the dataset must have augment enabled for them).

    Args:
        dataset: a YoloDataset (augment flag drives the extra copies).
        output_path: destination .h5 file.
        augmented_copies: number of augmented copies per sample.
    """
    builder = Hdf5Builder(output_path, dataset.image_size,
                          names=dataset.names)
    n = len(dataset)
    iterator = tqdm(range(n), desc="building hdf5", disable=not verbose,
                    leave=False, dynamic_ncols=True, ascii="░█")
    for idx in iterator:
        result = dataset._load_sample(idx)
        if result is None:
            logger.warning(f"Sample {idx} skipped (unreadable image)")
            continue
        img, labels, _ = result
        builder.add(img, labels)
        for _ in range(max(int(augmented_copies), 0)):
            aug_img, aug_labels = _augmented_copy(dataset, idx)
            if aug_img is not None:
                builder.add(aug_img, aug_labels)
    builder.close()
    return output_path


def _augmented_copy(dataset, idx):
    """Build one augmented copy of a sample. Returns (img, labels)."""
    use_mosaic = dataset.augmenter.use_mosaic()
    if use_mosaic:
        img, labels = dataset._load_mosaic(idx)
    else:
        result = dataset._load_sample(idx)
        if result is None:
            return None, None
        img, labels, _ = result
    img, labels = dataset.augmenter(
        img, labels, dataset._mixup_loader, mosaic_used=use_mosaic)
    return img, labels


class Hdf5Dataset(Dataset):
    """Read ready-to-train samples from an HDF5 file.

    The file handle is opened lazily and re-opened after a fork, so the
    dataset is safe to use inside DataLoader worker processes.
    """

    def __init__(self, path, max_samples=None, seed=0):
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")
        self._handle = None
        self._pid = None

        with _open_h5(self.path, 'r') as f:
            self.image_size = int(f.attrs['image_size'])
            raw_names = f.attrs.get('names', '')
            self.names = json.loads(raw_names) if raw_names else None
            self._offsets = np.asarray(f['label_offsets'])
            total = f['images'].shape[0]

        self._indices = list(range(total))
        if max_samples is not None and max_samples < total:
            import random as _random
            rng = _random.Random(seed)
            picked = rng.sample(range(total), max_samples)
            self._indices = sorted(picked)

    def _h5(self):
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = _open_h5(self.path, 'r')
            self._pid = pid
        return self._handle

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real = self._indices[idx]
        f = self._h5()
        img = np.asarray(f['images'][real])
        lo, hi = int(self._offsets[real]), int(self._offsets[real + 1])
        labels = np.asarray(f['labels'][lo:hi], dtype=np.float32)
        labels = labels.reshape(-1, 5)
        img = image_to_tensor(img)
        return img, torch.from_numpy(labels).float(), f"h5:{real}"

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_handle'] = None
        state['_pid'] = None
        return state

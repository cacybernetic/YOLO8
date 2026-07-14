"""Checkpoint files: naming, atomic writes, rotation, RNG capture.

Naming pattern (one file per save point, never merged):
    checkpoint_e0001c0012.pth
where e0001 is the 1-indexed epoch and c0012 is the number of
optimizer steps (train phase) or batches (eval phases) done inside
that epoch when the checkpoint was written.
"""

import random
import re
from pathlib import Path

import numpy as np
import torch
from loguru import logger

CKPT_PATTERN = re.compile(r'^checkpoint_e(\d+)c(\d+)\.pth$')


def checkpoint_name(epoch, step):
    """File name for a checkpoint (epoch is 1-indexed here)."""
    return f"checkpoint_e{epoch:04d}c{step:04d}.pth"


def parse_checkpoint_name(path):
    """Return (epoch, step) parsed from a file name, or None."""
    m = CKPT_PATTERN.match(Path(path).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def atomic_save(state, path):
    """Save through a temporary file then rename (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


class CheckpointManager:
    """Save, list, rotate and find checkpoints inside one folder."""

    def __init__(self, directory, max_keep=5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_keep = int(max_keep)

    def list(self):
        """Checkpoints sorted by (epoch, step), oldest first."""
        found = []
        for p in self.directory.glob('checkpoint_e*.pth'):
            parsed = parse_checkpoint_name(p)
            if parsed is not None:
                found.append((parsed, p))
        return [p for _, p in sorted(found)]

    def latest(self):
        """Most recent checkpoint path, or None."""
        ckpts = self.list()
        return ckpts[-1] if ckpts else None

    def save(self, state, epoch, step):
        """Write one checkpoint and rotate the old ones."""
        path = self.directory / checkpoint_name(epoch, step)
        atomic_save(state, path)
        logger.info(f"Checkpoint saved: {path.name}")
        self.rotate()
        return path

    def rotate(self):
        """Delete the oldest files above max_keep."""
        ckpts = self.list()
        if len(ckpts) <= self.max_keep:
            return
        for p in ckpts[:len(ckpts) - self.max_keep]:
            try:
                p.unlink()
                logger.info(f"Checkpoint removed (rotation): {p.name}")
            except OSError as e:
                logger.warning(f"Cannot remove {p}: {e}")


def capture_rng_state():
    """Snapshot every RNG state in a checkpoint-friendly form."""
    np_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': [np_state[0], np.asarray(np_state[1]).tolist(),
                  int(np_state[2]), int(np_state[3]),
                  float(np_state[4])],
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore RNG states saved by capture_rng_state."""
    if not state:
        return
    try:
        py_state = state.get('python')
        if py_state is not None:
            random.setstate(_to_tuple(py_state))
        np_state = state.get('numpy')
        if np_state is not None:
            np.random.set_state((
                np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                int(np_state[2]), int(np_state[3]), float(np_state[4])))
        if state.get('torch') is not None:
            torch.set_rng_state(
                state['torch'].to(torch.uint8).cpu())
        if state.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [t.to(torch.uint8).cpu() for t in state['cuda']])
    except Exception as e:
        logger.warning(f"Cannot restore RNG state ({e}); "
                       f"training goes on with fresh randomness.")


def _to_tuple(value):
    """Recursively turn lists back into tuples (pickle round trip)."""
    if isinstance(value, (list, tuple)):
        return tuple(_to_tuple(v) for v in value)
    return value

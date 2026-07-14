"""Run folders: runs/<run_name>/train, train2, ... and eval, eval2, ...

Rules (from the project spec):
  - the first run folder has no number: `train`, not `train1`;
  - the next ones are `train2`, `train3`, and so on;
  - with resume enabled, the highest numbered folder that holds a
    usable checkpoint is reused instead of creating a new folder.
"""

import re
from pathlib import Path

import yaml
from loguru import logger

SUBFOLDERS = ('weights', 'checkpoints', 'plotes', 'logs')


def _dir_number(name, kind):
    """`train` -> 1, `train2` -> 2 ... None when the name mismatches."""
    if name == kind:
        return 1
    m = re.fullmatch(rf'{re.escape(kind)}(\d+)', name)
    if m and int(m.group(1)) >= 2:
        return int(m.group(1))
    return None


def list_run_dirs(base, kind):
    """Existing run folders of a kind, sorted by number ascending."""
    base = Path(base)
    if not base.is_dir():
        return []
    found = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        num = _dir_number(p.name, kind)
        if num is not None:
            found.append((num, p))
    return sorted(found)


def has_checkpoint(run_dir):
    """True when the run folder holds at least one checkpoint file."""
    ckpt_dir = Path(run_dir) / 'checkpoints'
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob('checkpoint_e*.pth'))


def _new_run_dir(base, kind, existing):
    if not existing:
        return Path(base) / kind
    next_num = existing[-1][0] + 1
    return Path(base) / f"{kind}{next_num}"


def prepare_run_dir(output_dir, run_name, kind='train', resume=False):
    """Create (or reuse) the run folder and its sub folders.

    Args:
        output_dir: base folder, usually 'runs'.
        run_name: the run name from the config file.
        kind: 'train' or 'eval'.
        resume: reuse the highest numbered folder with a checkpoint.

    Returns:
        (run_dir, resumed): the folder path and True when an existing
        folder with a checkpoint was picked for resume.
    """
    base = Path(output_dir) / run_name
    existing = list_run_dirs(base, kind)

    if resume:
        for num, path in reversed(existing):
            if has_checkpoint(path):
                logger.info(f"Resume: reusing run folder {path}")
                _make_subfolders(path)
                return path, True
        logger.info("Resume requested but no checkpoint found; "
                    "starting a new run folder.")

    run_dir = _new_run_dir(base, kind, existing)
    _make_subfolders(run_dir)
    logger.info(f"Run folder: {run_dir}")
    return run_dir, False


def _make_subfolders(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in SUBFOLDERS:
        (run_dir / sub).mkdir(exist_ok=True)


def save_config_used(run_dir, raw_config):
    """Write the config used by this run as config_used.yaml."""
    path = Path(run_dir) / 'config_used.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(raw_config, f, sort_keys=False)
    logger.info(f"Config snapshot written: {path}")

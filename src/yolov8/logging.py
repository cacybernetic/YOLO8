"""Loguru setup and model summary logging for the whole project.

The console sink writes through tqdm, so log lines never break the
progress bars. A file sink stores every run log inside the run folder,
for example `runs/<name>/train/logs/train_2026-06-06_13-42-11.log`.
"""

import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

CONSOLE_FORMAT = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
                  "<level>{level: <7}</level> | {message}")
FILE_FORMAT = ("{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | "
               "{name}:{function}:{line} | {message}")


def _tqdm_sink(message):
    """Write a log line without breaking active tqdm bars."""
    tqdm.write(str(message), end='')


def setup_logging(level="INFO", log_dir=None, prefix="run"):
    """Configure loguru for the current program.

    Args:
        level: minimum level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_dir: folder for the log file. None disables the file sink.
        prefix: file name prefix, for example 'train' or 'eval'.

    Returns:
        Path of the log file, or None when log_dir is None.
    """
    logger.remove()
    logger.add(_tqdm_sink, format=CONSOLE_FORMAT, level=level,
               colorize=True, enqueue=False)

    if log_dir is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f"{prefix}_{stamp}.log"
    logger.add(str(log_file), level=level, format=FILE_FORMAT,
               enqueue=True, backtrace=True, diagnose=False)
    logger.info(f"Logging to file: {log_file}")
    return log_file


def add_file_logging(log_dir, prefix="run", level="INFO"):
    """Add a file sink to the current logger (console sink kept).

    Used when the log folder (inside the run folder) is only known
    after the console logging is already set up.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f"{prefix}_{stamp}.log"
    logger.add(str(log_file), level=level, format=FILE_FORMAT,
               enqueue=True, backtrace=True, diagnose=False)
    logger.info(f"Logging to file: {log_file}")
    return log_file


def log_dict(data, indent=2):
    """Log a nested dict, one line per key (used for config dumps)."""
    pad = ' ' * indent
    for key, value in data.items():
        if isinstance(value, dict):
            logger.info(f"{pad}{key}:")
            log_dict(value, indent=indent + 2)
        else:
            logger.info(f"{pad}{key}: {value}")


def log_model_summary(model: nn.Module, input_size=(1, 3, 640, 640),
                      device=None, depth=4):
    """Log a full torchinfo summary of the model, line by line.

    Falls back to a small native summary when torchinfo is missing or
    fails (the YOLO head returns a tuple in eval mode, which some
    torchinfo versions do not accept).
    """
    if device is None:
        device = next(model.parameters()).device
    try:
        from torchinfo import summary
    except ImportError:
        logger.warning("torchinfo is not installed "
                       "(pip install torchinfo)")
        _log_native_summary(model)
        return

    # Save the train/eval state of every submodule before switching to
    # eval for the summary forward. A plain model.train() afterwards
    # would wake up layers frozen on purpose (BatchNorm freeze).
    states = {name: m.training for name, m in model.named_modules()}
    model.eval()
    try:
        stats = summary(
            model, input_size=input_size, device=device, depth=depth,
            verbose=0,
            col_names=("input_size", "output_size", "num_params",
                       "trainable"),
            row_settings=("var_names",))
        for line in str(stats).splitlines():
            logger.info(f"  {line}")
    except Exception as e:
        logger.warning(f"torchinfo failed ({e.__class__.__name__}): {e}")
        _log_native_summary(model)
    finally:
        for name, m in model.named_modules():
            if name in states:
                m.training = states[name]


def _log_native_summary(model: nn.Module):
    """Small summary without torchinfo: parameter counts only."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total:>12,} ({total / 1e6:.3f} M)")
    logger.info(f"  Trainable params: {trainable:>12,} "
                f"({trainable / 1e6:.3f} M)")
    logger.info(f"  Non-trainable:    {total - trainable:>12,}")


def safe_torch_load(path, map_location='cpu', allow_pickle=False):
    """Load a checkpoint with safe deserialization by default.

    `torch.load(weights_only=False)` runs arbitrary pickle code, which
    is a code execution risk on untrusted files. The safe mode
    (`weights_only=True`) covers every checkpoint this project writes.

    An automatic fallback to full pickle would defeat the protection
    (a malicious file only has to make the safe load fail), so the
    fallback is opt-in: pass `allow_pickle=True` ONLY for files you
    fully trust.
    """
    try:
        return torch.load(path, map_location=map_location,
                          weights_only=True)
    except TypeError:
        # Old PyTorch without the weights_only argument.
        return torch.load(path, map_location=map_location)
    except Exception as e:
        if not allow_pickle:
            raise RuntimeError(
                f"Safe load (weights_only=True) failed for '{path}'. "
                f"This file needs full pickle deserialization, which "
                f"can execute arbitrary code. If you fully trust it, "
                f"load it with allow_pickle=True.\n  Cause: {e}"
            ) from e
        logger.warning(
            f"'{path}': safe load failed, falling back to full pickle "
            f"(allow_pickle=True). Only load trusted files.")
        return torch.load(path, map_location=map_location,
                          weights_only=False)

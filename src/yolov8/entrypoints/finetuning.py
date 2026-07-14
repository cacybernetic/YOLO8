"""Build a fine-tunable YOLOv8 checkpoint from pretrained weights.

Usage:
    ylft --config gpu/configs/finetuning.yaml

Why only the classification branches are re-initialized:
  - Backbone and neck extract generic features; they transfer well
    between domains, so they are kept as they are.
  - The box branches and the DFL predict pure geometry, independent
    of the class count; kept too.
  - Only the `cls` branches depend on num_classes. Their output size
    changes, so they restart from a fresh initialization with the
    Focal Loss bias prior b = -log((1 - pi) / pi), pi = 0.01
    (Lin et al., 2017). This keeps the first steps stable.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from yolov8.config import FinetuneConfig, load_finetune_config
from yolov8.devices import resolve_device
from yolov8.logging import (setup_logging, log_model_summary,
                            safe_torch_load)
from yolov8.model import MyYolo
from yolov8.training.checkpoints import atomic_save


def load_source_state(cfg: FinetuneConfig, device):
    """Load the state_dict of the pretrained model."""
    src_path = Path(cfg.pretrained_weights)
    if not src_path.exists():
        raise FileNotFoundError(f"Source weights not found: {src_path}")
    ckpt = safe_torch_load(src_path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    logger.info(f"Source loaded: {src_path} ({len(state)} tensors)")
    return state


def init_cls_head_bias(module: nn.Module, num_classes,
                       prior: float = 0.01):
    """Set the bias of the last Conv2d of one `cls` branch.

    b = -log((1 - prior) / prior), so the initial sigmoid output is
    exactly `prior`. The weights keep their default init.
    """
    final_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == num_classes:
            final_conv = m
    if final_conv is None or final_conv.bias is None:
        return
    bias_init = -math.log((1.0 - prior) / prior)
    with torch.no_grad():
        final_conv.bias.fill_(bias_init)


def transfer_weights(src_state, new_model, old_num_classes,
                     new_num_classes, strict_backbone=True):
    """Copy every shape-compatible tensor from source to target.

    Backbone and neck must load fully (a mismatch there almost always
    means a wrong `version`). Head tensors whose shape differs (the
    cls branches) keep their fresh initialization.

    Returns:
        (loaded, skipped): lists of tensor names copied / skipped.
    """
    new_state = new_model.state_dict()
    loaded, skipped, missing = [], [], []

    for key, new_tensor in new_state.items():
        if key not in src_state:
            missing.append(key)
            continue
        src_tensor = src_state[key]
        if src_tensor.shape == new_tensor.shape:
            new_state[key] = src_tensor.clone()
            loaded.append(key)
        else:
            skipped.append((key, tuple(src_tensor.shape),
                            tuple(new_tensor.shape)))

    if strict_backbone:
        _check_feature_layers(new_state, loaded, missing)

    new_model.load_state_dict(new_state)
    _log_transfer(len(new_state), loaded, skipped,
                  old_num_classes, new_num_classes)
    return loaded, skipped


def _check_feature_layers(new_state, loaded, missing):
    """Fail when backbone/neck tensors were not transferred."""
    not_loaded = [
        k for k in new_state
        if k.startswith(('backbone.', 'neck.'))
        and k not in loaded and k not in missing
        and not k.endswith('num_batches_tracked')]
    if not_loaded:
        raise RuntimeError(
            f"Transfer failed for {len(not_loaded)} backbone/neck "
            f"tensors. Check that `version` matches the source model."
            f"\nExamples: {not_loaded[:3]}")


def _log_transfer(total, loaded, skipped, old_nc, new_nc):
    logger.info(f"Transfer: {len(loaded)}/{total} tensors copied "
                f"from the source model")
    if skipped:
        logger.info(f"{len(skipped)} tensors re-initialized (shape "
                    f"mismatch, expected for cls heads when "
                    f"{old_nc} -> {new_nc} classes):")
        for key, old_shape, new_shape in skipped[:6]:
            logger.info(f"    {key}: {old_shape} -> {new_shape}")
        if len(skipped) > 6:
            logger.info(f"    ... and {len(skipped) - 6} more")


def save_finetune_checkpoint(model, cfg: FinetuneConfig, output_path):
    """Save with the standard checkpoint layout ('model' key)."""
    output_path = Path(output_path)
    state = {
        'model': model.state_dict(),
        'epoch': -1,
        'best_metric': -float('inf'),
        'finetune_origin': cfg.pretrained_weights,
        'finetune_old_num_classes': cfg.old_num_classes,
        'finetune_new_num_classes': cfg.new_num_classes,
    }
    atomic_save(state, output_path)
    size_mb = output_path.stat().st_size / 1e6
    logger.success(f"Fine-tunable model written: {output_path} "
                   f"({size_mb:.2f} MB)")


def run_finetune_build(cfg: FinetuneConfig):
    device = resolve_device(cfg.device)
    logger.info(f"device={device}")
    logger.info(f"version={cfg.version} | old_nc={cfg.old_num_classes} "
                f"-> new_nc={cfg.new_num_classes}")

    new_model = MyYolo(version=cfg.version,
                       num_classes=cfg.new_num_classes,
                       input_size=cfg.image_size).to(device)
    src_state = load_source_state(cfg, device)
    transfer_weights(src_state, new_model,
                     old_num_classes=cfg.old_num_classes,
                     new_num_classes=cfg.new_num_classes,
                     strict_backbone=cfg.strict_backbone_load)

    for cls_branch in new_model.head.cls:
        init_cls_head_bias(cls_branch, cfg.new_num_classes,
                           prior=cfg.cls_prior)
    bias_value = -math.log((1.0 - cfg.cls_prior) / cfg.cls_prior)
    logger.info(f"cls branch bias set to b = -log((1-pi)/pi) = "
                f"{bias_value:.4f} (pi={cfg.cls_prior})")

    logger.info("===== fine-tunable model architecture =====")
    log_model_summary(
        new_model, input_size=(1, 3, cfg.image_size, cfg.image_size),
        device=device)

    output_path = Path(cfg.output_weights)
    save_finetune_checkpoint(new_model, cfg, output_path)

    logger.success("Model ready for fine tuning.")
    logger.info("Next steps in train.yaml:")
    logger.info(f"  model.pretrained_weights: {output_path}")
    logger.info(f"  model.version: {cfg.version}")
    logger.info("  optional: model.freeze_feature_layers: true")
    return new_model, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build a fine-tunable YOLOv8 checkpoint "
                    "(new class count).")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to finetuning.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    cfg = load_finetune_config(args.config)
    run_finetune_build(cfg)


if __name__ == '__main__':
    main()

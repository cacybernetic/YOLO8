"""Training entrypoint.

Usage:
    yltrain --config gpu/configs/train.yaml

The program runs three passes per the project spec: training,
per-epoch validation on a fraction of the test split, and a final
evaluation on the held-out remainder of the test split (disjoint from
the validation part). Outputs go to runs/<run_name>/train[i]/
(weights, checkpoints, plots, logs, history.csv, config_used.yaml).
"""

import argparse
import random

import numpy as np
import torch
from loguru import logger

from yolov8.config import (load_train_config, config_to_dict,
                           TrainConfig)
from yolov8.dataset import DataLoaderAdapter, collate_detection_batch
from yolov8.dataset.factory import (build_train_dataset,
                                    build_test_dataset,
                                    resolve_class_names,
                                    split_val_from_test)
from yolov8.devices import resolve_device
from yolov8.logging import (setup_logging, add_file_logging,
                            log_model_summary, log_dict,
                            safe_torch_load)
from yolov8.lossfn import ComputeLoss
from yolov8.model import YOLO
from yolov8.training import (Trainer, ModelEMA, build_optimizer,
                             build_scheduler, freeze_feature_layers,
                             prepare_run_dir, save_config_used,
                             CheckpointManager)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: TrainConfig, device):
    """Build the three resumable loader adapters and the class names."""
    train_ds = build_train_dataset(cfg.dataset, seed=cfg.seed)
    test_ds = build_test_dataset(cfg.dataset, seed=cfg.seed)
    names = resolve_class_names(cfg.dataset, train_ds, test_ds)
    # val and final test are DISJOINT parts of the test split, so the
    # model selection never sees the samples of the final evaluation.
    val_ds, final_test_ds = split_val_from_test(
        test_ds, cfg.dataset.val_prob, seed=cfg.seed)
    logger.info(f"Data: train={len(train_ds)} | val={len(val_ds)} | "
                f"final test={len(final_test_ds)} | "
                f"classes={len(names)}")

    opt = cfg.optimization
    pin = device.type == 'cuda'
    # Persistent workers: keep the train workers alive across epochs
    # instead of re-spawning them every epoch. The trainer restarts
    # them explicitly when close_mosaic mutates the augmentations.
    train_loader = DataLoaderAdapter(
        train_ds, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers,
        collate_fn=collate_detection_batch, pin_memory=pin,
        drop_last=True, seed=cfg.seed, persistent=True)
    val_loader = DataLoaderAdapter(
        val_ds, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_detection_batch, pin_memory=pin,
        drop_last=False, seed=cfg.seed)
    test_loader = DataLoaderAdapter(
        final_test_ds, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=collate_detection_batch, pin_memory=pin,
        drop_last=False, seed=cfg.seed)
    return train_loader, val_loader, test_loader, names


def load_pretrained_weights(path, model, device):
    """Load ONLY the model weights (new training, fresh optimizer)."""
    ckpt = safe_torch_load(path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"[pretrained] Weights '{path}' do not match the model. "
            f"Check `version` and the class count.\n  Detail: {e}"
        ) from e
    logger.info(f"Pretrained weights loaded from '{path}'")
    logger.info("Starting a NEW training (epoch=0, fresh optimizer)")


def build_training_objects(cfg: TrainConfig, model, device,
                           steps_per_epoch):
    """Optimizer, scheduler, EMA and AMP scaler from the config."""
    opt = cfg.optimization
    # Ultralytics convention: the default weight decay (0.0005) was
    # tuned for an effective batch of `nbs` (64). Smaller batches need
    # proportionally less decay, otherwise they are over-regularized.
    weight_decay = opt.weight_decay
    if opt.nbs and opt.nbs > 0:
        effective_bs = opt.batch_size * max(opt.grad_accum, 1)
        weight_decay = opt.weight_decay * effective_bs / opt.nbs
        if abs(weight_decay - opt.weight_decay) > 1e-12:
            logger.info(
                f"weight_decay scaled: {opt.weight_decay} -> "
                f"{weight_decay:.6f} (effective batch {effective_bs} "
                f"/ nbs {opt.nbs})")
    optimizer = build_optimizer(
        model, name=opt.optimizer, lr=opt.max_lr,
        momentum=opt.momentum, weight_decay=weight_decay)
    scheduler = build_scheduler(
        opt.scheduler, opt.max_lr, opt.min_lr, opt.warmup_epochs,
        opt.epochs, steps_per_epoch, momentum=opt.momentum,
        warmup_momentum=opt.warmup_momentum,
        warmup_bias_lr=opt.warmup_bias_lr)

    ema = ModelEMA(model, decay=opt.ema_decay, tau=opt.ema_tau) \
        if opt.ema else None
    amp_enabled = bool(opt.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=True) \
        if amp_enabled else None
    if opt.amp and not amp_enabled:
        logger.info("AMP requested but device is CPU: FP32 training.")
    elif amp_enabled:
        logger.info("Mixed precision (AMP) enabled")
    return optimizer, scheduler, ema, scaler


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model (train + val + final test).")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to train.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger.info("Starting training pipeline")
    cfg = load_train_config(args.config)
    raw_config = config_to_dict(cfg)
    log_dict(raw_config)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"device={device}")
    if device.type == 'cuda' and cfg.optimization.cudnn_benchmark:
        # Fixed shapes (square letterbox), so autotuning pays off.
        torch.backends.cudnn.benchmark = True

    # --- Run folder (resume reuses the highest numbered folder) ---
    run_dir, resumed = prepare_run_dir(
        cfg.output_dir, cfg.run_name, kind='train',
        resume=cfg.checkpoint.resume)
    add_file_logging(run_dir / 'logs', prefix='train',
                     level=args.log_level)
    save_config_used(run_dir, raw_config)

    # --- Data ---
    loaders = build_dataloaders(cfg, device)
    train_loader, val_loader, test_loader, names = loaders

    # --- Model ---
    model = YOLO(version=cfg.model.version, num_classes=len(names),
                   input_size=cfg.dataset.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: YOLOv8-{cfg.model.version} | "
                f"{n_params:.3f}M params | "
                f"strides={model.head.stride.tolist()}")

    loss_fn = ComputeLoss(model, cfg.loss.gains())

    # --- Startup mode: checkpoint resume > pretrained > scratch ---
    if not resumed and cfg.model.pretrained_weights:
        load_pretrained_weights(
            cfg.model.pretrained_weights, model, device)
    elif not resumed:
        logger.info("No initial weights given. Training from scratch.")

    # Freeze BEFORE building the optimizer, so frozen parameters get
    # no optimizer state at all.
    if cfg.model.freeze_feature_layers:
        freeze_feature_layers(model)

    optimizer, scheduler, ema, scaler = build_training_objects(
        cfg, model, device, max(len(train_loader), 1))

    # Summary AFTER weight loading and freeze, so the "Trainable"
    # column shows the real state of the model.
    logger.info(f"===== model architecture =====")
    log_model_summary(
        model, input_size=(1, 3, cfg.dataset.image_size,
                           cfg.dataset.image_size), device=device)

    trainer = Trainer(
        model, loss_fn, optimizer, scheduler, train_loader,
        val_loader, test_loader, cfg, run_dir, device,
        ema=ema, scaler=scaler, raw_config=raw_config)

    if resumed:
        latest = CheckpointManager(run_dir / 'checkpoints').latest()
        if latest is not None:
            trainer.load_checkpoint(latest)

    trainer.fit()


if __name__ == '__main__':
    main()

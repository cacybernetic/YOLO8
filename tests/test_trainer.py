"""End to end Trainer tests on a tiny dataset (CPU, small images)."""

import torch
import pandas as pd

from yolov8.config import TrainConfig, config_to_dict
from yolov8.dataset import DataLoaderAdapter, collate_detection_batch
from yolov8.dataset.factory import (build_train_dataset,
                                    build_test_dataset,
                                    split_val_from_test)
from yolov8.lossfn import ComputeLoss
from yolov8.model import MyYolo
from yolov8.training import (Trainer, build_optimizer, build_scheduler,
                             prepare_run_dir, CheckpointManager)


def _make_cfg(dataset_paths, epochs=1, ckpt_step=1):
    cfg = TrainConfig()
    cfg.device = 'cpu'
    cfg.log_interval = 0
    cfg.dataset.train_path = str(dataset_paths['train'])
    cfg.dataset.test_path = str(dataset_paths['test'])
    cfg.dataset.image_size = 64
    cfg.dataset.val_prob = 0.5
    cfg.dataset.augment.enabled = False
    cfg.dataset.augment.close_mosaic = 0
    cfg.optimization.epochs = epochs
    cfg.optimization.batch_size = 2
    cfg.optimization.num_workers = 0
    cfg.optimization.amp = False
    cfg.optimization.ema = False
    cfg.checkpoint.ckpt_step = ckpt_step
    return cfg


def _make_trainer(cfg, run_dir):
    device = torch.device('cpu')
    train_ds = build_train_dataset(cfg.dataset)
    test_ds = build_test_dataset(cfg.dataset)
    val_ds = split_val_from_test(test_ds, cfg.dataset.val_prob)

    kwargs = dict(batch_size=cfg.optimization.batch_size,
                  num_workers=0, collate_fn=collate_detection_batch)
    train_loader = DataLoaderAdapter(train_ds, shuffle=True,
                                     drop_last=True, **kwargs)
    val_loader = DataLoaderAdapter(val_ds, **kwargs)
    test_loader = DataLoaderAdapter(test_ds, **kwargs)

    model = MyYolo(version='n', num_classes=2, input_size=64)
    loss_fn = ComputeLoss(model, cfg.loss.gains())
    optimizer = build_optimizer(model, lr=cfg.optimization.max_lr)
    scheduler = build_scheduler(
        'cosine', cfg.optimization.max_lr, cfg.optimization.min_lr,
        1, cfg.optimization.epochs, max(len(train_loader), 1))
    return Trainer(model, loss_fn, optimizer, scheduler, train_loader,
                   val_loader, test_loader, cfg, run_dir, device,
                   raw_config=config_to_dict(cfg))


def test_full_training_run(tiny_dataset, tmp_path):
    cfg = _make_cfg(tiny_dataset, epochs=1)
    run_dir, _ = prepare_run_dir(tmp_path / 'runs', 'demo', 'train')
    trainer = _make_trainer(cfg, run_dir)
    results = trainer.fit()

    # Final test metrics are returned and written.
    assert 'map50' in results and 'total' in results
    assert (run_dir / 'test_results.csv').exists()
    # Outputs required by the spec.
    assert (run_dir / 'weights' / 'last.pt').exists()
    assert (run_dir / 'history.csv').exists()
    assert (run_dir / 'plotes' / 'training_history.png').exists()
    assert any((run_dir / 'checkpoints').glob('checkpoint_e*.pth'))
    # History has one train epoch recorded.
    df = pd.read_csv(run_dir / 'history.csv')
    assert len(df) == 1
    assert 'map50' in df.columns


def test_mid_epoch_checkpoint_and_resume(tiny_dataset, tmp_path):
    cfg = _make_cfg(tiny_dataset, epochs=1, ckpt_step=1)
    run_dir, _ = prepare_run_dir(tmp_path / 'runs', 'demo', 'train')

    # First trainer: run exactly one train batch then checkpoint.
    trainer1 = _make_trainer(cfg, run_dir)
    it = iter(trainer1.train_loader)
    images, targets, _ = next(it)
    trainer1.scheduler.step(0, trainer1.optimizer)
    trainer1._train_step(images, targets, accum=1)
    trainer1._save_checkpoint()
    del it

    latest = CheckpointManager(run_dir / 'checkpoints').latest()
    assert latest is not None

    # Second trainer: resume and finish the whole run.
    trainer2 = _make_trainer(cfg, run_dir)
    trainer2.load_checkpoint(latest)
    assert trainer2.train_loader.position == 1  # one batch consumed
    assert trainer2.opt_step == 1
    results = trainer2.fit()
    assert 'map50' in results
    # The resumed epoch is complete: exactly one history row.
    df = pd.read_csv(run_dir / 'history.csv')
    assert list(df['epoch']) == [1]


def test_best_and_last_weights_load_back(tiny_dataset, tmp_path):
    cfg = _make_cfg(tiny_dataset, epochs=1)
    run_dir, _ = prepare_run_dir(tmp_path / 'runs', 'demo', 'train')
    trainer = _make_trainer(cfg, run_dir)
    trainer.fit()

    payload = torch.load(run_dir / 'weights' / 'last.pt',
                         map_location='cpu', weights_only=True)
    model = MyYolo(version='n', num_classes=2, input_size=64)
    model.load_state_dict(payload['model'])  # strict load must pass

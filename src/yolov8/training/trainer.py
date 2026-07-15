"""Trainer: training loop with validation, test and fault tolerance.

The Trainer runs three passes:
  - train: one epoch over the train loader;
  - val: per-epoch validation on a fraction of the test set;
  - test: final evaluation, after the last epoch, on the held-out
    part of the test split (disjoint from the validation part).

A checkpoint is written every `ckpt_step` optimizer steps (train) or
batches (val / test), plus at every phase boundary. Each checkpoint
stores the model, optimizer, EMA, AMP scaler, the three data loader
adapters, the partial loss meters and metric accumulators, the RNG
states and the history. Training can therefore resume in the middle
of any pass without seeing a sample twice in the same epoch.
"""

import math
import time
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from yolov8.logging import safe_torch_load
from yolov8.metrics import (MetricAccumulator, non_max_suppression,
                            build_val_targets)
from yolov8.plotting import plot_training_history
from yolov8.training.checkpoints import (CheckpointManager,
                                         capture_rng_state,
                                         restore_rng_state, atomic_save)
from yolov8.training.meters import LossMeters
from yolov8.training.optimizers import apply_batchnorm_freeze

BAR_CHARS = "░█"
BOLD = "\033[1m"
RESET = "\033[0m"

# Metrics accepted by checkpoint.best_metric (all maximized).
ALLOWED_BEST_METRICS = ('map50', 'map', 'precision', 'recall')


def empty_history():
    return {
        'epochs_train': [], 'train_loss': [], 'train_box': [],
        'train_cls': [], 'train_dfl': [],
        'epochs_val': [], 'val_loss': [], 'val_box': [],
        'val_cls': [], 'val_dfl': [],
        'val_precision': [], 'val_recall': [],
        'val_map50': [], 'val_map': [],
    }


class Trainer:
    """Orchestrates the train / val / test passes of one run."""

    def __init__(self, model, loss_fn, optimizer, scheduler,
                 train_loader, val_loader, test_loader, cfg, run_dir,
                 device, ema=None, scaler=None, raw_config=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.ema = ema
        self.scaler = scaler
        self.raw_config = raw_config or {}

        # Fail fast on a config typo: a silent warning at every epoch
        # would mean best.pt is never written.
        if cfg.checkpoint.best_metric not in ALLOWED_BEST_METRICS:
            raise ValueError(
                f"Unknown checkpoint.best_metric "
                f"'{cfg.checkpoint.best_metric}'. "
                f"Choose one of {ALLOWED_BEST_METRICS}.")

        self.ckpts = CheckpointManager(
            self.run_dir / 'checkpoints',
            max_keep=cfg.checkpoint.max_checkpoint)

        self.epoch = 0
        self.phase = 'train'   # 'train' | 'val' | 'test' | 'done'
        self.opt_step = 0      # optimizer steps done in current epoch
        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        self.history = empty_history()
        self.train_meters = LossMeters(device)
        self._eval_resume = None
        self._micro = 0        # micro batches since last optimizer step
        self._stop = False
        self._epoch_times = []
        self._mosaic_closed = False
        self._amp_enabled = scaler is not None

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def _checkpoint_state(self, eval_state=None):
        state = {
            'version': 1,
            'epoch': self.epoch,
            'phase': self.phase,
            'opt_step': self.opt_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loader': self.train_loader.state_dict(),
            'val_loader': self.val_loader.state_dict(),
            'test_loader': self.test_loader.state_dict(),
            'train_meters': self.train_meters.state_dict(),
            'eval_state': eval_state,
            'best_metric': self.best_metric,
            'epochs_no_improve': self.epochs_no_improve,
            'history': self.history,
            'rng': capture_rng_state(),
            'config': self.raw_config,
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        if self.ema is not None:
            state['ema'] = self.ema.ema.state_dict()
            state['ema_updates'] = self.ema.updates
        return state

    def _save_checkpoint(self, eval_state=None, step=None):
        step = self.opt_step if step is None else step
        state = self._checkpoint_state(eval_state=eval_state)
        self.ckpts.save(state, self.epoch + 1, step)

    def load_checkpoint(self, path):
        """Restore the full trainer state from a checkpoint file."""
        ckpt = safe_torch_load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self._load_optim_state(ckpt)
        self._load_loader_state(ckpt)
        self.epoch = int(ckpt.get('epoch', 0))
        self.phase = ckpt.get('phase', 'train')
        self.opt_step = int(ckpt.get('opt_step', 0))
        self.best_metric = float(
            ckpt.get('best_metric', -float('inf')))
        self.epochs_no_improve = int(ckpt.get('epochs_no_improve', 0))
        self.history = ckpt.get('history', empty_history())
        if 'train_meters' in ckpt:
            self.train_meters.load_state_dict(ckpt['train_meters'])
        self._eval_resume = ckpt.get('eval_state')
        restore_rng_state(ckpt.get('rng'))
        logger.info(f"Checkpoint restored: {Path(path).name} "
                    f"(epoch={self.epoch + 1}, phase={self.phase}, "
                    f"opt_step={self.opt_step}, "
                    f"best={self.best_metric:.4f})")

    def _load_optim_state(self, ckpt):
        if 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                logger.warning(f"Optimizer state not restored ({e}); "
                               f"a fresh optimizer is used.")
        if self.scaler is not None and ckpt.get('scaler') is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
        if self.ema is not None and 'ema' in ckpt:
            try:
                self.ema.load_state_dict(
                    ckpt['ema'], updates=ckpt.get('ema_updates', 0))
                logger.info(f"EMA restored (updates={self.ema.updates})")
            except Exception as e:
                logger.warning(f"EMA not restored ({e}); "
                               f"EMA restarts from the current weights.")

    def _load_loader_state(self, ckpt):
        pairs = (('train_loader', self.train_loader),
                 ('val_loader', self.val_loader),
                 ('test_loader', self.test_loader))
        for key, loader in pairs:
            if key in ckpt:
                loader.load_state_dict(ckpt[key])

    # ------------------------------------------------------------------
    # Weights export (best.pt / last.pt)
    # ------------------------------------------------------------------

    def _weights_payload(self, use_ema):
        model = self.ema.ema if (use_ema and self.ema) else self.model
        return {
            'model': model.state_dict(),
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': self.raw_config,
        }

    def _save_last(self):
        path = self.run_dir / 'weights' / 'last.pt'
        atomic_save(self._weights_payload(use_ema=False), path)

    def _save_best(self):
        path = self.run_dir / 'weights' / 'best.pt'
        atomic_save(self._weights_payload(use_ema=True), path)
        metric_name = self.cfg.checkpoint.best_metric
        logger.success(f"New best {metric_name} = "
                       f"{self.best_metric:.4f} -> saved best.pt")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        """Run the full training then the final test evaluation."""
        total = self.cfg.optimization.epochs
        self._log_run_summary()
        epoch_bar = tqdm(
            total=total, initial=min(self.epoch, total),
            desc=f"{BOLD}TRAINING{RESET}", colour='cyan', position=0,
            leave=True, dynamic_ncols=True, ascii=BAR_CHARS)
        self._update_epoch_bar(epoch_bar)

        while self.epoch < total and not self._stop:
            t0 = time.time()
            self._run_one_epoch(self.epoch)
            self._epoch_times.append(time.time() - t0)
            epoch_bar.update(1)
            self._update_epoch_bar(epoch_bar)
        epoch_bar.close()

        results = self._run_final_test()
        logger.info("Training finished.")
        if self.best_metric > -float('inf'):
            logger.info(f"Best {self.cfg.checkpoint.best_metric}: "
                        f"{self.best_metric:.4f}")
        return results

    def _run_one_epoch(self, epoch):
        logger.info(f"===== Starting epoch {epoch + 1}/"
                    f"{self.cfg.optimization.epochs} =====")
        self._maybe_close_mosaic(epoch)

        if self.phase == 'train':
            stats = self._train_epoch(epoch)
            self._record_train(epoch, stats)
            self.phase = 'val'
            self._save_checkpoint()

        if self.phase == 'val':
            val_stats = self._run_validation(epoch)
            self.phase = 'train'
            self.epoch = epoch + 1
            self.opt_step = 0
            self._finalize_epoch(epoch, val_stats)

    def _run_validation(self, epoch):
        interval = max(self.cfg.validation.interval, 1)
        if (epoch + 1) % interval != 0:
            self._eval_resume = None
            return {}
        return self._eval_pass(self.val_loader, 'val')

    def _run_final_test(self):
        if self.phase == 'done':
            return {}
        self.phase = 'test'
        logger.info("===== Final evaluation on the held-out test set "
                    "=====")
        results = self._eval_pass(self.test_loader, 'test')
        self._log_metrics_table("Final test metrics", {}, results,
                                right_name='test')
        self._write_test_results(results)
        self.phase = 'done'
        return results

    def _write_test_results(self, results):
        rows = [(k, round(float(v), 6)) for k, v in results.items()]
        df = pd.DataFrame(rows, columns=['metric', 'value'])
        path = self.run_dir / 'test_results.csv'
        df.to_csv(path, index=False)
        logger.info(f"Test results written: {path}")

    # ------------------------------------------------------------------
    # Train pass
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch):
        self.model.train()
        if self.cfg.model.freeze_feature_layers:
            apply_batchnorm_freeze(self.model)

        steps_per_epoch = max(len(self.train_loader), 1)
        accum = max(int(self.cfg.optimization.grad_accum), 1)
        ckpt_step = int(self.cfg.checkpoint.ckpt_step)
        log_interval = int(self.cfg.log_interval)
        t0 = time.time()

        step_bar = tqdm(
            total=steps_per_epoch, initial=self.train_loader.position,
            desc=f"  train e{epoch + 1}", colour='green', position=1,
            leave=False, dynamic_ncols=True, ascii=BAR_CHARS)

        for images, targets, _paths in self.train_loader:
            step = self.train_loader.position - 1
            global_step = epoch * steps_per_epoch + step
            self.scheduler.step(global_step, self.optimizer)
            self._train_step(images, targets, accum)
            if self._micro == 0 and ckpt_step > 0 \
                    and self.opt_step % ckpt_step == 0:
                self._save_checkpoint()
            step_bar.update(1)
            if log_interval and (step + 1) % log_interval == 0:
                self._log_train_step(step + 1, steps_per_epoch,
                                     epoch, step_bar)
        self._flush_pending_grads()
        step_bar.close()

        stats = self.train_meters.averages()
        self._ensure_finite(stats['total'], f"epoch {epoch + 1}")
        self.train_meters.reset()
        elapsed = time.time() - t0
        logger.info(
            f"epoch {epoch + 1} train done in {elapsed:.1f}s | "
            f"box={stats['box']:.4f} cls={stats['cls']:.4f} "
            f"dfl={stats['dfl']:.4f} total={stats['total']:.4f}")
        return stats

    @staticmethod
    def _ensure_finite(value, context):
        """Abort with a clear error when the loss went NaN/inf.

        Without this check a divergence under AMP only shows up as the
        GradScaler silently skipping every step: the run keeps going
        but the model stops learning.
        """
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite training loss ({value}) at {context}. "
                f"The run diverged: lower the learning rate, check the "
                f"dataset, or disable AMP to locate the source.")

    def _train_step(self, images, targets, accum):
        """One micro batch: forward, loss, backward, optional step."""
        images = images.to(self.device, non_blocking=True)
        bs = images.size(0)
        with torch.autocast(device_type=self.device.type,
                            enabled=self._amp_enabled):
            outputs = self.model(images)
            loss_box, loss_cls, loss_dfl = self.loss_fn(outputs, targets)
            loss = loss_box + loss_cls + loss_dfl

        # `* bs`: the loss is normalized per target; Ultralytics
        # backpropagates loss.sum() * batch_size and the default
        # hyperparameters are tuned for that gradient scale.
        scaled = loss * bs / accum
        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()
        self.train_meters.update(loss_box, loss_cls, loss_dfl, loss, bs)

        self._micro += 1
        if self._micro >= accum:
            self._apply_optimizer_step()

    def _apply_optimizer_step(self):
        """Optimizer step with grad clipping, AMP and EMA update."""
        grad_clip = self.cfg.optimization.grad_clip
        if self.scaler is not None:
            if grad_clip and grad_clip > 0:
                # unscale_ BEFORE clipping, otherwise the clipping
                # would see gradients multiplied by the scale factor.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)
        self._micro = 0
        self.opt_step += 1

    def _flush_pending_grads(self):
        """Apply the last incomplete accumulation group of the epoch."""
        if self._micro > 0:
            self._apply_optimizer_step()

    def _log_train_step(self, step, total_steps, epoch, step_bar):
        avg = self.train_meters.averages()
        self._ensure_finite(
            avg['total'],
            f"epoch {epoch + 1}, step {step}/{total_steps}")
        lr_now = self.optimizer.param_groups[0]['lr']
        step_bar.set_postfix({
            'loss': f"{avg['total']:.4f}",
            'box': f"{avg['box']:.4f}",
            'cls': f"{avg['cls']:.4f}",
            'dfl': f"{avg['dfl']:.4f}",
        })
        amp_note = ''
        if self.scaler is not None:
            amp_note = f" | amp_scale {self.scaler.get_scale():.0f}"
        logger.info(
            f"epoch {epoch + 1}/{self.cfg.optimization.epochs} | "
            f"step {step}/{total_steps} | lr {lr_now:.5f} | "
            f"avg_loss {avg['total']:.4f} (box {avg['box']:.4f} "
            f"cls {avg['cls']:.4f} dfl {avg['dfl']:.4f}){amp_note}")

    # ------------------------------------------------------------------
    # Eval passes (val and test)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_pass(self, loader, phase):
        """Run one evaluation pass, resumable batch by batch."""
        model = self.ema.ema if self.ema is not None else self.model
        model.eval()
        meters = LossMeters(self.device)
        accumulator = MetricAccumulator(self.device)
        self._restore_eval_state(phase, meters, accumulator)

        total = len(loader)
        ckpt_step = int(self.cfg.checkpoint.ckpt_step)
        step_bar = tqdm(
            total=total, initial=loader.position,
            desc=f"  {phase} e{min(self.epoch + 1, self.cfg.optimization.epochs)}",
            colour='yellow', position=1, leave=False,
            dynamic_ncols=True, ascii=BAR_CHARS)

        for images, targets, _paths in loader:
            self._eval_batch(model, images, targets, meters, accumulator)
            step_bar.update(1)
            if ckpt_step > 0 and loader.position < total \
                    and loader.position % ckpt_step == 0:
                eval_state = {
                    'phase': phase,
                    'meters': meters.state_dict(),
                    'accumulator': accumulator.state_dict(),
                }
                self._save_checkpoint(eval_state=eval_state,
                                      step=loader.position)
        step_bar.close()

        results = accumulator.compute()
        results.update(meters.averages())
        return results

    def _restore_eval_state(self, phase, meters, accumulator):
        pending = self._eval_resume
        self._eval_resume = None
        if pending and pending.get('phase') == phase:
            meters.load_state_dict(pending['meters'])
            accumulator.load_state_dict(pending['accumulator'])
            logger.info(f"Resumed a partial {phase} pass "
                        f"({len(accumulator.stats)} images already done)")

    def _eval_batch(self, model, images, targets, meters, accumulator):
        images = images.to(self.device, non_blocking=True)
        # Same mixed precision as training: halves the eval time on
        # GPU. The loss itself recomputes in fp32 internally, and the
        # decoded predictions are cast back before NMS and matching.
        with torch.autocast(device_type=self.device.type,
                            enabled=self._amp_enabled):
            out = model(images)
        if isinstance(out, tuple):
            inference_out, raw_outputs = out
        else:
            inference_out, raw_outputs = out, None
        inference_out = inference_out.float()

        if raw_outputs is not None:
            lb, lc, ld = self.loss_fn(raw_outputs, targets)
            meters.update(lb, lc, ld, lb + lc + ld, images.size(0))

        preds = non_max_suppression(
            inference_out,
            confidence_threshold=self.cfg.validation.conf_threshold,
            iou_threshold=self.cfg.validation.iou_threshold)
        gts = build_val_targets(
            images, targets, self.cfg.dataset.image_size, self.device)
        accumulator.update(preds, gts)

    # ------------------------------------------------------------------
    # End of epoch bookkeeping
    # ------------------------------------------------------------------

    def _record_train(self, epoch, stats):
        h = self.history
        h['epochs_train'].append(epoch + 1)
        h['train_loss'].append(float(stats.get('total', 0.0)))
        h['train_box'].append(float(stats.get('box', 0.0)))
        h['train_cls'].append(float(stats.get('cls', 0.0)))
        h['train_dfl'].append(float(stats.get('dfl', 0.0)))

    def _record_val(self, epoch, stats):
        h = self.history
        h['epochs_val'].append(epoch + 1)
        h['val_loss'].append(float(stats.get('total', 0.0)))
        h['val_box'].append(float(stats.get('box', 0.0)))
        h['val_cls'].append(float(stats.get('cls', 0.0)))
        h['val_dfl'].append(float(stats.get('dfl', 0.0)))
        h['val_precision'].append(float(stats.get('precision', 0.0)))
        h['val_recall'].append(float(stats.get('recall', 0.0)))
        h['val_map50'].append(float(stats.get('map50', 0.0)))
        h['val_map'].append(float(stats.get('map', 0.0)))

    def _finalize_epoch(self, epoch, val_stats):
        """History, plots, CSV, last/best weights, early stopping."""
        is_best = False
        if val_stats:
            self._record_val(epoch, val_stats)
            is_best = self._update_best(val_stats)
            self._log_metrics_table(
                f"Epoch {epoch + 1}/{self.cfg.optimization.epochs} "
                f"metrics", self._last_train_stats(), val_stats)

        self._plot_history()
        self._write_history_csv()
        self._save_last()
        if is_best:
            self._save_best()
        self._save_checkpoint()
        self._check_early_stop()

    def _last_train_stats(self):
        h = self.history
        if not h['epochs_train']:
            return {}
        return {'total': h['train_loss'][-1], 'box': h['train_box'][-1],
                'cls': h['train_cls'][-1], 'dfl': h['train_dfl'][-1]}

    def _update_best(self, val_stats):
        metric_name = self.cfg.checkpoint.best_metric
        metric = val_stats.get(metric_name)
        if metric is None:
            logger.warning(f"Unknown best metric '{metric_name}'")
            return False
        if metric > self.best_metric:
            self.best_metric = float(metric)
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    def _check_early_stop(self):
        patience = self.cfg.optimization.patience
        if patience > 0 and self.epochs_no_improve >= patience:
            logger.warning(
                f"Early stopping: no {self.cfg.checkpoint.best_metric} "
                f"improvement for {patience} validations "
                f"(best={self.best_metric:.4f}).")
            self._stop = True

    def _plot_history(self):
        # Never break the training because of a plotting problem.
        try:
            path = self.run_dir / 'plotes' / 'training_history.png'
            plot_training_history(self.history, path)
        except Exception as e:
            logger.warning(f"History plot failed: {e}")

    def _write_history_csv(self):
        h = self.history
        rows = []
        val_by_epoch = {e: i for i, e in enumerate(h['epochs_val'])}
        for i, e in enumerate(h['epochs_train']):
            row = {'epoch': e,
                   'train_loss': h['train_loss'][i],
                   'train_box': h['train_box'][i],
                   'train_cls': h['train_cls'][i],
                   'train_dfl': h['train_dfl'][i]}
            j = val_by_epoch.get(e)
            if j is not None:
                row.update({
                    'val_loss': h['val_loss'][j],
                    'val_box': h['val_box'][j],
                    'val_cls': h['val_cls'][j],
                    'val_dfl': h['val_dfl'][j],
                    'precision': h['val_precision'][j],
                    'recall': h['val_recall'][j],
                    'map50': h['val_map50'][j],
                    'map': h['val_map'][j]})
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            self.run_dir / 'history.csv', index=False)

    # ------------------------------------------------------------------
    # Console rendering
    # ------------------------------------------------------------------

    def _log_metrics_table(self, title, train_stats, right_stats,
                           right_name='val'):
        logger.info(f"===== {title} =====")
        logger.info(f"  {'metric':<12} {'train':>12} {right_name:>12}")
        loss_keys = (('loss', 'total'), ('box', 'box'),
                     ('cls', 'cls'), ('dfl', 'dfl'))
        for label, key in loss_keys:
            left = self._fmt(train_stats.get(key))
            right = self._fmt(right_stats.get(key))
            logger.info(f"  {label:<12} {left:>12} {right:>12}")
        for key in ('precision', 'recall', 'map50', 'map'):
            right = self._fmt(right_stats.get(key))
            logger.info(f"  {key:<12} {'-':>12} {right:>12}")

    @staticmethod
    def _fmt(value):
        return f"{value:.4f}" if value is not None else '-'

    def _update_epoch_bar(self, epoch_bar):
        postfix = {}
        if self._epoch_times:
            avg = sum(self._epoch_times) / len(self._epoch_times)
            m, s = divmod(int(avg), 60)
            postfix['avg_epoch'] = f"{m:02d}:{s:02d}"
        if self.best_metric > -float('inf'):
            metric = self.cfg.checkpoint.best_metric
            postfix[f'best_{metric}'] = f"{self.best_metric:.4f}"
        postfix['lr'] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
        epoch_bar.set_postfix(postfix)

    def _maybe_close_mosaic(self, epoch):
        close = int(self.cfg.dataset.augment.close_mosaic)
        total = self.cfg.optimization.epochs
        if close <= 0 or self._mosaic_closed or epoch < total - close:
            return
        dataset = getattr(self.train_loader, 'dataset', None)
        augmenter = getattr(dataset, 'augmenter', None)
        if augmenter is not None and (
                augmenter.params.get('mosaic', 0) > 0
                or augmenter.params.get('mixup', 0) > 0):
            logger.info(f"close_mosaic: mosaic and mixup disabled for "
                        f"the last {total - epoch} epochs")
            augmenter.params['mosaic'] = 0.0
            augmenter.params['mixup'] = 0.0
            # Persistent DataLoader workers hold their own dataset
            # copy: they must be restarted to see the new params.
            invalidate = getattr(self.train_loader,
                                 'invalidate_workers', None)
            if invalidate is not None:
                invalidate()
        self._mosaic_closed = True

    def _log_run_summary(self):
        cfg = self.cfg
        opt = cfg.optimization
        effective = opt.batch_size * max(opt.grad_accum, 1)
        steps = max(len(self.train_loader), 1)
        opt_steps = math.ceil(steps / max(opt.grad_accum, 1))
        logger.info("===== Run summary =====")
        logger.info(f"  device                = {self.device}")
        logger.info(f"  epochs                = {opt.epochs} "
                    f"(start_epoch={self.epoch})")
        logger.info(f"  batch_size            = {opt.batch_size} x "
                    f"grad_accum={opt.grad_accum} "
                    f"-> effective={effective}")
        logger.info(f"  batches/epoch         = {steps} | "
                    f"optimizer_steps/epoch={opt_steps}")
        logger.info(f"  optimizer             = {opt.optimizer} "
                    f"lr={opt.max_lr:.3e} wd={opt.weight_decay}")
        logger.info(f"  scheduler             = {opt.scheduler}")
        logger.info(f"  grad_clip_norm        = {opt.grad_clip}")
        logger.info(f"  amp={self._amp_enabled} | "
                    f"ema={self.ema is not None}")
        logger.info(f"  best criterion        = "
                    f"{cfg.checkpoint.best_metric} (mode=max)")
        logger.info(f"  ckpt_step             = "
                    f"{cfg.checkpoint.ckpt_step} | "
                    f"max_checkpoint={cfg.checkpoint.max_checkpoint}")
        logger.info(f"  data                  = "
                    f"train {len(self.train_loader.dataset)} | "
                    f"val {len(self.val_loader.dataset)} | "
                    f"test {len(self.test_loader.dataset)}")
        logger.info(f"  outputs               = {self.run_dir}")

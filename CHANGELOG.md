# Changelog


Le format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

- YOLO model implementation.
- Training implementation.
- Evaluation implementation.
- Checkpoint management.
- Best model saving.
- Gradient accumulation step.
- Progress bar on training, validation and evaluation.
- Model inference on image file.
- Model summary in train, evaluate and export program.
- Model fine-tuning.
- Prediction with onnx model inference + preprocessing and postprocessing with numpy.
- Inference of ONNX YOLO model on video capture in Rust and Python.
- Evaluation Metrics results building.
- Plotting of Training/Validation history.
- Setting of logging system (loguru).
- README: a **Model architecture** section (sizes and their scaling
  factors, backbone, PAN-FPN neck, decoupled anchor-free head, DFL and
  the two easy-to-miss details: branch widths and bias init) and a
  **How the training works** section (the three loss terms, the TAL
  assignment, parameter groups, warmup, EMA, the three passes). The
  project tree was wrong about the Rust binaries (`yolov8rust/` and
  `yololivers/`, not `src/main.rs`) and left out several modules.
- `grad_norm` in the periodic training log: the gradient norm measured
  **before** clipping, so it still says something useful when the
  clipping is active. Shown as `n/a` on the logs that happen before
  the first optimizer step (only when `log_interval < grad_accum`).
 

### Changed 

- Training step log format. The fields use a `label: value` style, the
  loss parts moved into brackets and `lr` moved to the end:
  ```
  epoch: 12/50 | step: 60/62 | avg_loss: 7.1637 [box: 2.0917, cls: 3.0699, dfl: 2.0020] | grad_norm: 0.2392 | lr: 0.00914
  ```
  With AMP on, an `amp_scale: N` field is inserted before `grad_norm`.
- Progress bars now receive the **running average** of the metrics at
  every loop turn. Before, the train bar was only refreshed every
  `log_interval` steps (and never when `log_interval: 0`), and the
  val / test / `evalyolo8` bars showed no metric at all. The eval bar
  also shows the running mean IoU. Cost: `LossMeters.averages()` now
  forces one GPU sync per step.
- `logger.success` is reserved for a new best score (`New best <metric>
  -> saved best.pt`). The end of epoch summary, the end of training,
  the export, the evaluation and the HDF5 build lines are plain
  `logger.info`, so a SUCCESS line in a log file always means the model
  improved.
- Under AMP, `unscale_` now runs before every optimizer step instead of
  only when clipping is on, so the reported `grad_norm` is the true
  unscaled norm and not the value multiplied by the AMP scale factor.
- **BREAKING — model class renamed**: `MyYolo` is now `YOLO`
  (`from yolov8 import YOLO`). No alias is kept, so imports of
  `MyYolo` must be updated. Checkpoints are not affected: the class
  name is not part of the saved state dict.
- **BREAKING — head architecture**: the intermediate widths of the
  detection head now follow the standard YOLOv8 convention
  (`c2 = max(16, ch0/4, 64)`, `c3 = max(ch0, min(nc, 100))`). Small
  class counts no longer squeeze the classification branch through an
  `nc`-channel bottleneck. Parameter counts now match the reference
  models (v8s: 10.50M -> 11.17M). **Checkpoints trained before this
  change cannot be loaded** (different tensor shapes).
- **BREAKING — C2f concat order**: the bottleneck chain now runs on
  the second chunk and outputs are appended (canonical YOLOv8 layout).
  Old checkpoints load but produce different results; retrain.
- Validation and final test are now **disjoint** parts of the test
  split: model selection (best.pt, early stopping) no longer sees the
  samples used for the final reported metrics.
- The per-epoch validation and the evaluation CLI now share one COCO
  101-point AP implementation, so `map50` is comparable between
  `history.csv` and `results.csv`.
- Weight decay is scaled by the effective batch size against a nominal
  batch (`optimization.nbs`, default 64).
- Augmentation pipeline reordered to the YOLOv8 convention
  (geometry -> mixup -> HSV -> flips -> pixel effects); mosaic tiles
  are resized on their long side without letterbox padding; the MixUp
  partner goes through the same mosaic + geometry treatment.
- The training loss always runs in float32 under AMP (targets are no
  longer quantized to fp16); the eval forward now uses autocast.
- `DataLoaderAdapter` is a plain class (no more nn.Module) with
  optional persistent workers for the train loader; old checkpoint
  states still load.
- `safe_torch_load` no longer falls back to full pickle automatically;
  pass `allow_pickle=True` explicitly for trusted legacy files.
- Scan cache bumped to version 2 (existing caches rebuild once).

### Fixed

- Scheduler: `warmup_epochs: 0` is respected, and the warmup is capped
  to 30% of the total step budget so short runs keep a real decay
  phase (previously a 100-step floor could consume nearly the whole
  training).
- The dataset scan cache is invalidated when label files are edited in
  place (the fingerprint now tracks image AND label file mtimes).
- Label class ids are validated against the class count at scan time
  and in the loss, with a clear error instead of a delayed CUDA
  device-side assert.
- Train/test `data.yaml` class list mismatches are a fatal error
  instead of a warning (class ids would silently disagree).
- Unknown `checkpoint.best_metric` values abort at startup instead of
  logging a warning every epoch and never writing best.pt.
- Non-finite training losses abort with a clear diagnostic instead of
  letting the AMP scaler silently skip every step; the AMP scale is
  logged with the periodic training stats.
- `close_mosaic` restarts persistent DataLoader workers so the change
  actually reaches them.
- `results.csv`: `n_pred` is filtered at the same confidence threshold
  as `n_tp`/`n_fp`, so `n_tp + n_fp == n_pred`.
- NMS no longer aliases one shared empty tensor across the batch.
- Per-step GPU syncs removed from the loss (vectorized target packing,
  unconditional box/DFL computation).


### Deprecated
<!-- - L'ancienne API v1 est dépréciée et sera supprimée dans la version 2.0 -->


### Security
<!-- - Mise à jour de la bibliothèque Y pour corriger une faille de sécurité -->

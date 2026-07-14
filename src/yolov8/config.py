"""Load and validate the YAML configuration files.

The train/eval configs are nested: they contain a `dataset` block, a
`model` block, and so on. Unknown keys are reported with a warning and
ignored, so comments and extra fields never break a run.
"""

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Nested blocks
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    enabled: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flip_lr: float = 0.5
    flip_ud: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    cutout: float = 0.0
    cutout_n_max: int = 4
    cutout_size_max: float = 0.25
    blur: float = 0.0
    noise: float = 0.0
    grayscale: float = 0.0
    # Turn off mosaic and mixup during the last N epochs (0 = never).
    close_mosaic: int = 10

    def params(self):
        """Augmentation values as a plain dict for the Augmenter."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name != 'close_mosaic'}


@dataclass
class DatasetConfig:
    train_path: str = ''
    test_path: str = ''
    use_hdf5: bool = False
    train_h5: str = 'data/train.h5'
    test_h5: str = 'data/test.h5'
    # Pre-flight scan: drop corrupt or invalid entries up front.
    validate: bool = True
    # Read and write the <name>.cache.json scan cache.
    cache: bool = True
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    # Fraction of the test set used for the per-epoch validation.
    val_prob: float = 0.5
    image_size: int = 640
    # Fallback class names when the dataset has no data.yaml.
    class_names: Optional[List[str]] = None
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class ModelConfig:
    version: str = 'n'
    # Start a NEW training from these weights (fine tuning or transfer
    # learning). Ignored when a checkpoint is resumed.
    pretrained_weights: Optional[str] = None
    freeze_feature_layers: bool = False


@dataclass
class OptimizationConfig:
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    optimizer: str = 'sgd'      # 'sgd' | 'adam' | 'adamw'
    max_lr: float = 0.01
    min_lr: float = 0.0001
    momentum: float = 0.937
    weight_decay: float = 0.0005
    scheduler: str = 'cosine'   # 'cosine' | 'linear'
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    grad_clip: float = 10.0
    # Gradient accumulation over N micro batches.
    grad_accum: int = 1
    amp: bool = True
    ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0
    # Early stopping: stop after N validations without improvement
    # of the best metric (0 = disabled).
    patience: int = 0
    cudnn_benchmark: bool = True


@dataclass
class LossConfig:
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5

    def gains(self):
        return {'box': self.box_gain, 'cls': self.cls_gain,
                'dfl': self.dfl_gain}


@dataclass
class CheckpointConfig:
    # Save a checkpoint every N optimizer steps (train phase) or every
    # N batches (val and test phases).
    ckpt_step: int = 200
    max_checkpoint: int = 5
    # Reuse the last run folder and its latest checkpoint.
    resume: bool = True
    # Metric used to pick the best model (map50, map, precision,
    # recall).
    best_metric: str = 'map50'


@dataclass
class ValidationConfig:
    interval: int = 1
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7


# ---------------------------------------------------------------------------
# Top level configs
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    run_name: str = 'yolov8'
    output_dir: str = 'runs'
    device: str = 'cuda'
    seed: int = 0
    log_interval: int = 10
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=CheckpointConfig)
    validation: ValidationConfig = field(
        default_factory=ValidationConfig)


@dataclass
class EvalConfig:
    run_name: str = 'yolov8'
    output_dir: str = 'runs'
    device: str = 'cuda'
    seed: int = 0
    weights: str = ''
    batch_size: int = 16
    num_workers: int = 4
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7
    # Number of example prediction images written to renders/.
    n_renders: int = 8
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class Hdf5BuildConfig:
    """Config of buildds: turn a dataset into ready-to-train HDF5."""
    device: str = 'cpu'
    # Number of extra augmented copies stored per train sample.
    augmented_copies: int = 0
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


@dataclass
class ExportConfig:
    weights: str = ''
    num_classes: int = 80
    version: str = 'n'
    image_size: int = 640
    output_path: str = 'weights/best.onnx'
    opset: int = 17
    dynamic: bool = True
    simplify: bool = True
    half: bool = False
    check: bool = True
    verify: bool = True
    verify_tolerance: float = 1e-3
    device: str = 'cpu'


@dataclass
class FinetuneConfig:
    """Build a fine-tunable checkpoint from pretrained weights.

    Backbone, neck, box branches and DFL do not depend on the number
    of classes: they are transferred. Only the classification branches
    are re-initialized for the new class count, with the Focal Loss
    bias prior b = -log((1 - pi) / pi), pi = cls_prior.
    """
    pretrained_weights: str = ''
    old_num_classes: int = 80
    new_num_classes: int = 80
    version: str = 'n'
    image_size: int = 640
    output_weights: str = 'weights/finetune_init.pt'
    cls_prior: float = 0.01
    strict_backbone_load: bool = True
    device: str = 'cpu'


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def _from_dict(cls, data, path='config'):
    """Build a dataclass from a dict, warning on unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data.keys()) - set(known)
    if unknown:
        logger.warning(
            f"Unknown keys ignored in {path}: {sorted(unknown)}")

    kwargs = {}
    for name, f in known.items():
        if name not in data or data[name] is None:
            continue
        value = data[name]
        if is_dataclass(f.type) or _is_dataclass_field(cls, name):
            sub_cls = _field_dataclass(cls, name)
            if isinstance(value, dict):
                kwargs[name] = _from_dict(
                    sub_cls, value, path=f"{path}.{name}")
                continue
        kwargs[name] = value
    return cls(**kwargs)


def _field_dataclass(cls, name):
    """Return the dataclass type of a field, or None."""
    for f in fields(cls):
        if f.name == name and is_dataclass(f.type):
            return f.type
    return None


def _is_dataclass_field(cls, name):
    return _field_dataclass(cls, name) is not None


def config_to_dict(cfg):
    """Turn a (possibly nested) dataclass into a plain dict."""
    out = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        out[f.name] = config_to_dict(value) if is_dataclass(value) \
            else value
    return out


def _load_yaml(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_train_config(path) -> TrainConfig:
    return _from_dict(TrainConfig, _load_yaml(path))


def load_eval_config(path) -> EvalConfig:
    return _from_dict(EvalConfig, _load_yaml(path))


def load_hdf5_build_config(path) -> Hdf5BuildConfig:
    return _from_dict(Hdf5BuildConfig, _load_yaml(path))


def load_export_config(path) -> ExportConfig:
    return _from_dict(ExportConfig, _load_yaml(path))


def load_finetune_config(path) -> FinetuneConfig:
    return _from_dict(FinetuneConfig, _load_yaml(path))

"""
Chargement et validation de la configuration YAML.

Utilisé par train.py et evaluate.py pour parser train.yaml / eval.yaml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    # Dataset
    dataset_dir: str
    num_classes: int
    image_size: int = 640
    augment: bool = True

    # Modèle
    version: str = 'n'        # 'n', 's', 'm', 'l', 'x'
    resume: Optional[str] = None  # chemin d'un checkpoint à reprendre

    # Optimisation
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    max_lr: float = 0.01
    min_lr: float = 0.0001
    warmup_epochs: float = 3.0
    momentum: float = 0.937
    weight_decay: float = 0.0005
    scheduler: str = 'cosine'  # 'cosine' ou 'linear'
    grad_clip: float = 10.0
    grad_accumulation_steps: int = 1   # accumulation de gradient sur N micro-batches

    # Loss gains
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5

    # Checkpoints
    checkpoint_dir: str = 'checkpoints'
    max_checkpoints: int = 5
    save_best_metric: str = 'map50'  # métrique pour sélectionner le "best"
    auto_resume: bool = True          # si True et `resume` non défini, charge le dernier epoch_*.pt

    # Divers
    device: str = 'cuda'       # 'cuda' | 'cpu' | 'cuda:0'
    seed: int = 0
    log_interval: int = 10     # afficher la loss toutes les N itérations
    val_interval: int = 1      # valider toutes les N epochs

    # Validation / NMS
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7


@dataclass
class EvalConfig:
    dataset_dir: str
    num_classes: int
    weights: str               # chemin du checkpoint à évaluer
    version: str = 'n'
    image_size: int = 640
    batch_size: int = 16
    num_workers: int = 4
    device: str = 'cuda'
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7
    split: str = 'test'        # 'test' ou 'train'


def _load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_train_config(path) -> TrainConfig:
    data = _load_yaml(path)
    # Filtre les clés inconnues pour plus de robustesse
    fields = {f for f in TrainConfig.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        print(f"[config] Clés ignorées: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in fields}
    return TrainConfig(**data)


def load_eval_config(path) -> EvalConfig:
    data = _load_yaml(path)
    fields = {f for f in EvalConfig.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        print(f"[config] Clés ignorées: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in fields}
    return EvalConfig(**data)

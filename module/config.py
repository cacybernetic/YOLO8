"""
Chargement et validation de la configuration YAML.

Utilisé par train.py et evaluate.py pour parser train.yaml / eval.yaml.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class TrainConfig:
    # Dataset
    dataset_dir: str
    num_classes: int
    image_size: int = 640
    augment: bool = True
    augment_params: Optional[Dict[str, Any]] = None  # voir DEFAULT_AUGMENT_PARAMS
    check_images: bool = True     # vérifie l'intégrité des images au démarrage (lent mais évite crashs)

    # Modèle
    version: str = 'n'        # 'n', 's', 'm', 'l', 'x'
    resume: Optional[str] = None  # chemin d'un checkpoint à reprendre (poids + optimizer + epoch)
    pretrained_weights: Optional[str] = None  # chemin de poids initiaux (poids seulement, epoch=0)
    freeze_feature_layers: bool = False   # gèle backbone + neck (fine-tuning seulement tête)

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

    # Historique d'entraînement
    # Le graphique 4 subplots (avg_loss, avg_box, avg_cls, avg_dfl) train/val
    # est régénéré à la fin de chaque epoch à ce chemin.
    history_plot_path: str = 'checkpoints/training_history.png'

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
    output_dir: str = 'results'      # dossier de sortie pour CSVs et figures
    class_names: Optional[list] = None  # noms des classes (optionnel, pour les plots)
    # Loss gains pour calcul des losses sur val (mêmes valeurs qu'en train)
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5


@dataclass
class InferConfig:
    weights: str                              # chemin du checkpoint à charger
    num_classes: int
    version: str = 'n'
    image_size: int = 640
    device: str = 'cuda'
    conf_threshold: float = 0.25              # plus élevé qu'en eval: on veut des boites visibles
    iou_threshold: float = 0.45
    class_names: Optional[list] = None        # liste des noms de classes (len == num_classes)
    line_thickness: int = 2
    font_scale: float = 0.5
    box_opacity: float = 0.75                 # opacité globale du graphisme des boites [0, 1]
    save_path: Optional[str] = None           # si défini, sauvegarde l'image annotée
    show: bool = True                         # affiche avec cv2.imshow


@dataclass
class ExportConfig:
    # Modèle
    weights: str                              # chemin du checkpoint (.pt) à convertir
    num_classes: int
    version: str = 'n'
    image_size: int = 640

    # Sortie
    output_path: str = 'weights/best.onnx'

    # Options d'export ONNX
    opset: int = 17                           # version d'opset ONNX (17 = bon défaut moderne)
    dynamic: bool = True                      # batch size dynamique
    simplify: bool = True                     # simplifier le graphe avec onnxsim (si installé)
    half: bool = False                        # export FP16 (utile GPU / mobile)

    # Validation post-export
    check: bool = True                        # vérifie le modèle ONNX avec onnx.checker
    verify: bool = True                       # compare sortie PyTorch vs ONNX Runtime (si installé)
    verify_tolerance: float = 1e-3            # tolérance max sur l'écart absolu

    # Divers
    device: str = 'cpu'                       # 'cpu' recommandé pour l'export, plus stable


@dataclass
class FinetuneConfig:
    """Configuration pour la construction d'un modèle fine-tunable.

    Le fine-tuning YOLOv8 s'appuie sur trois principes (cf. littérature transfer learning
    et pratiques Ultralytics):
      - Le backbone et le neck extraient des features génériques, réutilisables
        sur de nouvelles classes. On les conserve tels quels.
      - Les branches de régression de boites (`box`) et le DFL produisent
        des coordonnées géométriques, indépendantes du nombre de classes.
      - Seules les branches de classification (`cls`) dépendent de num_classes
        et doivent être réinitialisées avec la nouvelle taille de sortie.
    """
    # Modèle source
    pretrained_weights: str              # chemin du .pt pré-entraîné
    old_num_classes: int                 # nombre de classes du modèle source
    new_num_classes: int                 # nombre de classes du modèle cible
    version: str = 'n'                   # doit matcher celui du modèle source
    image_size: int = 640

    # Sortie
    output_weights: str = 'weights/finetune_init.pt'

    # Options d'initialisation
    cls_prior: float = 0.01              # prior pour l'init du biais cls (cf. Focal Loss)
    strict_backbone_load: bool = True    # exige que backbone+neck chargent sans clé manquante

    # Divers
    device: str = 'cpu'


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


def load_infer_config(path) -> InferConfig:
    data = _load_yaml(path)
    fields = {f for f in InferConfig.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        print(f"[config] Clés ignorées: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in fields}
    return InferConfig(**data)


def load_export_config(path) -> ExportConfig:
    data = _load_yaml(path)
    fields = {f for f in ExportConfig.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        print(f"[config] Clés ignorées: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in fields}
    return ExportConfig(**data)


def load_finetune_config(path) -> FinetuneConfig:
    data = _load_yaml(path)
    fields = {f for f in FinetuneConfig.__dataclass_fields__}
    unknown = set(data.keys()) - fields
    if unknown:
        print(f"[config] Clés ignorées: {sorted(unknown)}")
    data = {k: v for k, v in data.items() if k in fields}
    return FinetuneConfig(**data)

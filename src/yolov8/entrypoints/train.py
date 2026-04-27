"""
Script principal d'entraînement YOLOv8.

Usage :
    python -m yolov8.train --config configs/train.yaml

Gestion des checkpoints :
  - Sauvegarde à chaque fin d'epoch (model, optimizer, scheduler, epoch, best_metric)
  - Rotation FIFO : garde au max `max_checkpoints` fichiers, supprime les plus anciens
  - Sauvegarde séparée du meilleur modèle (best.pt) selon la métrique configurée
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov8.config import load_train_config, TrainConfig
from yolov8.dataset import YOLODataset
from yolov8.lossfn import ComputeLoss
from yolov8.metrics import MetricAccumulator, non_max_suppression
from yolov8.model import MyYolo
from loguru import logger

from yolov8.utils import print_model_summary, plot_training_history, setup_logging, build_val_targets


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Schedulers LR
# ---------------------------------------------------------------------------

class CosineLR:
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, num_steps):
        warmup_steps = int(max(warmup_epochs * num_steps, 100))
        decay_steps = int(total_epochs * num_steps - warmup_steps)
        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps)
        decay_lr = [
            min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * s / decay_steps))
            for s in range(1, decay_steps + 1)
        ]
        self.lrs = np.concatenate((warmup_lr, decay_lr))

    def step(self, global_step, optimizer):
        idx = min(global_step, len(self.lrs) - 1)
        for pg in optimizer.param_groups:
            pg['lr'] = self.lrs[idx]


class LinearLR:
    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, num_steps):
        warmup_steps = int(max(warmup_epochs * num_steps, 100))
        decay_steps = int(total_epochs * num_steps - warmup_steps)
        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps, endpoint=False)
        decay_lr = np.linspace(max_lr, min_lr, decay_steps)
        self.lrs = np.concatenate((warmup_lr, decay_lr))

    def step(self, global_step, optimizer):
        idx = min(global_step, len(self.lrs) - 1)
        for pg in optimizer.param_groups:
            pg['lr'] = self.lrs[idx]


def build_scheduler(name, max_lr, min_lr, warmup_epochs, total_epochs, num_steps):
    name = name.lower()
    if name == 'cosine':
        return CosineLR(max_lr, min_lr, warmup_epochs, total_epochs, num_steps)
    if name == 'linear':
        return LinearLR(max_lr, min_lr, warmup_epochs, total_epochs, num_steps)
    raise ValueError(f"Scheduler inconnu: {name}")


# ---------------------------------------------------------------------------
# Optimizer helper (weight-decay séparé sur bias/BN)
# ---------------------------------------------------------------------------

def build_optimizer(model, lr, momentum, weight_decay):
    p_decay, p_nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            p_nodecay.append(param)
        else:
            p_decay.append(param)
    return torch.optim.SGD([
        {'params': p_nodecay, 'weight_decay': 0.0},
        {'params': p_decay, 'weight_decay': weight_decay},
    ], lr=lr, momentum=momentum, nesterov=True)


def freeze_feature_layers(model):
    """Gèle le backbone et le neck du modèle pour un fine-tuning de la tête seule.

    Principe académique:
      Lorsque le dataset cible est petit ou ressemble fortement à celui du
      pré-entraînement, on peut figer les couches d'extraction de
      caractéristiques (backbone + neck) et n'entraîner que la tête de
      détection. Cela :
        - Réduit drastiquement le nombre de paramètres optimisés (moins de
          risque de sur-apprentissage sur un petit dataset).
        - Préserve les features génériques apprises sur le pré-entraînement.
        - Accélère l'entraînement (pas de backprop dans backbone+neck).

    Implémentation:
      On met requires_grad=False sur tous les paramètres de backbone et neck.
      Les modules BatchNorm contenus dedans sont aussi mis en mode eval() pour
      figer leurs statistiques running_mean/running_var — sinon, même avec
      requires_grad=False sur leurs poids, les statistiques seraient encore
      mises à jour durant le forward (comportement non désiré au fine-tuning).
    """
    n_frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith('backbone.') or name.startswith('neck.'):
            param.requires_grad = False
            n_frozen_params += param.numel()

    # Figer aussi les statistiques de BatchNorm
    # Note: cela ne désactive pas l'ajout de dropout hypothétique, mais notre
    # modèle n'en contient pas ; seule BatchNorm a un état interne.
    n_bn_frozen = 0
    for mod_name, module in model.named_modules():
        if not (mod_name.startswith('backbone.') or mod_name.startswith('neck.')):
            continue
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d)):
            module.eval()
            n_bn_frozen += 1

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone + Neck gelés: {n_frozen_params/1e6:.3f}M params "
                f"figés, {n_bn_frozen} couches BN mises en eval()")
    logger.info(f"Paramètres entraînables: {trainable/1e6:.3f}M / "
                f"{total/1e6:.3f}M  ({100*trainable/total:.1f}%)")
    return n_frozen_params


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _parse_epoch_num(path):
    """Extrait le numéro d'epoch depuis 'epoch_XXXX.pt'. -1 si illisible."""
    try:
        return int(path.stem.split('_')[-1])
    except (IndexError, ValueError):
        return -1


def list_checkpoints(ckpt_dir):
    """Liste les checkpoints d'epoch triés par numéro d'epoch croissant.

    On trie sur le numéro d'epoch (parsé depuis le nom de fichier) plutôt que
    sur le mtime: c'est plus robuste si un fichier a été touché/copié.
    `best.pt` est exclu (ne matche pas le pattern).
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []
    ckpts = [p for p in ckpt_dir.glob('epoch_*.pt') if _parse_epoch_num(p) >= 0]
    return sorted(ckpts, key=_parse_epoch_num)


def find_latest_checkpoint(ckpt_dir):
    """Retourne le chemin du dernier epoch_*.pt ou None."""
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def resolve_resume_path(cfg):
    """Détermine quel checkpoint charger en fonction de la config.

    - Si `cfg.resume` est un chemin existant -> on l'utilise.
    - Si `cfg.resume` est renseigné mais n'existe pas -> avertissement, pas de resume.
    - Sinon si `cfg.auto_resume` est True -> cherche le dernier dans checkpoint_dir.
    - Sinon -> None (démarrage from scratch).
    """
    if cfg.resume:
        p = Path(cfg.resume)
        if p.exists():
            return p
        logger.warning(f"'{cfg.resume}' n'existe pas, auto-resume tenté.")

    if cfg.auto_resume:
        latest = find_latest_checkpoint(cfg.checkpoint_dir)
        if latest is not None:
            logger.info(f"Auto-detect: dernier checkpoint trouvé -> {latest}")
            return latest
        logger.info(f"Aucun checkpoint dans '{cfg.checkpoint_dir}', démarrage from scratch.")

    return None


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)  # .replace() est atomique ET fonctionne sur Windows


def rotate_checkpoints(ckpt_dir, max_keep):
    ckpts = list_checkpoints(ckpt_dir)
    if len(ckpts) <= max_keep:
        return
    to_remove = ckpts[:len(ckpts) - max_keep]
    for p in to_remove:
        try:
            p.unlink()
            logger.info(f"Checkpoint supprimé (rotation): {p.name}")
        except OSError as e:
            logger.warning(f"Impossible de supprimer {p}: {e}")


def load_checkpoint_if_any(path, model, optimizer=None, device='cpu'):
    """Charge un checkpoint si `path` est fourni et existe.

    Retourne (epoch_start, best_metric). Si aucun checkpoint n'est chargé,
    retourne (0, -inf).
    """
    if not path:
        return 0, -float('inf')

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        logger.warning(f"Le fichier '{path}' n'existe pas. "
                       f"Entraînement démarré from scratch.")
        return 0, -float('inf')

    # weights_only=False est nécessaire pour charger l'optimizer et les
    # métadonnées (epoch, config, val_metrics...). Depuis PyTorch >= 2.6,
    # la valeur par défaut est True et refuse ce type de contenu.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # Anciennes versions de PyTorch qui n'ont pas l'argument weights_only
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"[resume] Impossible de lire le checkpoint '{path}': {e}"
        ) from e

    # Accepte soit notre dict complet, soit un state_dict nu
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
        has_meta = True
    else:
        state = ckpt
        has_meta = False

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"[resume] Les poids du checkpoint ne correspondent pas au modèle courant. "
            f"Vérifiez que `version` et `num_classes` sont identiques à ceux de la "
            f"sauvegarde.\n  Détail: {e}"
        ) from e

    if optimizer is not None and has_meta and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            logger.warning(f"Impossible de restaurer l'état de l'optimizer "
                           f"({e}). L'optimizer repart de zéro.")

    epoch_start = (ckpt.get('epoch', -1) + 1) if has_meta else 0
    best_metric = ckpt.get('best_metric', -float('inf')) if has_meta else -float('inf')

    logger.info(f"Reprise depuis '{path}' (epoch_start={epoch_start}, "
                f"best_metric={best_metric:.4f})")
    return epoch_start, best_metric


def load_pretrained_weights(path, model, device='cpu'):
    """Charge UNIQUEMENT les poids depuis un fichier (pas l'optimizer ni l'epoch).

    Différence fondamentale avec `load_checkpoint_if_any` :
      - `load_checkpoint_if_any` est pour REPRENDRE un entraînement interrompu :
        elle restaure l'optimizer, le scheduler implicite (via epoch_start) et
        le best_metric. L'entraînement continue comme si on n'avait jamais arrêté.
      - `load_pretrained_weights` est pour DÉMARRER un nouvel entraînement avec
        des poids pré-existants : on charge uniquement le state_dict du modèle,
        on repart de epoch=0 avec un optimizer frais et un best_metric vierge.

    Ce mode est utile pour:
      - Fine-tuning : charger un modèle issu de finetuning.py dont les têtes cls
        ont été ré-initialisées. On veut un nouvel apprentissage, pas une reprise.
      - Transfer learning : charger un modèle entraîné sur un autre dataset.
      - Initialisation à chaud : démarrer depuis un modèle de référence pour
        bénéficier de bonnes features initiales.

    Args:
        path: chemin du fichier .pt
        model: modèle cible
        device: device pour map_location

    Returns:
        True si les poids ont été chargés, False sinon.
    """
    if not path:
        return False

    weights_path = Path(path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"[pretrained] Fichier de poids pré-entraînés introuvable: {path}"
        )

    # Même gestion weights_only que pour les checkpoints complets.
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)

    # Accepte aussi bien un checkpoint complet qu'un state_dict nu.
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"[pretrained] Les poids ne correspondent pas au modèle courant. "
            f"Vérifiez que `version` et `num_classes` dans train.yaml matchent "
            f"ceux utilisés pour produire '{path}'.\n  Détail: {e}"
        ) from e

    # Message informatif avec l'origine si disponible (cas d'un checkpoint
    # produit par finetuning.py qui stocke la provenance).
    origin = ""
    if isinstance(ckpt, dict):
        src = ckpt.get('finetune_origin')
        old_nc = ckpt.get('finetune_old_num_classes')
        new_nc = ckpt.get('finetune_new_num_classes')
        if src is not None:
            origin = (f" (fine-tune: {old_nc} -> {new_nc} classes, "
                      f"source: {Path(src).name})")

    logger.info(f"Poids chargés depuis '{path}'{origin}")
    logger.info("Démarrage d'un nouvel entraînement "
                "(epoch=0, optimizer fresh, best_metric=-inf)")
    return True


# ---------------------------------------------------------------------------
# Une epoch d'entraînement
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler,
                    epoch, total_epochs, num_steps_per_epoch,
                    device, grad_clip, log_interval,
                    grad_accumulation_steps=1,
                    freeze_feature_layers=False):
    """Exécute une epoch d'entraînement.

    Args:
        freeze_feature_layers: si True, après `model.train()`, on remet
            explicitement les BatchNorm de backbone+neck en mode eval. C'est
            nécessaire car `model.train()` propage train() à tous les
            sous-modules, ce qui réactiverait la mise à jour des statistiques
            running_mean/running_var dans les couches gelées.
    """
    model.train()

    # Si freeze actif, on remet les BN du backbone+neck en eval() pour figer
    # leurs statistiques running_mean/running_var. Sans cela, les stats
    # seraient mises à jour à chaque forward même si les poids sont gelés.
    if freeze_feature_layers:
        for mod_name, module in model.named_modules():
            if not (mod_name.startswith('backbone.') or mod_name.startswith('neck.')):
                continue
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                   torch.nn.BatchNorm3d)):
                module.eval()

    running = {'box': 0.0, 'cls': 0.0, 'dfl': 0.0, 'total': 0.0, 'n': 0}
    t0 = time.time()

    accum = max(int(grad_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    has_pending_grads = False  # True dès qu'un .backward() a ajouté des gradients non encore appliqués

    pbar = tqdm(
        enumerate(loader),
        total=total_steps,
        desc=f"Epoch {epoch+1}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
    )

    for step, (images, targets, _paths) in pbar:
        global_step = epoch * num_steps_per_epoch + step
        scheduler.step(global_step, optimizer)

        images = images.to(device, non_blocking=True)

        outputs = model(images)
        loss_box, loss_cls, loss_dfl = loss_fn(outputs, targets)
        loss = loss_box + loss_cls + loss_dfl

        # Mise à l'échelle pour l'accumulation de gradient
        (loss / accum).backward()
        has_pending_grads = True

        # Pas d'optimisation seulement aux frontières d'accumulation.
        # Le reste (accumulation incomplète en fin d'epoch) est flushé APRÈS la boucle.
        if (step + 1) % accum == 0:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            has_pending_grads = False

        bs = images.size(0)
        running['box'] += loss_box.item() * bs
        running['cls'] += loss_cls.item() * bs
        running['dfl'] += loss_dfl.item() * bs
        running['total'] += loss.item() * bs
        running['n'] += bs

        # Moyennes courantes (running averages) depuis le début de l'epoch
        n_seen = max(running['n'], 1)
        avg_loss = running['total'] / n_seen
        avg_box = running['box'] / n_seen
        avg_cls = running['cls'] / n_seen
        avg_dfl = running['dfl'] / n_seen

        # Mise à jour de la barre de progression avec les moyennes
        pbar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.5f}",
            'loss': f"{avg_loss:.4f}",
            'box': f"{avg_box:.4f}",
            'cls': f"{avg_cls:.4f}",
            'dfl': f"{avg_dfl:.4f}",
        })

        if log_interval and ((step + 1) % log_interval == 0):
            pbar.write(
                f"  step {step+1}/{total_steps} "
                f"| lr {optimizer.param_groups[0]['lr']:.5f} "
                f"| avg_loss {avg_loss:.4f} "
                f"(box {avg_box:.4f} cls {avg_cls:.4f} dfl {avg_dfl:.4f})"
            )

    # Flush de fin d'epoch: si l'epoch s'est terminée au milieu d'un groupe
    # d'accumulation, on applique quand même ce qui a été accumulé pour ne pas
    # perdre ces gradients.
    if has_pending_grads:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        has_pending_grads = False

    n = max(running['n'], 1)
    elapsed = time.time() - t0
    logger.success(f"epoch {epoch+1} terminée en {elapsed:.1f}s "
                   f"| box={running['box']/n:.4f} cls={running['cls']/n:.4f} "
                   f"dfl={running['dfl']/n:.4f} total={running['total']/n:.4f}")
    return {k: v / n for k, v in running.items() if k != 'n'}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(model, loader, device, image_size, conf_threshold, iou_threshold,
             loss_fn=None):
    """Évaluation sur le set de validation.

    Calcule:
      - les métriques de détection (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
      - les pertes box / cls / dfl / total si `loss_fn` est fourni

    Pour pouvoir tracer les courbes train/val des pertes en parallèle, on doit
    obligatoirement passer une `loss_fn` ici (sinon les pertes val resteront à 0).
    Le calcul est fait avec `torch.no_grad()` car on est en évaluation, mais
    `loss_fn` reste utilisable car `ComputeLoss` n'a pas besoin du graphe pour
    calculer la valeur scalaire de la perte (on n'appelle jamais `.backward()`).
    """
    model.eval()
    accumulator = MetricAccumulator(device=device)

    # Accumulateurs de pertes pondérées par batch_size (cohérent avec train_one_epoch)
    losses_sum = {'box': 0.0, 'cls': 0.0, 'dfl': 0.0, 'total': 0.0}
    n_samples = 0

    pbar = tqdm(loader, desc="[val]", leave=False, dynamic_ncols=True)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        # En eval la tête renvoie (inference_tensor, raw_outputs).
        # `raw_outputs` est exactement ce dont la loss a besoin.
        out = model(images)
        if isinstance(out, tuple):
            inference_out, raw_outputs = out
        else:
            inference_out, raw_outputs = out, None

        # --- Calcul des pertes sur la validation ---
        if loss_fn is not None and raw_outputs is not None:
            try:
                lb, lc, ld = loss_fn(raw_outputs, targets)
                bs = images.size(0)
                losses_sum['box']   += float(lb.item()) * bs
                losses_sum['cls']   += float(lc.item()) * bs
                losses_sum['dfl']   += float(ld.item()) * bs
                losses_sum['total'] += float((lb + lc + ld).item()) * bs
                n_samples += bs
            except Exception as e:
                pbar.write(f"[val] échec calcul loss sur ce batch: {e}")

        # --- NMS + accumulation des métriques de détection ---
        preds = non_max_suppression(
            inference_out,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        targets_per_image = build_val_targets(images, targets, image_size, device)
        accumulator.update(preds, targets_per_image)

    results = accumulator.compute()

    # Moyennes des pertes (mêmes clés que train_one_epoch pour symétrie)
    if n_samples > 0:
        for k, v in losses_sum.items():
            results[k] = v / n_samples
    else:
        # Pas de loss_fn fourni: on remplit avec 0 pour rester compatible
        for k in losses_sum:
            results[k] = 0.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers train.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Niveau de log loguru')
    args = parser.parse_args()

    # Configuration du logging unifié pour tout le projet
    setup_logging(level=args.log_level)

    cfg: TrainConfig = load_train_config(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available()
                          or not cfg.device.startswith('cuda') else 'cpu')
    logger.info(f"device={device}")

    # --- Data ---
    train_ds = YOLODataset(cfg.dataset_dir, split='train',
                           image_size=cfg.image_size, augment=cfg.augment,
                           augment_params=cfg.augment_params,
                           check_images=cfg.check_images)
    val_ds = YOLODataset(cfg.dataset_dir, split='test',
                         image_size=cfg.image_size, augment=False,
                         check_images=cfg.check_images)
    logger.info(f"Données: train={len(train_ds)} | val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith('cuda')),
        collate_fn=YOLODataset.collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith('cuda')),
        collate_fn=YOLODataset.collate_fn, drop_last=False,
    )
    num_steps_per_epoch = max(len(train_loader), 1)

    # --- Model ---
    model = MyYolo(version=cfg.version, num_classes=cfg.num_classes,
                   input_size=cfg.image_size).to(device)
    # S'assurer que les strides sont sur le bon device
    model.head.stride = model.head.stride.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Modèle: YOLOv8-{cfg.version} | {n_params:.3f}M params | "
                f"strides={model.head.stride.tolist()}")
    # NOTE: le résumé du modèle est imprimé plus loin, APRÈS le chargement des
    # poids et le freeze éventuel. À ce stade, la colonne "Trainable" de
    # torchinfo et le total des params entraînables reflètent alors l'état réel
    # du modèle tel qu'il sera entraîné. Afficher le résumé ici produirait des
    # chiffres de "trainable" trompeurs en cas de freeze_feature_layers=true.

    # --- Loss ---
    loss_params = {'box': cfg.box_gain, 'cls': cfg.cls_gain, 'dfl': cfg.dfl_gain}
    loss_fn = ComputeLoss(model, loss_params)

    # --- Stratégie de démarrage ---
    # Trois modes possibles, évalués dans cet ordre de priorité:
    #
    #   MODE 1 (resume): si cfg.resume est renseigné et existe, ou si
    #     auto_resume trouve un epoch_*.pt dans checkpoint_dir, on REPREND
    #     l'entraînement (poids + optimizer + epoch + best_metric).
    #
    #   MODE 2 (pretrained_weights): si cfg.pretrained_weights est renseigné,
    #     on DÉMARRE UN NOUVEL ENTRAÎNEMENT en chargeant uniquement les poids.
    #     L'optimizer est frais, on repart de epoch=0. Cas d'usage principal:
    #     démarrer un fine-tuning depuis le .pt produit par finetuning.py.
    #
    #   MODE 3 (from scratch): aucun des deux ci-dessus → initialisation
    #     aléatoire du modèle.
    #
    # Le MODE 1 a la priorité sur le MODE 2 car si on a à la fois un checkpoint
    # de reprise disponible ET des poids pré-entraînés, on veut reprendre le
    # dernier entraînement en cours, pas redémarrer depuis la source.
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(cfg)

    startup_mode = None  # 'resume' | 'pretrained' | 'scratch'
    epoch_start = 0
    best_metric = -float('inf')

    if resume_path is not None:
        # MODE 1: reprise complète
        startup_mode = 'resume'
        epoch_start, best_metric = load_checkpoint_if_any(
            resume_path, model, optimizer=None, device=device
        )
    elif cfg.pretrained_weights:
        # MODE 2: démarrage avec poids initiaux (fine-tune ou transfer learning)
        startup_mode = 'pretrained'
        load_pretrained_weights(cfg.pretrained_weights, model, device=device)
        # epoch_start=0, best_metric=-inf (déjà initialisés ci-dessus)
    else:
        # MODE 3: from scratch
        startup_mode = 'scratch'
        logger.info("Aucun poids initial fourni. Entraînement from scratch.")

    # --- Freeze (optionnel) — doit être avant build_optimizer ---
    # Rationale: un paramètre gelé (requires_grad=False) est ignoré par
    # build_optimizer, qui ne créera donc pas d'états d'optimisation inutiles
    # pour backbone/neck. Cela économise de la mémoire et garantit qu'aucune
    # mise à jour n'est appliquée, même en cas de bug.
    if cfg.freeze_feature_layers:
        freeze_feature_layers(model)

    # --- Optim (après freeze pour n'enregistrer que les params entraînables) ---
    optimizer = build_optimizer(model, lr=cfg.max_lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(cfg.scheduler, cfg.max_lr, cfg.min_lr,
                                cfg.warmup_epochs, cfg.epochs,
                                num_steps_per_epoch)

    # Chargement différé de l'état de l'optimizer UNIQUEMENT en mode resume.
    # En mode pretrained_weights ou scratch, on démarre avec un optimizer frais
    # même si le fichier source contenait un état d'optimizer.
    # En mode freeze, les param_groups sont différents (moins de paramètres),
    # donc l'état de l'optimizer stocké devient incompatible et doit être ignoré.
    if startup_mode == 'resume' and not cfg.freeze_feature_layers:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                logger.info("État de l'optimizer restauré")
        except Exception as e:
            logger.warning(f"Impossible de restaurer l'optimizer ({e}). "
                           f"Poursuite avec un optimizer fresh.")
    elif startup_mode == 'resume' and cfg.freeze_feature_layers:
        logger.info("État de l'optimizer IGNORÉ "
                    "(freeze_feature_layers actif, groupes de params incompatibles)")

    best_path = ckpt_dir / 'best.pt'

    # --- Résumé final du modèle ---
    # Affiché en dernier, APRÈS chargement des poids et freeze éventuel.
    # À ce stade, `requires_grad` reflète l'état réel de chaque paramètre,
    # donc la colonne "Trainable" de torchinfo et le total des params
    # entraînables sont cohérents avec ce qui sera réellement optimisé.
    logger.info(f"Résumé du modèle (mode={startup_mode}, "
                f"freeze={cfg.freeze_feature_layers}):")
    print_model_summary(model, input_size=(1, 3, cfg.image_size, cfg.image_size),
                        device=device)

    # --- Training loop ---
    if cfg.grad_accumulation_steps > 1:
        effective_bs = cfg.batch_size * cfg.grad_accumulation_steps
        logger.info(f"grad_accumulation_steps={cfg.grad_accumulation_steps} "
                    f"| effective batch size = {effective_bs}")

    # Historique des pertes train/val pour le tracé en fin d'epoch.
    # Stocké dans un dict simple, sérialisable, restauré depuis le checkpoint
    # en cas de reprise pour conserver la continuité visuelle.
    history = {
        'epochs_train': [], 'train_loss': [], 'train_box': [],
        'train_cls': [],    'train_dfl':  [],
        'epochs_val':   [], 'val_loss':   [], 'val_box':   [],
        'val_cls':     [],  'val_dfl':    [],
    }
    # Si on reprend depuis un checkpoint qui contenait déjà un historique,
    # on le récupère pour que le graphique reste cohérent.
    if startup_mode == 'resume' and resume_path is not None:
        try:
            ckpt_for_history = torch.load(resume_path, map_location='cpu',
                                          weights_only=False)
            if isinstance(ckpt_for_history, dict) and 'history' in ckpt_for_history:
                history = ckpt_for_history['history']
                logger.info(f"Historique restauré depuis le checkpoint "
                            f"({len(history.get('epochs_train', []))} epochs train, "
                            f"{len(history.get('epochs_val', []))} epochs val)")
        except Exception as e:
            logger.warning(f"Impossible de restaurer l'historique ({e}). "
                           f"Repart d'un historique vide.")

    history_path = Path(cfg.history_plot_path)

    for epoch in range(epoch_start, cfg.epochs):
        logger.info(f"=== Epoch {epoch+1}/{cfg.epochs} ===")
        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            epoch, cfg.epochs, num_steps_per_epoch,
            device, cfg.grad_clip, cfg.log_interval,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
            freeze_feature_layers=cfg.freeze_feature_layers,
        )

        # On ajoute toujours l'epoch d'entraînement à l'historique,
        # même si on ne valide pas à cette epoch (val_interval > 1).
        history['epochs_train'].append(epoch + 1)
        history['train_loss'].append(float(train_stats.get('total', 0.0)))
        history['train_box'].append(float(train_stats.get('box', 0.0)))
        history['train_cls'].append(float(train_stats.get('cls', 0.0)))
        history['train_dfl'].append(float(train_stats.get('dfl', 0.0)))

        val_metrics = {}
        if (epoch + 1) % cfg.val_interval == 0:
            val_metrics = validate(
                model, val_loader, device, cfg.image_size,
                cfg.conf_threshold, cfg.iou_threshold,
                loss_fn=loss_fn,    # IMPORTANT: passe loss_fn pour calcul des pertes val
            )
            logger.info(f"[val] P={val_metrics['precision']:.4f} "
                        f"R={val_metrics['recall']:.4f} "
                        f"mAP@0.5={val_metrics['map50']:.4f} "
                        f"mAP@0.5:0.95={val_metrics['map']:.4f} "
                        f"| total={val_metrics.get('total', 0):.4f} "
                        f"(box {val_metrics.get('box', 0):.4f} "
                        f"cls {val_metrics.get('cls', 0):.4f} "
                        f"dfl {val_metrics.get('dfl', 0):.4f})")

            # Accumulation de l'historique val (uniquement aux epochs où on valide)
            history['epochs_val'].append(epoch + 1)
            history['val_loss'].append(float(val_metrics.get('total', 0.0)))
            history['val_box'].append(float(val_metrics.get('box', 0.0)))
            history['val_cls'].append(float(val_metrics.get('cls', 0.0)))
            history['val_dfl'].append(float(val_metrics.get('dfl', 0.0)))

        # Tracé du graphique d'historique (régénéré à chaque epoch).
        # Encapsulé dans try/except pour ne JAMAIS casser l'entraînement à cause
        # d'un problème de matplotlib (filesystem plein, backend manquant, etc.)
        try:
            plot_training_history(history, history_path)
        except Exception as e:
            logger.warning(f"Échec du tracé de l'historique: {e}")

        # Checkpoint d'epoch (inclut l'historique pour la continuité en cas de reprise)
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_metric,
            'config': cfg.__dict__,
            'train_stats': train_stats,
            'val_metrics': val_metrics,
            'history': history,
        }
        epoch_ckpt = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
        save_checkpoint(state, epoch_ckpt)
        logger.info(f"Checkpoint sauvegardé: {epoch_ckpt.name}")
        rotate_checkpoints(ckpt_dir, cfg.max_checkpoints)

        # Best model
        if val_metrics:
            metric = val_metrics.get(cfg.save_best_metric, None)
            if metric is not None and metric > best_metric:
                best_metric = metric
                state['best_metric'] = best_metric
                save_checkpoint(state, best_path)
                logger.success(f"Nouveau meilleur {cfg.save_best_metric}="
                               f"{best_metric:.4f} -> {best_path}")

    logger.success("Entraînement terminé.")
    logger.info(f"Graphique d'historique: {history_path}")
    if best_metric > -float('inf'):
        logger.info(f"Best {cfg.save_best_metric}: {best_metric:.4f}")


if __name__ == '__main__':
    main()

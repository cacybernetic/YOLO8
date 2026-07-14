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

from yolov8.utils import (print_model_summary, plot_training_history,
                          setup_logging, build_val_targets, safe_torch_load,
                          ModelEMA)


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Schedulers LR (warmup par groupe + warmup momentum, convention YOLOv8)
# ---------------------------------------------------------------------------

class BaseLR:
    """Scheduler par step avec warmup complet.

    Pendant le warmup (convention Ultralytics) :
      - la LR des poids monte linéairement de 0 vers max_lr ;
      - la LR des biais DESCEND de warmup_bias_lr (0.1) vers max_lr — les
        biais peuvent bouger vite dès le départ sans déstabiliser les poids ;
      - le momentum monte de warmup_momentum (0.8) vers momentum (0.937).

    Garde-fous petits datasets : le warmup est plafonné pour ne jamais
    consommer tout le budget de steps (sinon decay_steps <= 0 → crash
    np.linspace de l'ancienne implémentation, ou LR figée à max_lr).
    """

    def __init__(self, max_lr, min_lr, warmup_epochs, total_epochs, num_steps,
                 momentum=0.937, warmup_momentum=0.8, warmup_bias_lr=0.1):
        total_steps = max(int(round(total_epochs * num_steps)), 1)
        warmup_steps = int(max(warmup_epochs * num_steps, 100))
        self.warmup_steps = max(min(warmup_steps, total_steps - 1), 0)
        self.decay_steps = max(total_steps - self.warmup_steps, 1)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.momentum = momentum
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr

    def _decayed_lr(self, s):
        """LR après warmup, s dans [0, decay_steps]."""
        raise NotImplementedError

    def step(self, global_step, optimizer):
        if global_step < self.warmup_steps:
            f = (global_step + 1) / self.warmup_steps
            momentum = (self.warmup_momentum +
                        (self.momentum - self.warmup_momentum) * f)
            for pg in optimizer.param_groups:
                start_lr = self.warmup_bias_lr if pg.get('is_bias') else 0.0
                pg['lr'] = start_lr + (self.max_lr - start_lr) * f
                if 'momentum' in pg:
                    pg['momentum'] = momentum
        else:
            s = min(global_step - self.warmup_steps, self.decay_steps)
            lr = self._decayed_lr(s)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                if 'momentum' in pg:
                    pg['momentum'] = self.momentum


class CosineLR(BaseLR):
    def _decayed_lr(self, s):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * s / self.decay_steps))


class LinearLR(BaseLR):
    def _decayed_lr(self, s):
        return self.max_lr + (self.min_lr - self.max_lr) * (s / self.decay_steps)


def build_scheduler(name, max_lr, min_lr, warmup_epochs, total_epochs, num_steps,
                    momentum=0.937, warmup_momentum=0.8, warmup_bias_lr=0.1):
    name = name.lower()
    kwargs = dict(momentum=momentum, warmup_momentum=warmup_momentum,
                  warmup_bias_lr=warmup_bias_lr)
    if name == 'cosine':
        return CosineLR(max_lr, min_lr, warmup_epochs, total_epochs,
                        num_steps, **kwargs)
    if name == 'linear':
        return LinearLR(max_lr, min_lr, warmup_epochs, total_epochs,
                        num_steps, **kwargs)
    raise ValueError(f"Scheduler inconnu: {name}")


# ---------------------------------------------------------------------------
# Optimizer helper (weight-decay séparé sur bias/BN, groupe biais marqué)
# ---------------------------------------------------------------------------

def build_optimizer(model, lr, momentum, weight_decay):
    """Trois groupes de paramètres (convention YOLOv8) :
      0. poids >= 2D          -> weight decay
      1. poids 1D (BN, etc.)  -> pas de decay
      2. biais                -> pas de decay, marqué `is_bias` pour que le
                                 scheduler leur applique le warmup LR dédié.
    """
    p_bias, p_nodecay, p_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias'):
            p_bias.append(param)
        elif param.ndim <= 1:
            p_nodecay.append(param)
        else:
            p_decay.append(param)
    return torch.optim.SGD([
        {'params': p_decay, 'weight_decay': weight_decay, 'is_bias': False},
        {'params': p_nodecay, 'weight_decay': 0.0, 'is_bias': False},
        {'params': p_bias, 'weight_decay': 0.0, 'is_bias': True},
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


def load_checkpoint_if_any(path, model, device='cpu'):
    """Charge un checkpoint si `path` est fourni et existe.

    Retourne (epoch_start, best_metric, ckpt). `ckpt` (dict ou None) est
    retourné pour que l'appelant restaure optimizer / historique / EMA sans
    relire le fichier depuis le disque. Si aucun checkpoint n'est chargé,
    retourne (0, -inf, None).
    """
    if not path:
        return 0, -float('inf'), None

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        logger.warning(f"Le fichier '{path}' n'existe pas. "
                       f"Entraînement démarré from scratch.")
        return 0, -float('inf'), None

    try:
        ckpt = safe_torch_load(ckpt_path, map_location=device)
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

    epoch_start = (ckpt.get('epoch', -1) + 1) if has_meta else 0
    best_metric = ckpt.get('best_metric', -float('inf')) if has_meta else -float('inf')

    logger.info(f"Reprise depuis '{path}' (epoch_start={epoch_start}, "
                f"best_metric={best_metric:.4f})")
    return epoch_start, best_metric, (ckpt if has_meta else None)


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

    ckpt = safe_torch_load(weights_path, map_location=device)

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

def _apply_optimizer_step(model, optimizer, scaler, grad_clip, ema):
    """Applique un pas d'optimisation (avec ou sans AMP) + mise à jour EMA."""
    if scaler is not None:
        if grad_clip and grad_clip > 0:
            # unscale_ AVANT le clipping: sinon on clipperait des gradients
            # multipliés par le scale factor du GradScaler.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(model)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler,
                    epoch, total_epochs, num_steps_per_epoch,
                    device, grad_clip, log_interval,
                    grad_accumulation_steps=1,
                    freeze_feature_layers=False,
                    scaler=None, ema=None, amp_enabled=False):
    """Exécute une epoch d'entraînement.

    Args:
        freeze_feature_layers: si True, après `model.train()`, on remet
            explicitement les BatchNorm de backbone+neck en mode eval. C'est
            nécessaire car `model.train()` propage train() à tous les
            sous-modules, ce qui réactiverait la mise à jour des statistiques
            running_mean/running_var dans les couches gelées.
        scaler: torch.amp.GradScaler (ou None si AMP désactivé).
        ema: ModelEMA mis à jour après chaque optimizer.step (ou None).
        amp_enabled: active autocast pour le forward + la loss.
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

    # Accumulateurs GPU [box, cls, dfl, total] : une seule synchronisation
    # host<->device par intervalle de log au lieu de 4 .item() par step.
    running = torch.zeros(4, device=device)
    n_seen = 0
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
        bs = images.size(0)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_box, loss_cls, loss_dfl = loss_fn(outputs, targets)
            loss = loss_box + loss_cls + loss_dfl

        # ×bs : la loss est normalisée par target_scores_sum (par cible) ;
        # Ultralytics rétropropage `loss.sum() * batch_size` et les
        # hyperparamètres (max_lr=0.01, gains 7.5/0.5/1.5) sont calibrés pour
        # cette échelle de gradient. Sans ce facteur, les gradients seraient
        # ~bs× plus faibles que la recette de référence.
        scaled = loss * bs / accum
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        has_pending_grads = True

        # Pas d'optimisation seulement aux frontières d'accumulation.
        # Le reste (accumulation incomplète en fin d'epoch) est flushé APRÈS la boucle.
        if (step + 1) % accum == 0:
            _apply_optimizer_step(model, optimizer, scaler, grad_clip, ema)
            has_pending_grads = False

        # Accumulation des pertes (valeurs par cible, non ×bs) pondérées par bs
        running += torch.stack((loss_box.detach(), loss_cls.detach(),
                                loss_dfl.detach(), loss.detach())) * bs
        n_seen += bs

        if log_interval and ((step + 1) % log_interval == 0):
            # Synchronisation unique pour le log
            avg_box, avg_cls, avg_dfl, avg_loss = (running / max(n_seen, 1)).tolist()
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'lr': f"{lr_now:.5f}",
                'loss': f"{avg_loss:.4f}",
                'box': f"{avg_box:.4f}",
                'cls': f"{avg_cls:.4f}",
                'dfl': f"{avg_dfl:.4f}",
            })
            pbar.write(
                f"  step {step+1}/{total_steps} "
                f"| lr {lr_now:.5f} "
                f"| avg_loss {avg_loss:.4f} "
                f"(box {avg_box:.4f} cls {avg_cls:.4f} dfl {avg_dfl:.4f})"
            )

    # Flush de fin d'epoch: si l'epoch s'est terminée au milieu d'un groupe
    # d'accumulation, on applique quand même ce qui a été accumulé pour ne pas
    # perdre ces gradients.
    if has_pending_grads:
        _apply_optimizer_step(model, optimizer, scaler, grad_clip, ema)
        has_pending_grads = False

    n = max(n_seen, 1)
    avg_box, avg_cls, avg_dfl, avg_loss = (running / n).tolist()
    elapsed = time.time() - t0
    logger.success(f"epoch {epoch+1} terminée en {elapsed:.1f}s "
                   f"| box={avg_box:.4f} cls={avg_cls:.4f} "
                   f"dfl={avg_dfl:.4f} total={avg_loss:.4f}")
    return {'box': avg_box, 'cls': avg_cls, 'dfl': avg_dfl, 'total': avg_loss}


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
        # Pas de try/except large ici: une erreur récurrente (device, dtype,
        # shapes) produirait des courbes val silencieusement à zéro. On
        # préfère échouer explicitement (fail fast).
        if loss_fn is not None and raw_outputs is not None:
            lb, lc, ld = loss_fn(raw_outputs, targets)
            bs = images.size(0)
            losses_sum['box']   += float(lb.item()) * bs
            losses_sum['cls']   += float(lc.item()) * bs
            losses_sum['dfl']   += float(ld.item()) * bs
            losses_sum['total'] += float((lb + lc + ld).item()) * bs
            n_samples += bs

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
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA indisponible, fallback sur CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(cfg.device)
    logger.info(f"device={device}")

    # Autotuning cuDNN: gain de perf notable car nos shapes sont fixes
    # (letterbox carré). À désactiver via cudnn_benchmark: false si besoin
    # d'un déterminisme strict.
    if device.type == 'cuda' and cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # --- Split de validation ---
    # La sélection du meilleur modèle DOIT se faire sur un split val dédié:
    # sélectionner sur le test set (puis évaluer dessus) est une fuite
    # méthodologique qui rend les métriques finales optimistes.
    val_split = cfg.val_split
    if val_split is None:
        if (Path(cfg.dataset_dir) / 'val' / 'images').is_dir():
            val_split = 'val'
        else:
            val_split = 'test'
            logger.warning(
                "Aucun split 'val/' trouvé: la validation (et la sélection du "
                "best) se fera sur 'test/'. ATTENTION: les métriques finales "
                "de yleval sur ce même split seront optimistes. Créez un "
                "split val dédié pour un protocole propre.")
    logger.info(f"Split de validation: '{val_split}'")

    # --- Data ---
    train_ds = YOLODataset(cfg.dataset_dir, split='train',
                           image_size=cfg.image_size, augment=cfg.augment,
                           augment_params=cfg.augment_params,
                           check_images=cfg.check_images)
    val_ds = YOLODataset(cfg.dataset_dir, split=val_split,
                         image_size=cfg.image_size, augment=False,
                         check_images=cfg.check_images)
    logger.info(f"Données: train={len(train_ds)} | val={len(val_ds)}")

    pin_memory = (device.type == 'cuda')
    persistent = cfg.num_workers > 0
    # Generator dédié: rend l'ordre de shuffle reproductible avec le seed
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)

    def make_train_loader():
        return DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=pin_memory,
            collate_fn=YOLODataset.collate_fn, drop_last=True,
            persistent_workers=persistent, generator=data_generator,
        )

    train_loader = make_train_loader()
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory,
        collate_fn=YOLODataset.collate_fn, drop_last=False,
        persistent_workers=persistent,
    )
    num_steps_per_epoch = max(len(train_loader), 1)

    # --- Model ---
    # (head.stride est un buffer: il suit .to(device) automatiquement)
    model = MyYolo(version=cfg.version, num_classes=cfg.num_classes,
                   input_size=cfg.image_size).to(device)
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
    resume_ckpt = None   # dict du checkpoint (un seul torch.load pour tout)

    if resume_path is not None:
        # MODE 1: reprise complète
        startup_mode = 'resume'
        epoch_start, best_metric, resume_ckpt = load_checkpoint_if_any(
            resume_path, model, device=device
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
                                num_steps_per_epoch,
                                momentum=cfg.momentum,
                                warmup_momentum=cfg.warmup_momentum,
                                warmup_bias_lr=cfg.warmup_bias_lr)

    # Restauration de l'état de l'optimizer UNIQUEMENT en mode resume
    # (depuis le checkpoint déjà chargé — pas de relecture disque).
    # En mode pretrained_weights ou scratch, on démarre avec un optimizer frais
    # même si le fichier source contenait un état d'optimizer.
    # En mode freeze, les param_groups sont différents (moins de paramètres),
    # donc l'état de l'optimizer stocké devient incompatible et doit être ignoré.
    if startup_mode == 'resume' and not cfg.freeze_feature_layers:
        if resume_ckpt is not None and 'optimizer' in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt['optimizer'])
                logger.info("État de l'optimizer restauré")
            except Exception as e:
                logger.warning(f"Impossible de restaurer l'optimizer ({e}). "
                               f"Poursuite avec un optimizer fresh.")
    elif startup_mode == 'resume' and cfg.freeze_feature_layers:
        logger.info("État de l'optimizer IGNORÉ "
                    "(freeze_feature_layers actif, groupes de params incompatibles)")

    # --- EMA (Exponential Moving Average des poids) ---
    # Créée APRÈS le chargement des poids pour partir de l'état courant.
    # La validation et best.pt utilisent l'EMA: métrique nettement plus stable.
    ema = None
    if cfg.ema:
        ema = ModelEMA(model, decay=cfg.ema_decay, tau=cfg.ema_tau)
        if (startup_mode == 'resume' and resume_ckpt is not None
                and 'ema' in resume_ckpt):
            try:
                ema.load_state_dict(resume_ckpt['ema'],
                                    updates=resume_ckpt.get('ema_updates', 0))
                logger.info(f"EMA restaurée (updates={ema.updates})")
            except Exception as e:
                logger.warning(f"Impossible de restaurer l'EMA ({e}). "
                               f"EMA réinitialisée depuis les poids courants.")

    # --- AMP (mixed precision) ---
    amp_enabled = bool(cfg.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=True) if amp_enabled else None
    if cfg.amp and not amp_enabled:
        logger.info("AMP demandé mais device CPU: entraînement en FP32.")
    elif amp_enabled:
        logger.info("Mixed precision (AMP) activée")

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
    # on le récupère pour que le graphique reste cohérent (le checkpoint est
    # déjà en mémoire: pas de relecture disque).
    if startup_mode == 'resume' and resume_ckpt is not None and 'history' in resume_ckpt:
        history = resume_ckpt['history']
        logger.info(f"Historique restauré depuis le checkpoint "
                    f"({len(history.get('epochs_train', []))} epochs train, "
                    f"{len(history.get('epochs_val', []))} epochs val)")

    # Le checkpoint de reprise n'est plus nécessaire: on libère la mémoire
    # (il contient une copie complète des poids + optimizer).
    resume_ckpt = None

    history_path = Path(cfg.history_plot_path)

    mosaic_closed = False
    epochs_no_improve = 0  # compteur pour l'early stopping (patience)

    for epoch in range(epoch_start, cfg.epochs):
        logger.info(f"=== Epoch {epoch+1}/{cfg.epochs} ===")

        # --- close_mosaic: désactive mosaic+mixup sur les dernières epochs ---
        # (convention YOLOv8: finir l'entraînement sur des images "réelles"
        # améliore la précision finale). Le DataLoader est recréé car avec
        # persistent_workers les workers gardent une copie du dataset.
        if (cfg.close_mosaic > 0 and not mosaic_closed
                and epoch >= cfg.epochs - cfg.close_mosaic
                and (train_ds.aug.get('mosaic', 0) > 0
                     or train_ds.aug.get('mixup', 0) > 0)):
            logger.info(f"close_mosaic: mosaic/mixup désactivés pour les "
                        f"{cfg.epochs - epoch} dernières epochs")
            train_ds.aug['mosaic'] = 0.0
            train_ds.aug['mixup'] = 0.0
            mosaic_closed = True
            train_loader = make_train_loader()

        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            epoch, cfg.epochs, num_steps_per_epoch,
            device, cfg.grad_clip, cfg.log_interval,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
            freeze_feature_layers=cfg.freeze_feature_layers,
            scaler=scaler, ema=ema, amp_enabled=amp_enabled,
        )

        # On ajoute toujours l'epoch d'entraînement à l'historique,
        # même si on ne valide pas à cette epoch (val_interval > 1).
        history['epochs_train'].append(epoch + 1)
        history['train_loss'].append(float(train_stats.get('total', 0.0)))
        history['train_box'].append(float(train_stats.get('box', 0.0)))
        history['train_cls'].append(float(train_stats.get('cls', 0.0)))
        history['train_dfl'].append(float(train_stats.get('dfl', 0.0)))

        val_metrics = {}
        is_best = False
        if (epoch + 1) % cfg.val_interval == 0:
            # La validation porte sur les poids EMA (plus stables) quand
            # l'EMA est active — c'est aussi ce que sauvegarde best.pt.
            eval_model = ema.ema if ema is not None else model
            val_metrics = validate(
                eval_model, val_loader, device, cfg.image_size,
                cfg.conf_threshold, cfg.iou_threshold,
                loss_fn=loss_fn,    # IMPORTANT: passe loss_fn pour calcul des pertes val
            )
            logger.info(f"[val{'(EMA)' if ema is not None else ''}] "
                        f"P={val_metrics['precision']:.4f} "
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

            # Mise à jour du best AVANT la sauvegarde du checkpoint d'epoch:
            # ainsi epoch_NNNN.pt embarque toujours le best_metric à jour et
            # une reprise ne peut pas écraser best.pt avec un modèle moins bon.
            metric = val_metrics.get(cfg.save_best_metric, None)
            if metric is not None and metric > best_metric:
                best_metric = metric
                is_best = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        # Tracé du graphique d'historique (régénéré à chaque epoch).
        # Encapsulé dans try/except pour ne JAMAIS casser l'entraînement à cause
        # d'un problème de matplotlib (filesystem plein, backend manquant, etc.)
        try:
            plot_training_history(history, history_path)
        except Exception as e:
            logger.warning(f"Échec du tracé de l'historique: {e}")

        # Checkpoint d'epoch (inclut l'historique et l'EMA pour la reprise)
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
        if ema is not None:
            state['ema'] = ema.ema.state_dict()
            state['ema_updates'] = ema.updates
        epoch_ckpt = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
        save_checkpoint(state, epoch_ckpt)
        logger.info(f"Checkpoint sauvegardé: {epoch_ckpt.name}")
        rotate_checkpoints(ckpt_dir, cfg.max_checkpoints)

        # Best model: best.pt stocke les poids ÉVALUÉS (EMA si active) sous la
        # clé 'model', pour que yleval / ylinfer / ylexport utilisent
        # directement les poids qui ont produit la meilleure métrique.
        if is_best:
            best_state = dict(state)
            if ema is not None:
                best_state['model'] = ema.ema.state_dict()
            save_checkpoint(best_state, best_path)
            logger.success(f"Nouveau meilleur {cfg.save_best_metric}="
                           f"{best_metric:.4f} -> {best_path}")

        # Early stopping (patience en nombre de validations sans amélioration)
        if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
            logger.warning(
                f"Early stopping: pas d'amélioration de "
                f"{cfg.save_best_metric} depuis {cfg.patience} validations "
                f"(best={best_metric:.4f}).")
            break

    logger.success("Entraînement terminé.")
    logger.info(f"Graphique d'historique: {history_path}")
    if best_metric > -float('inf'):
        logger.info(f"Best {cfg.save_best_metric}: {best_metric:.4f}")


if __name__ == '__main__':
    main()

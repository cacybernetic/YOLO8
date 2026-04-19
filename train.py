"""
Script principal d'entraînement YOLOv8.

Usage :
    python -m module.train --config module/train.yaml

Gestion des checkpoints :
  - Sauvegarde à chaque fin d'epoch (model, optimizer, scheduler, epoch, best_metric)
  - Rotation FIFO : garde au max `max_checkpoints` fichiers, supprime les plus anciens
  - Sauvegarde séparée du meilleur modèle (best.pt) selon la métrique configurée
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.config import load_train_config, TrainConfig
from module.dataset import YOLODataset
from module.lossfn import ComputeLoss
from module.metrics import MetricAccumulator, non_max_suppression
from module.model import MyYolo


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
        print(f"[resume] ATTENTION: '{cfg.resume}' n'existe pas, auto-resume tenté.")

    if cfg.auto_resume:
        latest = find_latest_checkpoint(cfg.checkpoint_dir)
        if latest is not None:
            print(f"[resume] Auto-detect: dernier checkpoint trouvé -> {latest}")
            return latest
        print(f"[resume] Aucun checkpoint dans '{cfg.checkpoint_dir}', démarrage from scratch.")

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
            print(f"  [ckpt] Supprimé: {p.name}")
        except OSError as e:
            print(f"  [ckpt] Impossible de supprimer {p}: {e}")


def load_checkpoint_if_any(path, model, optimizer=None, device='cpu'):
    """Charge un checkpoint si `path` est fourni et existe.

    Retourne (epoch_start, best_metric). Si aucun checkpoint n'est chargé,
    retourne (0, -inf).
    """
    if not path:
        return 0, -float('inf')

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"[resume] ATTENTION: le fichier '{path}' n'existe pas. "
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
            print(f"[resume] ATTENTION: impossible de restaurer l'état de l'optimizer "
                  f"({e}). L'optimizer repart de zéro.")

    epoch_start = (ckpt.get('epoch', -1) + 1) if has_meta else 0
    best_metric = ckpt.get('best_metric', -float('inf')) if has_meta else -float('inf')

    print(f"[resume] Reprise depuis '{path}' (epoch_start={epoch_start}, "
          f"best_metric={best_metric:.4f})")
    return epoch_start, best_metric


# ---------------------------------------------------------------------------
# Une epoch d'entraînement
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler,
                    epoch, total_epochs, num_steps_per_epoch,
                    device, grad_clip, log_interval,
                    grad_accumulation_steps=1):
    model.train()
    running = {'box': 0.0, 'cls': 0.0, 'dfl': 0.0, 'total': 0.0, 'n': 0}
    t0 = time.time()

    accum = max(int(grad_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)

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

        # Pas d'optimizer uniquement à la fin d'un groupe d'accumulation
        # (ou à la toute fin de l'epoch, pour ne pas perdre les derniers micro-batches)
        is_accum_boundary = ((step + 1) % accum == 0) or ((step + 1) == total_steps)
        if is_accum_boundary:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        running['box'] += loss_box.item() * bs
        running['cls'] += loss_cls.item() * bs
        running['dfl'] += loss_dfl.item() * bs
        running['total'] += loss.item() * bs
        running['n'] += bs

        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.5f}",
            'loss': f"{loss.item():.4f}",
            'box': f"{loss_box.item():.4f}",
            'cls': f"{loss_cls.item():.4f}",
            'dfl': f"{loss_dfl.item():.4f}",
        })

        if log_interval and ((step + 1) % log_interval == 0):
            pbar.write(
                f"  step {step+1}/{total_steps} "
                f"| lr {optimizer.param_groups[0]['lr']:.5f} "
                f"| loss {loss.item():.4f} "
                f"(box {loss_box.item():.4f} cls {loss_cls.item():.4f} dfl {loss_dfl.item():.4f})"
            )

    n = max(running['n'], 1)
    elapsed = time.time() - t0
    print(f"  [train] epoch {epoch+1} done in {elapsed:.1f}s "
          f"| box={running['box']/n:.4f} cls={running['cls']/n:.4f} "
          f"dfl={running['dfl']/n:.4f} total={running['total']/n:.4f}")
    return {k: v / n for k, v in running.items() if k != 'n'}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def build_val_targets(images, targets_dict, image_size, device):
    """Reconstruit, par image, un tenseur GT (n_gt, 5) = [cls, x1, y1, x2, y2] en pixels."""
    bs = images.size(0)
    idx = targets_dict['idx']
    cls = targets_dict['cls']
    box = targets_dict['box']  # (N, 4) cx, cy, w, h normalisé

    per_image = []
    for i in range(bs):
        mask = (idx == i)
        if mask.sum() == 0:
            per_image.append(torch.zeros((0, 5), device=device))
            continue
        c = cls[mask].view(-1, 1)
        b = box[mask]
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        x1 = (cx - w / 2) * image_size
        y1 = (cy - h / 2) * image_size
        x2 = (cx + w / 2) * image_size
        y2 = (cy + h / 2) * image_size
        gt = torch.stack((c.view(-1), x1, y1, x2, y2), dim=1).to(device)
        per_image.append(gt)
    return per_image


@torch.no_grad()
def validate(model, loader, device, image_size, conf_threshold, iou_threshold):
    model.eval()
    accumulator = MetricAccumulator(device=device)

    pbar = tqdm(loader, desc="[val]", leave=False, dynamic_ncols=True)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        # En eval la tête renvoie (inference_tensor, raw_outputs)
        out = model(images)
        if isinstance(out, tuple):
            inference_out = out[0]
        else:
            inference_out = out

        preds = non_max_suppression(
            inference_out,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        targets_per_image = build_val_targets(images, targets, image_size, device)
        accumulator.update(preds, targets_per_image)

    results = accumulator.compute()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers train.yaml')
    args = parser.parse_args()

    cfg: TrainConfig = load_train_config(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available()
                          or not cfg.device.startswith('cuda') else 'cpu')
    print(f"[setup] device={device}")

    # --- Data ---
    train_ds = YOLODataset(cfg.dataset_dir, split='train',
                           image_size=cfg.image_size, augment=cfg.augment)
    val_ds = YOLODataset(cfg.dataset_dir, split='test',
                         image_size=cfg.image_size, augment=False)
    print(f"[data] train={len(train_ds)} | val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=YOLODataset.collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=YOLODataset.collate_fn, drop_last=False,
    )
    num_steps_per_epoch = max(len(train_loader), 1)

    # --- Model ---
    model = MyYolo(version=cfg.version, num_classes=cfg.num_classes,
                   input_size=cfg.image_size).to(device)
    # S'assurer que les strides sont sur le bon device
    model.head.stride = model.head.stride.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] YOLOv8-{cfg.version} | {n_params:.3f}M params | strides={model.head.stride.tolist()}")

    # --- Loss ---
    loss_params = {'box': cfg.box_gain, 'cls': cfg.cls_gain, 'dfl': cfg.dfl_gain}
    loss_fn = ComputeLoss(model, loss_params)

    # --- Optim ---
    optimizer = build_optimizer(model, lr=cfg.max_lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(cfg.scheduler, cfg.max_lr, cfg.min_lr,
                                cfg.warmup_epochs, cfg.epochs,
                                num_steps_per_epoch)

    # --- Resume ---
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(cfg)
    epoch_start, best_metric = load_checkpoint_if_any(
        resume_path, model, optimizer, device=device
    )

    best_path = ckpt_dir / 'best.pt'

    # --- Training loop ---
    if cfg.grad_accumulation_steps > 1:
        effective_bs = cfg.batch_size * cfg.grad_accumulation_steps
        print(f"[train] grad_accumulation_steps={cfg.grad_accumulation_steps} "
              f"| effective batch size = {effective_bs}")

    for epoch in range(epoch_start, cfg.epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            epoch, cfg.epochs, num_steps_per_epoch,
            device, cfg.grad_clip, cfg.log_interval,
            grad_accumulation_steps=cfg.grad_accumulation_steps,
        )

        val_metrics = {}
        if (epoch + 1) % cfg.val_interval == 0:
            val_metrics = validate(
                model, val_loader, device, cfg.image_size,
                cfg.conf_threshold, cfg.iou_threshold
            )
            print(f"  [val] P={val_metrics['precision']:.4f} "
                  f"R={val_metrics['recall']:.4f} "
                  f"mAP@0.5={val_metrics['map50']:.4f} "
                  f"mAP@0.5:0.95={val_metrics['map']:.4f}")

        # Checkpoint d'epoch
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_metric,
            'config': cfg.__dict__,
            'train_stats': train_stats,
            'val_metrics': val_metrics,
        }
        epoch_ckpt = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
        save_checkpoint(state, epoch_ckpt)
        print(f"  [ckpt] Sauvegardé: {epoch_ckpt.name}")
        rotate_checkpoints(ckpt_dir, cfg.max_checkpoints)

        # Best model
        if val_metrics:
            metric = val_metrics.get(cfg.save_best_metric, None)
            if metric is not None and metric > best_metric:
                best_metric = metric
                state['best_metric'] = best_metric
                save_checkpoint(state, best_path)
                print(f"  [best] Nouveau meilleur {cfg.save_best_metric}={best_metric:.4f} -> {best_path}")

    print("\n[done] Entraînement terminé.")
    if best_metric > -float('inf'):
        print(f"  Best {cfg.save_best_metric}: {best_metric:.4f}")


if __name__ == '__main__':
    main()

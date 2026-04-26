"""
Script d'évaluation complet d'un modèle YOLOv8 entraîné.

Conformément au document `yolov8_metriques.tex`, ce script calcule :

  Métriques par classe (CSV `per_class.csv`) :
    - n_gt, n_pred, Precision, Recall, F1-Score, AP@0.5, AP@0.5:0.95, Cls Loss

  Métriques globales (CSV `global.csv`) :
    - mAP@0.5, mAP@0.5:0.95
    - Precision/Recall/F1 macro et micro
    - Box Loss, Cls Loss, DFL Loss, Total Loss
    - IoU moyen sur les TP
    - Seuil de confiance optimal (max F1 macro) + F1 à ce seuil
    - Comptes globaux TP/FP/FN/GT/Pred

  Visualisations (PNG dans `<output_dir>/figures/`) :
    - Courbe Précision-Rappel par classe et moyenne
    - Courbe F1-Confiance avec seuil optimal marqué
    - Matrice de confusion (N+1 x N+1, normalisée et brute)

Usage :
    python -m module.evaluate --config module/eval.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from module.config import load_eval_config, EvalConfig
from module.dataset import YOLODataset
from module.lossfn import ComputeLoss
from module.metrics import non_max_suppression
from module.metrics_eval import (
    box_iou,
    build_confusion_matrix,
    build_global_csv,
    build_per_class_csv,
    compute_ap_per_class,
    find_best_f1_threshold,
    match_predictions_to_gt,
    metrics_at_threshold,
    plot_confusion_matrix,
    plot_f1_confidence,
    plot_pr_curves,
)
from module.model import MyYolo
from module.train import build_val_targets
from module.utils import print_model_summary, setup_logging


# ---------------------------------------------------------------------------
# Boucle d'évaluation : un seul passage sur le dataset, on agrège tout
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions_and_losses(model, loader, device, image_size,
                                    conf_threshold, iou_threshold,
                                    loss_fn, num_classes, iou_v):
    """Parcourt le dataset une seule fois et accumule :
      - les prédictions filtrées par NMS, par image
      - les GT, par image
      - les TP/FP par seuil IoU pour chaque prédiction (matching greedy COCO)
      - les pertes (box, cls, dfl) cumulées
      - la cls_loss BCE par classe (proxy basé sur les confidences)
      - la somme des IoU sur les TP (pour calculer l'IoU moyen reporté)

    Cette approche en un seul passage est essentielle car on ne peut pas se
    permettre de re-faire l'inférence à chaque seuil de confiance ou seuil IoU.
    On stocke à la place toutes les paires (prediction, TP/FP par seuil)
    pour pouvoir interpoler les courbes a posteriori.
    """
    model.eval()
    n_iou = len(iou_v)

    all_tp_matrices = []   # liste de (n_pred_img, n_iou)
    all_conf = []          # (n_pred_img,)
    all_pred_cls = []      # (n_pred_img,)
    all_target_cls = []    # (n_gt_img,)

    # Pour la matrice de confusion : on garde les boites brutes par image
    # car le filtrage par seuil de conf se fera dans build_confusion_matrix
    # avec le seuil optimal trouvé en post-traitement.
    all_preds_cm = []      # liste de (preds_xyxy, pred_cls, pred_conf)
    all_gts_cm = []        # liste de (gts_xyxy, gt_cls)

    # Accumulateurs de pertes (pondérées par batch_size pour moyenner correctement)
    losses_sum = {'box': 0.0, 'cls': 0.0, 'dfl': 0.0, 'total': 0.0}
    n_samples = 0

    # Somme IoU des TP au seuil 0.5 (pour la métrique "IoU moyen" reportée
    # dans le CSV global, indicateur de la qualité de localisation)
    iou_sum_tp = 0.0
    iou_count_tp = 0

    pbar = tqdm(loader, desc="[eval]", leave=False, dynamic_ncols=True)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)

        out = model(images)
        if isinstance(out, tuple):
            inference_out, raw_outputs = out
        else:
            inference_out, raw_outputs = out, None

        # --- Calcul des pertes sur ce batch (si raw_outputs est dispo) ---
        # Important : la tête de YOLOv8 retourne (inference_out, raw_outputs)
        # en mode eval, ce qui nous permet de calculer les mêmes pertes qu'en
        # train sans avoir à repasser par model.train().
        if raw_outputs is not None:
            try:
                lb, lc, ld = loss_fn(raw_outputs, targets)
                bs = images.size(0)
                losses_sum['box']   += float(lb.item()) * bs
                losses_sum['cls']   += float(lc.item()) * bs
                losses_sum['dfl']   += float(ld.item()) * bs
                losses_sum['total'] += float((lb + lc + ld).item()) * bs
                n_samples += bs
            except Exception as e:
                pbar.write(f"[warn] échec calcul loss sur ce batch: {e}")

        # --- NMS pour obtenir les prédictions finales ---
        preds_per_image = non_max_suppression(
            inference_out,
            confidence_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        # --- GT par image en pixels (xyxy) ---
        targets_per_image = build_val_targets(images, targets, image_size, device)

        # --- Matching pour TP/FP/FN par image ---
        for preds, gt in zip(preds_per_image, targets_per_image):
            preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
            gt = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

            preds_xyxy = preds[:, :4] if preds.size else np.zeros((0, 4))
            pred_conf = preds[:, 4] if preds.size else np.zeros(0)
            pred_cls = preds[:, 5].astype(int) if preds.size else np.zeros(0, dtype=int)

            gts_xyxy = gt[:, 1:5] if gt.size else np.zeros((0, 4))
            gt_cls = gt[:, 0].astype(int) if gt.size else np.zeros(0, dtype=int)

            tp = match_predictions_to_gt(
                preds_xyxy, pred_cls, pred_conf,
                gts_xyxy, gt_cls,
                iou_thresholds=iou_v,
            )
            all_tp_matrices.append(tp)
            all_conf.append(pred_conf)
            all_pred_cls.append(pred_cls)
            all_target_cls.append(gt_cls)

            all_preds_cm.append((preds_xyxy, pred_cls, pred_conf))
            all_gts_cm.append((gts_xyxy, gt_cls))

            # IoU moyen sur les TP @ 0.5 (qualité de localisation)
            if preds_xyxy.shape[0] > 0 and gts_xyxy.shape[0] > 0 and tp[:, 0].any():
                iou_mat = box_iou(preds_xyxy, gts_xyxy)
                cls_match = pred_cls[:, None] == gt_cls[None, :]
                iou_mat = np.where(cls_match, iou_mat, 0.0)
                tp_indices = np.where(tp[:, 0])[0]
                for i in tp_indices:
                    iou_max = float(iou_mat[i].max()) if iou_mat[i].size else 0.0
                    if iou_max > 0:
                        iou_sum_tp += iou_max
                        iou_count_tp += 1

    # --- Concaténation finale ---
    if all_tp_matrices:
        tp_matrix = np.concatenate(all_tp_matrices, axis=0)
        conf = np.concatenate(all_conf, axis=0)
        pred_cls = np.concatenate(all_pred_cls, axis=0)
    else:
        tp_matrix = np.zeros((0, n_iou), dtype=bool)
        conf = np.zeros(0)
        pred_cls = np.zeros(0, dtype=int)
    target_cls = np.concatenate(all_target_cls, axis=0) if all_target_cls else np.zeros(0, dtype=int)

    # Moyennes des pertes
    if n_samples > 0:
        losses = {k: v / n_samples for k, v in losses_sum.items()}
    else:
        losses = {k: 0.0 for k in losses_sum}

    # IoU moyen sur les TP
    iou_mean = iou_sum_tp / iou_count_tp if iou_count_tp > 0 else 0.0

    # Cls loss par classe : proxy BCE basé sur les confidences des prédictions
    # filtrées par NMS.
    #   - TP : cls_loss = -log(conf)        (on aurait voulu prédire avec conf=1)
    #   - FP : cls_loss = -log(1 - conf)    (on aurait voulu prédire avec conf=0)
    # Cette approximation est cohérente avec la BCE per-class et permet
    # d'identifier rapidement les classes "difficiles" pour le modèle.
    eps = 1e-9
    cls_loss_per_class = np.zeros(num_classes, dtype=np.float64)
    cls_loss_count = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(conf)):
        c = int(pred_cls[i])
        if 0 <= c < num_classes:
            if tp_matrix[i, 0]:
                cls_loss_per_class[c] += -float(np.log(max(conf[i], eps)))
            else:
                cls_loss_per_class[c] += -float(np.log(max(1.0 - conf[i], eps)))
            cls_loss_count[c] += 1
    cls_loss_per_class_dict = {}
    for c in range(num_classes):
        if cls_loss_count[c] > 0:
            cls_loss_per_class_dict[c] = float(cls_loss_per_class[c] / cls_loss_count[c])
        else:
            cls_loss_per_class_dict[c] = 0.0

    return {
        'tp_matrix': tp_matrix,
        'conf': conf,
        'pred_cls': pred_cls,
        'target_cls': target_cls,
        'all_preds_cm': all_preds_cm,
        'all_gts_cm': all_gts_cm,
        'losses': losses,
        'cls_loss_per_class': cls_loss_per_class_dict,
        'iou_mean': iou_mean,
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg: EvalConfig):
    # Détection automatique du device si CUDA non dispo
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA indisponible, fallback sur CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(cfg.device)
    logger.info(f"device={device}")

    # Dossier de sortie pour CSVs et figures
    output_dir = Path(cfg.output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Résultats dans: {output_dir}")

    # --- Dataset + DataLoader ---
    ds = YOLODataset(cfg.dataset_dir, split=cfg.split,
                     image_size=cfg.image_size, augment=False)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=YOLODataset.collate_fn, drop_last=False,
    )
    logger.info(f"Données: {cfg.split}={len(ds)}")

    # --- Modèle + chargement des poids ---
    model = MyYolo(version=cfg.version, num_classes=cfg.num_classes,
                   input_size=cfg.image_size).to(device)
    model.head.stride = model.head.stride.to(device)

    weights_path = Path(cfg.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Poids introuvables: {weights_path}")
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"unexpected keys: {len(unexpected)}")
    logger.info(f"Poids chargés: {weights_path}")
    print_model_summary(model, input_size=(1, 3, cfg.image_size, cfg.image_size),
                        device=device)

    # --- Loss (mêmes gains qu'en train pour cohérence) ---
    loss_params = {'box': cfg.box_gain, 'cls': cfg.cls_gain, 'dfl': cfg.dfl_gain}
    loss_fn = ComputeLoss(model, loss_params)

    # --- Seuils IoU pour mAP@0.5:0.95 ---
    # Convention COCO : 10 seuils espacés de 0.05 entre 0.5 et 0.95 (inclus)
    iou_v = np.linspace(0.5, 0.95, 10)

    # --- Collecte (un seul passage sur le dataset) ---
    logger.info("Boucle d'évaluation...")
    data = collect_predictions_and_losses(
        model, loader, device, cfg.image_size,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        loss_fn=loss_fn,
        num_classes=cfg.num_classes,
        iou_v=iou_v,
    )

    # --- Calcul AP par classe + courbes interpolées ---
    logger.info("Calcul des AP par classe...")
    per_class = compute_ap_per_class(
        tp_matrix=data['tp_matrix'],
        conf=data['conf'],
        pred_cls=data['pred_cls'],
        target_cls=data['target_cls'],
        num_classes=cfg.num_classes,
    )

    # --- Seuil de confiance optimal (max F1 macro) ---
    # Le seuil optimal est celui qui maximise la moyenne arithmétique des F1
    # sur toutes les classes. C'est la convention de Ultralytics et c'est ce
    # que recommande le doc LaTeX dans la section sur la courbe F1-Confidence.
    px = np.linspace(0, 1, 1000)
    best_conf, best_f1 = find_best_f1_threshold(per_class, px)
    logger.info(f"Seuil de confiance optimal: {best_conf:.4f} "
                f"(F1 macro = {best_f1:.4f})")

    # Métriques par classe au seuil optimal pour les CSVs
    metrics_at_thr = metrics_at_threshold(per_class, px, best_conf)

    # --- Comptes globaux TP/FP/FN au seuil optimal pour le micro-averaging ---
    n_total_gt = int(data['target_cls'].size)
    n_total_pred = int(data['pred_cls'].size)
    keep = data['conf'] >= best_conf
    tp_filtered = data['tp_matrix'][keep, 0]  # @ IoU 0.5
    n_total_tp = int(tp_filtered.sum())
    n_total_fp = int((~tp_filtered).sum())
    n_total_fn = max(0, n_total_gt - n_total_tp)

    # --- Résolution des noms de classes ---
    if cfg.class_names and len(cfg.class_names) == cfg.num_classes:
        class_names = [str(n) for n in cfg.class_names]
    else:
        class_names = [f'class_{i}' for i in range(cfg.num_classes)]

    # === CSVs ===
    logger.info("Génération des CSVs...")
    df_per_class = build_per_class_csv(
        per_class, metrics_at_thr, class_names,
        cls_loss_per_class=data['cls_loss_per_class'],
    )
    df_global = build_global_csv(
        per_class, metrics_at_thr, data['losses'],
        iou_mean=data['iou_mean'],
        best_conf_threshold=best_conf,
        best_f1=best_f1,
        n_total_gt=n_total_gt,
        n_total_pred=n_total_pred,
        n_total_tp=n_total_tp,
        n_total_fp=n_total_fp,
        n_total_fn=n_total_fn,
    )

    per_class_csv = output_dir / 'per_class.csv'
    global_csv = output_dir / 'global.csv'
    df_per_class.to_csv(per_class_csv, index=False)
    df_global.to_csv(global_csv, index=False)
    logger.success(f"CSV écrit: {per_class_csv}")
    logger.success(f"CSV écrit: {global_csv}")

    # === Figures matplotlib ===
    logger.info("Génération des figures...")
    plot_pr_curves(per_class, class_names, figures_dir / 'pr_curve.png')
    plot_f1_confidence(per_class, class_names, figures_dir / 'f1_confidence.png')

    # Matrice de confusion : on l'utilise au seuil de conf optimal pour
    # une lecture cohérente avec les autres métriques rapportées.
    logger.info("Calcul de la matrice de confusion...")
    cm = build_confusion_matrix(
        data['all_preds_cm'], data['all_gts_cm'],
        num_classes=cfg.num_classes,
        iou_threshold=0.45,
        conf_threshold=best_conf,
    )
    plot_confusion_matrix(cm, class_names,
                          figures_dir / 'confusion_matrix_normalized.png',
                          normalize=True)
    plot_confusion_matrix(cm, class_names,
                          figures_dir / 'confusion_matrix.png',
                          normalize=False)
    logger.success(f"Figures écrites dans: {figures_dir}/")
    for fname in ['pr_curve.png', 'f1_confidence.png',
                  'confusion_matrix.png', 'confusion_matrix_normalized.png']:
        logger.info(f"  - {fname}")

    # === Résumé console ===
    # Note: on utilise `print` ici (pas logger) pour les tableaux pandas, car
    # loguru préfixe chaque ligne avec un timestamp ce qui rendrait la mise en
    # forme tabulaire illisible. Même justification que pour torchinfo.summary.
    print("\n" + "=" * 80)
    print("Résultats globaux")
    print("=" * 80)
    print(df_global.to_string(index=False))

    # Top et bottom 5 classes par AP@0.5 (utile pour identifier rapidement
    # les classes problématiques)
    if len(df_per_class) > 0:
        print("\nTop 5 classes par AP@0.5:")
        print(df_per_class.nlargest(5, 'ap50').to_string(index=False))
        if len(df_per_class) > 5:
            print("\nBottom 5 classes par AP@0.5:")
            print(df_per_class.nsmallest(5, 'ap50').to_string(index=False))

    return {
        'per_class': df_per_class,
        'global': df_global,
        'best_conf': best_conf,
        'output_dir': str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation YOLOv8 — toutes les métriques du doc LaTeX, "
                    "avec CSVs et figures.",
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers eval.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Niveau de log loguru')
    args = parser.parse_args()

    # Configuration du logging unifié
    setup_logging(level=args.log_level)

    cfg = load_eval_config(args.config)
    evaluate(cfg)


if __name__ == '__main__':
    main()

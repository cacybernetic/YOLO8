"""
Métriques d'évaluation détaillées pour YOLOv8.

Implémente les métriques décrites dans le document LaTeX de référence :
  - TP / FP / FN par classe et par seuil IoU
  - Precision, Recall, F1-Score par classe
  - AP@0.5, AP@0.5:0.95 par classe (intégration COCO 101 points)
  - Macro et micro moyennes (mAP, mP, mR, mF1)
  - Matrice de confusion N+1 x N+1 (la classe N représente le background)
  - Courbe Précision-Rappel par classe
  - Courbe F1-Confiance avec seuil optimal

Toutes les fonctions de plotting utilisent matplotlib en backend non-interactif
('Agg'), ce qui les rend utilisables sur des serveurs headless.

Référence académique :
  - Padilla et al., "A Survey on Performance Metrics for Object-Detection Algorithms" (2020)
  - COCO evaluation protocol (Lin et al., 2014)
  - Document interne `yolov8_metriques.tex`
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # backend non-interactif, compatible serveur headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calcul d'IoU (cohérent avec module/metrics.py mais autoportant)
# ---------------------------------------------------------------------------

def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matricielle entre deux ensembles de boites au format xyxy.

    Args:
        boxes_a: (N, 4) numpy
        boxes_b: (M, 4) numpy
    Returns:
        (N, M) matrice IoU.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


# ---------------------------------------------------------------------------
# Statistiques par image (TP/FP/FN par seuil IoU, par classe)
# ---------------------------------------------------------------------------

@dataclass
class PerImageStats:
    """Stats brutes par image, agrégées ensuite pour calculer toutes les métriques.

    Pour chaque prédiction:
      - tp_matrix[i, k] = True si la pred i est un TP au seuil iou_v[k]
      - conf[i] = score de confiance
      - pred_cls[i] = classe prédite
    Pour chaque GT:
      - target_cls[j] = classe vraie
    """
    tp_matrix: np.ndarray   # (n_pred, n_iou_thresh) bool
    conf: np.ndarray        # (n_pred,)
    pred_cls: np.ndarray    # (n_pred,)
    target_cls: np.ndarray  # (n_gt,)


def match_predictions_to_gt(
    preds_xyxy: np.ndarray,         # (n_pred, 4)
    pred_cls: np.ndarray,           # (n_pred,)
    pred_conf: np.ndarray,          # (n_pred,)
    gts_xyxy: np.ndarray,           # (n_gt, 4)
    gt_cls: np.ndarray,             # (n_gt,)
    iou_thresholds: np.ndarray,     # (n_iou,)
) -> np.ndarray:
    """Pour chaque prédiction, détermine si c'est un TP à chaque seuil IoU.

    Algorithme COCO standard :
      1. Trier les prédictions par confiance décroissante.
      2. Pour chaque prédiction (de la plus confiante à la moins) :
         - Trouver la meilleure GT non encore matchée, de même classe,
           dont l'IoU dépasse le seuil.
         - Si trouvée : TP, marquer la GT comme matchée.
         - Sinon : FP (implicite: tp_matrix[i, k] = False).

    Une GT non matchée par aucune prédiction devient un FN.
    Cela correspond à la définition du document LaTeX :
      "TP : prédiction correspondant à un GT, classe correcte, IoU >= seuil"

    Returns:
        tp_matrix: (n_pred, n_iou_thresh) bool
    """
    n_pred = preds_xyxy.shape[0]
    n_iou = len(iou_thresholds)
    tp_matrix = np.zeros((n_pred, n_iou), dtype=bool)

    if n_pred == 0 or gts_xyxy.shape[0] == 0:
        return tp_matrix

    # IoU entre toutes les paires (n_pred, n_gt)
    iou_mat = box_iou(preds_xyxy, gts_xyxy)
    # Masque de compatibilité de classe
    cls_match = pred_cls[:, None] == gt_cls[None, :]
    iou_mat = np.where(cls_match, iou_mat, 0.0)

    # Pour chaque seuil IoU, on fait un matching greedy par confiance
    # décroissante. Une GT ne peut être matchée qu'à une seule prédiction
    # (la plus confiante qui dépasse le seuil).
    order = np.argsort(-pred_conf)  # indices triés par confiance décroissante
    for k, iou_thr in enumerate(iou_thresholds):
        gt_matched = np.zeros(gts_xyxy.shape[0], dtype=bool)
        for i in order:
            ious_i = iou_mat[i]
            # On retire les GT déjà matchées
            ious_i_avail = np.where(gt_matched, -1.0, ious_i)
            j_best = int(np.argmax(ious_i_avail))
            if ious_i_avail[j_best] >= iou_thr:
                tp_matrix[i, k] = True
                gt_matched[j_best] = True
    return tp_matrix


# ---------------------------------------------------------------------------
# Calcul de l'AP (méthode COCO 101 points)
# ---------------------------------------------------------------------------

def compute_ap_per_class(
    tp_matrix: np.ndarray,    # (N, n_iou) bool, concatenation sur tout le dataset
    conf: np.ndarray,         # (N,)
    pred_cls: np.ndarray,     # (N,)
    target_cls: np.ndarray,   # (M,)
    num_classes: int,
    eps: float = 1e-9,
) -> Dict[int, dict]:
    """Calcule AP, P, R, F1 par classe.

    Méthodologie COCO :
      - Pour chaque classe, trier les prédictions par confiance décroissante.
      - Calculer la courbe Précision-Rappel cumulée.
      - AP = aire sous la courbe via interpolation 101 points (COCO standard).

    Returns:
        dict[cls_id] -> {
            'ap50': AP au seuil IoU 0.5,
            'ap5095': AP moyenne sur les seuils 0.5:0.05:0.95 (10 valeurs),
            'p_curve': (1000,) precision interpolée sur grille de confiance,
            'r_curve': (1000,) recall interpolée,
            'f1_curve': (1000,) F1 interpolée,
            'n_gt': nombre de GT pour cette classe,
            'n_pred': nombre de prédictions pour cette classe,
        }
    """
    # Tri décroissant par confiance (global, mais on filtrera par classe ensuite)
    order = np.argsort(-conf)
    tp_matrix = tp_matrix[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    px = np.linspace(0, 1, 1000)  # grille de confiance pour les courbes
    results: Dict[int, dict] = {}

    for c in range(num_classes):
        mask_pred = pred_cls == c
        n_pred_c = int(mask_pred.sum())
        n_gt_c = int(np.sum(target_cls == c))

        # Skip si aucune GT ET aucune prédiction
        if n_gt_c == 0 and n_pred_c == 0:
            continue

        if n_gt_c == 0:
            # Que des FP, AP non définie -> 0
            results[c] = {
                'ap50': 0.0, 'ap5095': 0.0,
                'p_curve': np.zeros_like(px),
                'r_curve': np.zeros_like(px),
                'f1_curve': np.zeros_like(px),
                'n_gt': 0, 'n_pred': n_pred_c,
            }
            continue

        if n_pred_c == 0:
            # Aucune prédiction de cette classe -> tous FN
            results[c] = {
                'ap50': 0.0, 'ap5095': 0.0,
                'p_curve': np.zeros_like(px),
                'r_curve': np.zeros_like(px),
                'f1_curve': np.zeros_like(px),
                'n_gt': n_gt_c, 'n_pred': 0,
            }
            continue

        tp_c = tp_matrix[mask_pred]   # (n_pred_c, n_iou)
        conf_c = conf[mask_pred]      # (n_pred_c,)

        # Cumuls
        tp_cum = np.cumsum(tp_c, axis=0).astype(np.float64)  # (n_pred_c, n_iou)
        fp_cum = np.cumsum(~tp_c, axis=0).astype(np.float64)
        recall_cum = tp_cum / (n_gt_c + eps)
        precision_cum = tp_cum / (tp_cum + fp_cum + eps)

        # AP via interpolation 101 points (COCO)
        n_iou = tp_c.shape[1]
        ap_per_iou = np.zeros(n_iou)
        for k in range(n_iou):
            r = recall_cum[:, k]
            p = precision_cum[:, k]
            # Précision interpolée monotone (envelope)
            p_envelope = np.flip(np.maximum.accumulate(np.flip(p)))
            # Échantillonnage 101 points sur [0, 1]
            x_eval = np.linspace(0, 1, 101)
            p_interp = np.interp(x_eval, r, p_envelope, left=0)
            ap_per_iou[k] = p_interp.mean()

        # Courbes interpolées sur la grille de confiance px (pour plots)
        # Précision et rappel au seuil IoU 0.5 (k=0)
        # On interpole en fonction de la confiance décroissante.
        p_curve = np.interp(-px, -conf_c, precision_cum[:, 0], left=1)
        r_curve = np.interp(-px, -conf_c, recall_cum[:, 0], left=0)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

        results[c] = {
            'ap50': float(ap_per_iou[0]),
            'ap5095': float(ap_per_iou.mean()),
            'p_curve': p_curve,
            'r_curve': r_curve,
            'f1_curve': f1_curve,
            'n_gt': n_gt_c,
            'n_pred': n_pred_c,
        }

    return results


# ---------------------------------------------------------------------------
# Sélection du seuil de confiance optimal sur la F1-Confidence
# ---------------------------------------------------------------------------

def find_best_f1_threshold(
    per_class: Dict[int, dict],
    px: np.ndarray,
) -> Tuple[float, float]:
    """Retourne (seuil de confiance optimal, F1 macro à ce seuil).

    Le seuil est déterminé sur la moyenne (macro) des F1 par classe :
    on cherche la confiance qui maximise la F1 moyenne sur toutes les classes
    présentes. C'est la convention de Ultralytics (cf. plots/F1_curve.png).
    """
    if not per_class:
        return 0.0, 0.0
    f1_stack = np.stack([v['f1_curve'] for v in per_class.values()], axis=0)
    f1_macro = f1_stack.mean(axis=0)  # (1000,)
    best_idx = int(np.argmax(f1_macro))
    return float(px[best_idx]), float(f1_macro[best_idx])


# ---------------------------------------------------------------------------
# Métriques scalaires P, R, F1 à un seuil de confiance donné
# ---------------------------------------------------------------------------

def metrics_at_threshold(
    per_class: Dict[int, dict],
    px: np.ndarray,
    conf_threshold: float,
) -> Dict[int, dict]:
    """Évalue P, R, F1 par classe au seuil de confiance donné.

    Le seuil est utilisé pour échantillonner les courbes interpolées sur px.
    Cela évite de refaire un matching à chaque seuil de confiance.
    """
    idx = int(np.argmin(np.abs(px - conf_threshold)))
    out = {}
    for c, d in per_class.items():
        p = float(d['p_curve'][idx])
        r = float(d['r_curve'][idx])
        f1 = float(d['f1_curve'][idx])
        out[c] = {'precision': p, 'recall': r, 'f1': f1}
    return out


# ---------------------------------------------------------------------------
# Matrice de confusion (N+1)x(N+1)
# ---------------------------------------------------------------------------

def build_confusion_matrix(
    all_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],  # liste par image
    all_gts: List[Tuple[np.ndarray, np.ndarray]],                 # liste par image
    num_classes: int,
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Construit une matrice de confusion (N+1) x (N+1).

    Convention :
      - lignes = classes prédites + une ligne "background" (index N)
      - colonnes = classes vraies + une colonne "background" (index N)
      - cm[i, j] = nombre de prédictions de classe i associées à un GT de classe j

    Cas spéciaux :
      - cm[N, j] = nombre de GT de classe j non détectés (FN par rapport à background)
      - cm[i, N] = nombre de prédictions de classe i sans GT correspondant (FP)

    Args:
        all_preds: liste, une entrée par image, chaque entrée est
                   (preds_xyxy, pred_cls, pred_conf)
        all_gts: liste, une entrée par image, chaque entrée est (gts_xyxy, gt_cls)
        iou_threshold: seuil IoU pour considérer un match
        conf_threshold: seuil de confiance pour filtrer les prédictions
    """
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    bg = num_classes  # index "background"

    for (preds_xyxy, pred_cls, pred_conf), (gts_xyxy, gt_cls) in zip(all_preds, all_gts):
        # Filtrage par seuil de confiance
        keep = pred_conf >= conf_threshold
        preds_xyxy = preds_xyxy[keep]
        pred_cls = pred_cls[keep]

        n_pred = preds_xyxy.shape[0]
        n_gt = gts_xyxy.shape[0]

        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            # Toutes les prédictions sont des FP (-> background)
            for c in pred_cls.astype(int):
                cm[c, bg] += 1
            continue
        if n_pred == 0:
            # Toutes les GT sont des FN (-> background)
            for c in gt_cls.astype(int):
                cm[bg, c] += 1
            continue

        # Matching greedy : pour chaque pred (par conf décroissante),
        # on cherche la meilleure GT non matchée avec IoU > seuil.
        # Note : on ne contraint PAS la classe ici, car on veut justement
        # voir les confusions inter-classes.
        iou_mat = box_iou(preds_xyxy, gts_xyxy)  # (n_pred, n_gt)
        order = np.argsort(-pred_conf[keep]) if keep.any() else np.argsort(-pred_conf)

        # Approche standard d'Ultralytics : on utilise les IoU pour le matching
        # mais on enregistre les confusions de classes dans la matrice.
        gt_matched = np.zeros(n_gt, dtype=bool)
        pred_matched = np.zeros(n_pred, dtype=bool)
        for i in order:
            ious_i = np.where(gt_matched, -1.0, iou_mat[i])
            j_best = int(np.argmax(ious_i))
            if ious_i[j_best] >= iou_threshold:
                cm[int(pred_cls[i]), int(gt_cls[j_best])] += 1
                gt_matched[j_best] = True
                pred_matched[i] = True

        # GT non matchées -> FN (ligne background)
        for j in np.where(~gt_matched)[0]:
            cm[bg, int(gt_cls[j])] += 1
        # Prédictions non matchées -> FP (colonne background)
        for i in np.where(~pred_matched)[0]:
            cm[int(pred_cls[i]), bg] += 1

    return cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pr_curves(
    per_class: Dict[int, dict],
    class_names: List[str],
    output_path: Path,
    title: str = "Courbe Précision-Rappel",
):
    """Plot des courbes PR par classe + moyenne (mAP@0.5).

    Style Ultralytics : courbes individuelles fines, courbe moyenne épaisse.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.linspace(0, 1, 1000)

    if not per_class:
        ax.text(0.5, 0.5, 'Aucune donnée à afficher',
                ha='center', va='center', transform=ax.transAxes)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    # Stack de toutes les courbes pour la moyenne
    all_p = np.stack([v['p_curve'] for v in per_class.values()], axis=0)
    all_r = np.stack([v['r_curve'] for v in per_class.values()], axis=0)

    # Recall trié pour la courbe PR (recall en abscisse)
    # On affiche P en fonction de R, en triant chaque courbe par recall croissant
    for c, d in per_class.items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        # Tri par recall croissant pour avoir une courbe propre
        idx = np.argsort(d['r_curve'])
        ax.plot(d['r_curve'][idx], d['p_curve'][idx],
                linewidth=1, alpha=0.5, label=f"{name} (AP@.5={d['ap50']:.3f})")

    # Courbe moyenne
    mean_p = all_p.mean(axis=0)
    mean_r = all_r.mean(axis=0)
    idx = np.argsort(mean_r)
    map50 = float(np.mean([v['ap50'] for v in per_class.values()]))
    ax.plot(mean_r[idx], mean_p[idx], 'b-', linewidth=2.5,
            label=f"toutes classes (mAP@.5={map50:.3f})")

    ax.set_xlabel('Rappel')
    ax.set_ylabel('Précision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(per_class) <= 20:
        ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    else:
        # Trop de classes -> seulement la moyenne dans la légende
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[-1]], [labels[-1]], loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_f1_confidence(
    per_class: Dict[int, dict],
    class_names: List[str],
    output_path: Path,
    title: str = "Courbe F1-Confiance",
):
    """Plot F1 = f(confidence) avec marquage du seuil optimal."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.linspace(0, 1, 1000)

    if not per_class:
        ax.text(0.5, 0.5, 'Aucune donnée', ha='center', va='center',
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    for c, d in per_class.items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        ax.plot(x, d['f1_curve'], linewidth=1, alpha=0.5, label=name)

    # F1 moyenne (macro)
    f1_stack = np.stack([v['f1_curve'] for v in per_class.values()], axis=0)
    f1_mean = f1_stack.mean(axis=0)
    ax.plot(x, f1_mean, 'b-', linewidth=2.5,
            label=f'toutes classes')

    # Seuil optimal
    best_idx = int(np.argmax(f1_mean))
    best_conf = float(x[best_idx])
    best_f1 = float(f1_mean[best_idx])
    ax.axvline(x=best_conf, color='r', linestyle='--', linewidth=1.5,
               label=f'seuil optimal: {best_conf:.3f} (F1={best_f1:.3f})')
    ax.scatter([best_conf], [best_f1], color='r', s=80, zorder=5)
    ax.annotate(f'  ({best_conf:.3f}, {best_f1:.3f})',
                xy=(best_conf, best_f1), fontsize=10, color='red',
                xytext=(8, 0), textcoords='offset points')

    ax.set_xlabel('Confiance')
    ax.set_ylabel('F1-Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(per_class) <= 20:
        ax.legend(loc='lower center', fontsize=7, framealpha=0.9)
    else:
        handles, labels = ax.get_legend_handles_labels()
        # On garde seulement la courbe moyenne et le marqueur du seuil optimal
        ax.legend(handles[-2:], labels[-2:], loc='lower center', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
    title: str = "Matrice de Confusion",
):
    """Plot de la matrice de confusion (N+1 x N+1, dernière classe = background).

    Si normalize=True, normalisation par colonne (par classe vraie),
    convention Ultralytics : chaque colonne somme à 1, lecture "quelle proportion
    des GT de la classe j a été prédite comme classe i".
    """
    nc = len(class_names)  # nb de "vraies" classes
    full_names = list(class_names) + ['background']

    cm_display = cm.astype(np.float64)
    if normalize:
        col_sums = cm_display.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # éviter div par zéro
        cm_display = cm_display / col_sums

    # Taille adaptée au nb de classes
    fig_size = max(8, min(20, 0.5 * (nc + 1) + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_display, cmap='Blues', aspect='auto',
                   vmin=0, vmax=1.0 if normalize else cm.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(full_names)))
    ax.set_yticks(np.arange(len(full_names)))
    ax.set_xticklabels(full_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(full_names, fontsize=11)
    ax.set_xlabel('Vraie classe', fontsize=12)
    ax.set_ylabel('Classe prédite', fontsize=12)
    ax.set_title(title + (' (normalisée)' if normalize else ' (counts)'),
                 fontsize=13)

    # Annotations dans les cellules (texte)
    threshold = cm_display.max() / 2.0
    if nc <= 30:  # n'afficher les chiffres que si la matrice n'est pas trop grosse
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                v = cm_display[i, j]
                if v > 0:
                    text = f"{v:.2f}" if normalize else f"{int(v)}"
                    ax.text(j, i, text,
                            ha='center', va='center',
                            color='white' if v > threshold else 'black',
                            fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Construction des CSV
# ---------------------------------------------------------------------------

def build_per_class_csv(
    per_class: Dict[int, dict],
    metrics_at_thr: Dict[int, dict],
    class_names: List[str],
    cls_loss_per_class: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """CSV de métriques par classe.

    Colonnes : class_id, class_name, n_gt, n_pred, precision, recall, f1,
               ap50, ap50_95, cls_loss
    """
    rows = []
    for c, d in sorted(per_class.items()):
        m = metrics_at_thr.get(c, {})
        row = {
            'class_id': c,
            'class_name': class_names[c] if c < len(class_names) else f'class_{c}',
            'n_gt': d['n_gt'],
            'n_pred': d['n_pred'],
            'precision': round(m.get('precision', 0.0), 4),
            'recall': round(m.get('recall', 0.0), 4),
            'f1': round(m.get('f1', 0.0), 4),
            'ap50': round(d['ap50'], 4),
            'ap50_95': round(d['ap5095'], 4),
        }
        if cls_loss_per_class is not None:
            row['cls_loss'] = round(cls_loss_per_class.get(c, 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_global_csv(
    per_class: Dict[int, dict],
    metrics_at_thr: Dict[int, dict],
    losses: Dict[str, float],
    iou_mean: float,
    best_conf_threshold: float,
    best_f1: float,
    n_total_gt: int,
    n_total_pred: int,
    n_total_tp: int,
    n_total_fp: int,
    n_total_fn: int,
) -> pd.DataFrame:
    """CSV de métriques globales.

    On distingue:
      - Macro : moyenne arithmétique non pondérée des métriques par classe
      - Micro : agrégation sur l'ensemble des prédictions (chaque pred = 1 vote)
    """
    if per_class:
        ap50_per_class = np.array([d['ap50'] for d in per_class.values()])
        ap5095_per_class = np.array([d['ap5095'] for d in per_class.values()])
        p_per_class = np.array([metrics_at_thr.get(c, {}).get('precision', 0.0)
                                 for c in per_class])
        r_per_class = np.array([metrics_at_thr.get(c, {}).get('recall', 0.0)
                                 for c in per_class])
        f1_per_class = np.array([metrics_at_thr.get(c, {}).get('f1', 0.0)
                                  for c in per_class])
        map50 = float(ap50_per_class.mean())
        map5095 = float(ap5095_per_class.mean())
        p_macro = float(p_per_class.mean())
        r_macro = float(r_per_class.mean())
        f1_macro = float(f1_per_class.mean())
    else:
        map50 = map5095 = p_macro = r_macro = f1_macro = 0.0

    # Micro: TP / (TP+FP) etc. agrégés sur tout le dataset
    eps = 1e-9
    p_micro = n_total_tp / (n_total_tp + n_total_fp + eps)
    r_micro = n_total_tp / (n_total_tp + n_total_fn + eps)
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro + eps)

    rows = [
        ('mAP50',           round(map50, 4)),
        ('mAP50-95',        round(map5095, 4)),
        ('precision_macro', round(p_macro, 4)),
        ('recall_macro',    round(r_macro, 4)),
        ('f1_macro',        round(f1_macro, 4)),
        ('precision_micro', round(p_micro, 4)),
        ('recall_micro',    round(r_micro, 4)),
        ('f1_micro',        round(f1_micro, 4)),
        ('iou_mean',        round(iou_mean, 4)),
        ('box_loss',        round(losses.get('box', 0.0), 4)),
        ('cls_loss',        round(losses.get('cls', 0.0), 4)),
        ('dfl_loss',        round(losses.get('dfl', 0.0), 4)),
        ('total_loss',      round(losses.get('total', 0.0), 4)),
        ('best_conf_threshold', round(best_conf_threshold, 4)),
        ('best_f1_macro',   round(best_f1, 4)),
        ('n_total_gt',      int(n_total_gt)),
        ('n_total_pred',    int(n_total_pred)),
        ('n_total_tp',      int(n_total_tp)),
        ('n_total_fp',      int(n_total_fp)),
        ('n_total_fn',      int(n_total_fn)),
    ]
    return pd.DataFrame(rows, columns=['metric', 'value'])

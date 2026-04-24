"""
Construction d'un modèle YOLOv8 fine-tunable à partir de poids pré-entraînés.

===============================================================================
                 Principe académique du fine-tuning en détection
===============================================================================

Un détecteur d'objets moderne comme YOLOv8 se décompose en trois blocs fonctionnels
qui n'ont pas la même dépendance vis-à-vis du domaine cible :

    1. Backbone (CSPDarkNet + C2f) :
       Extracteur de caractéristiques hiérarchique. Les premières couches
       apprennent des primitives visuelles universelles (arêtes, textures,
       motifs locaux), les couches profondes codent des patterns sémantiques
       plus spécifiques. Ces représentations sont en grande partie transférables
       entre domaines, ce qui motive leur conservation lors du fine-tuning.

    2. Neck (PAN-FPN) :
       Agrège les features multi-échelles du backbone. Indépendant du nombre de
       classes. Conservé pour les mêmes raisons que le backbone.

    3. Head (Detect) :
       Produit deux sorties parallèles par niveau d'échelle :
         - branche `box` : 4*reg_max canaux (régression DFL des coordonnées).
           Géométrique pure, indépendante de num_classes.
         - branche `cls` : num_classes canaux (scores de classification).
           C'est la SEULE partie qui dépend du nombre de classes cibles.

Conséquence pratique : pour adapter un modèle entraîné sur K anciennes classes à
un nouveau problème à K' classes, on réinitialise uniquement les branches `cls`
(shape incompatible entre K et K') et on conserve tout le reste.

===============================================================================
        Initialisation du biais des nouvelles couches de classification
===============================================================================

Focal Loss (Lin et al., 2017) a popularisé l'initialisation du biais final des
classifieurs de détection avec un prior faible. Les branches `cls` de YOLOv8
utilisent une activation sigmoïde et une BCE, où la probabilité initiale doit
refléter le déséquilibre massif entre arrière-plan et objets.

On pose une probabilité a priori π ≈ 0.01 (environ 1% de positifs), ce qui donne
le biais initial b = -log((1 - π) / π) ≈ -4.595. Cette initialisation évite
l'explosion du gradient aux premières itérations et aligne le comportement de la
nouvelle tête avec la littérature standard de détection.

Référence :
  Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
  Ge et al., "YOLOX: Exceeding YOLO Series in 2021", (reprend la même convention).

===============================================================================
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from module.config import FinetuneConfig, load_finetune_config
from module.model import MyYolo
from module.utils import print_model_summary


# ---------------------------------------------------------------------------
# Chargement des poids source
# ---------------------------------------------------------------------------

def load_source_checkpoint(cfg: FinetuneConfig, device: torch.device):
    """Charge le state_dict du modèle pré-entraîné à partir d'un fichier .pt.

    Accepte aussi bien un checkpoint complet (dict avec clé 'model') qu'un
    state_dict nu. Utilise weights_only=False pour rester compatible avec
    PyTorch >= 2.6 qui refuse par défaut de désérialiser les objets non-tensoriels.
    """
    src_path = Path(cfg.pretrained_weights)
    if not src_path.exists():
        raise FileNotFoundError(f"Poids source introuvables: {src_path}")

    try:
        ckpt = torch.load(src_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(src_path, map_location=device)

    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    print(f"[source] Chargé: {src_path}  ({len(state)} tenseurs)")
    return state


# ---------------------------------------------------------------------------
# Initialisation des nouvelles têtes de classification
# ---------------------------------------------------------------------------

def init_cls_head_bias(module: nn.Module, num_classes: int, prior: float = 0.01):
    """Initialise le biais de la dernière Conv2d de chaque branche `cls`.

    Convention issue de Focal Loss (Lin et al. 2017) : on pose un prior π sur
    la probabilité positive attendue, et on règle le biais à
        b = -log((1 - π) / π)
    de sorte que la sigmoïde initiale donne exactement π. Le poids est laissé
    à sa valeur par défaut (Kaiming via la construction de nn.Conv2d), car un
    biais cohérent suffit pour stabiliser les premiers pas de descente.

    Args:
        module: la branche cls[i] du Head (nn.Sequential: Conv, Conv, Conv2d)
        num_classes: nombre de classes (utilisé pour dimensionner le biais)
        prior: probabilité a priori d'un positif (défaut 0.01)
    """
    # La dernière Conv2d 1x1 produit les logits de classification
    final_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.out_channels == num_classes:
            final_conv = m
    if final_conv is None or final_conv.bias is None:
        return

    # Calcul du biais (même valeur pour toutes les classes)
    bias_init = -math.log((1.0 - prior) / prior)
    with torch.no_grad():
        final_conv.bias.fill_(bias_init)


# ---------------------------------------------------------------------------
# Transfert sélectif des poids (vue par vue)
# ---------------------------------------------------------------------------

def transfer_weights(src_state: dict, new_model: MyYolo,
                     old_num_classes: int, new_num_classes: int,
                     strict_backbone: bool = True):
    """Transfère les poids du modèle source vers le nouveau modèle.

    Les règles de transfert sont les suivantes :
      - Toutes les clés de `backbone.*` et `neck.*` sont chargées strictement
        (shape-compatible). Un mismatch ici signale une erreur d'usage
        (mauvais `version` par exemple).
      - Dans `head.*`, les clés dont la shape correspond exactement sont
        chargées (branches `box.*`, `dfl.*`). Celles qui ne correspondent pas
        sont laissées à leur initialisation aléatoire (branches `cls.*`,
        car leur dimension de sortie dépend de num_classes).

    Cette approche est plus robuste que `load_state_dict(..., strict=False)`
    pur car elle catégorise les mismatches et lève une exception explicite
    si le backbone/neck ne chargent pas, ce qui est quasi-toujours un bug.

    Returns:
        (loaded, skipped): listes des noms de tenseurs chargés / écartés
    """
    new_state = new_model.state_dict()
    loaded, skipped_shape_mismatch, missing = [], [], []

    for key, new_tensor in new_state.items():
        if key not in src_state:
            # Clé inexistante dans le modèle source : souvent un buffer (stride,
            # num_batches_tracked...) ou une initialisation ajoutée récemment.
            missing.append(key)
            continue
        src_tensor = src_state[key]
        if src_tensor.shape == new_tensor.shape:
            # Copie directe: les formes concordent (backbone, neck, box, dfl...)
            new_state[key] = src_tensor.clone()
            loaded.append(key)
        else:
            # Forme incompatible: presque toujours une couche liée à num_classes.
            # On laisse l'initialisation aléatoire du nouveau modèle en place.
            skipped_shape_mismatch.append(
                (key, tuple(src_tensor.shape), tuple(new_tensor.shape))
            )

    # Vérification stricte du backbone/neck: si un tenseur dans ces blocs n'a
    # pas été chargé ET qu'on est en mode strict, on lève une exception.
    # C'est un garde-fou contre les erreurs de config (mauvaise version, etc.).
    if strict_backbone:
        feature_layer_prefixes = ('backbone.', 'neck.')
        not_loaded_feature = [
            k for k in new_state
            if k.startswith(feature_layer_prefixes) and k not in loaded
            and k not in missing  # les buffers non-critiques sont tolérés
        ]
        # On filtre les buffers classiques (num_batches_tracked) qui peuvent
        # manquer sans que ce soit problématique.
        not_loaded_feature = [k for k in not_loaded_feature
                              if not k.endswith('num_batches_tracked')]
        if not_loaded_feature:
            raise RuntimeError(
                f"Transfert échoué pour {len(not_loaded_feature)} tenseurs de "
                f"backbone/neck. Vérifiez que `version` est identique dans la "
                f"config de fine-tuning et le modèle source.\n"
                f"Exemples: {not_loaded_feature[:3]}"
            )

    new_model.load_state_dict(new_state)

    # Rapport détaillé
    total = len(new_state)
    print(f"[transfer] {len(loaded)}/{total} tenseurs chargés depuis le modèle source")
    if skipped_shape_mismatch:
        print(f"[transfer] {len(skipped_shape_mismatch)} tenseurs réinitialisés "
              f"(shape mismatch — attendu pour les têtes cls si "
              f"old_num_classes={old_num_classes} -> new={new_num_classes}):")
        for key, old_shape, new_shape in skipped_shape_mismatch[:6]:
            print(f"    · {key}: {old_shape} -> {new_shape}")
        if len(skipped_shape_mismatch) > 6:
            print(f"    · ... et {len(skipped_shape_mismatch) - 6} autres")

    return loaded, skipped_shape_mismatch


# ---------------------------------------------------------------------------
# Sauvegarde du modèle fine-tunable
# ---------------------------------------------------------------------------

def save_finetune_checkpoint(model: MyYolo, cfg: FinetuneConfig, output_path: Path):
    """Sauvegarde le modèle initialisé au format checkpoint standard du projet.

    On utilise le même schéma que `train.py` (clé 'model' pour le state_dict)
    pour que le checkpoint puisse être repris avec `resume:` sans modification.
    On met `epoch=-1` pour signaler qu'aucune epoch de fine-tuning n'a encore
    été faite (la reprise repartira de l'epoch 0).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'model': model.state_dict(),
        'epoch': -1,
        'best_metric': -float('inf'),
        'finetune_origin': cfg.pretrained_weights,
        'finetune_old_num_classes': cfg.old_num_classes,
        'finetune_new_num_classes': cfg.new_num_classes,
    }
    # Écriture atomique via fichier temporaire + rename
    tmp = output_path.with_suffix(output_path.suffix + '.tmp')
    torch.save(state, tmp)
    tmp.replace(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[save] Nouveau modèle initialisé écrit: {output_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_finetune_build(cfg: FinetuneConfig):
    """Construit un modèle YOLOv8 fine-tunable à partir du checkpoint source.

    Étapes (dans l'ordre strict) :
      1. Instancier un nouveau MyYolo avec new_num_classes.
         → cela initialise aléatoirement backbone, neck et head (avec la bonne
            dimension pour `cls`).
      2. Charger le state_dict du modèle source.
      3. Transférer sélectivement les poids (shape-compatible → copie,
         sinon réinitialisation aléatoire conservée).
      4. Réinitialiser le biais de chaque branche `cls` avec le prior de
         Focal Loss pour stabiliser les premières itérations.
      5. Afficher le résumé du nouveau modèle avec torchinfo.
      6. Sauvegarder le state_dict résultant.
    """
    device = torch.device(cfg.device)
    print(f"[setup] device={device}")
    print(f"[setup] version={cfg.version} | old_nc={cfg.old_num_classes} "
          f"-> new_nc={cfg.new_num_classes}")

    # --- Étape 1: instanciation du nouveau modèle ---
    # Le constructeur MyYolo construit le graphe complet et calibre les strides.
    # La tête `cls` est déjà dimensionnée pour new_num_classes grâce à notre
    # paramètre ; c'est là que réside toute la logique d'adaptation.
    new_model = MyYolo(
        version=cfg.version,
        num_classes=cfg.new_num_classes,
        input_size=cfg.image_size,
    ).to(device)
    new_model.head.stride = new_model.head.stride.to(device)

    # --- Étape 2: chargement du checkpoint source ---
    src_state = load_source_checkpoint(cfg, device)

    # --- Étape 3: transfert sélectif ---
    transfer_weights(
        src_state, new_model,
        old_num_classes=cfg.old_num_classes,
        new_num_classes=cfg.new_num_classes,
        strict_backbone=cfg.strict_backbone_load,
    )

    # --- Étape 4: initialisation du biais des branches cls ---
    # new_model.head.cls est un nn.ModuleList avec 3 branches (une par échelle).
    # Chaque branche est un nn.Sequential dont la dernière couche est une Conv2d
    # de sortie = new_num_classes.
    for i, cls_branch in enumerate(new_model.head.cls):
        init_cls_head_bias(cls_branch, cfg.new_num_classes, prior=cfg.cls_prior)
    bias_value = -math.log((1.0 - cfg.cls_prior) / cfg.cls_prior)
    print(f"[init] Biais des 3 branches cls fixé à b = -log((1-π)/π) = "
          f"{bias_value:.4f}  (π={cfg.cls_prior})")

    # --- Étape 5: résumé du modèle ---
    print("\n" + "=" * 80)
    print("Résumé du modèle fine-tunable construit")
    print("=" * 80)
    print_model_summary(
        new_model,
        input_size=(1, 3, cfg.image_size, cfg.image_size),
        device=device,
    )

    # --- Étape 6: sauvegarde ---
    output_path = Path(cfg.output_weights)
    save_finetune_checkpoint(new_model, cfg, output_path)

    # Instructions pour la suite
    print("\n[done] Modèle prêt pour le fine-tuning.")
    print(f"       Pour lancer l'entraînement sur les nouvelles classes :")
    print(f"         1. Mettez dans train.yaml :")
    print(f"              resume: {output_path}")
    print(f"              num_classes: {cfg.new_num_classes}")
    print(f"              version: {cfg.version}")
    print(f"         2. Optionnel : freeze_feature_layers: true "
          f"(pour figer backbone + neck)")
    print(f"         3. python -m module.train --config train.yaml")

    return new_model, output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Construit un modèle YOLOv8 fine-tunable à partir de poids "
                    "pré-entraînés (adaptation à un nouveau nombre de classes).",
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers finetune.yaml')
    args = parser.parse_args()

    cfg = load_finetune_config(args.config)
    run_finetune_build(cfg)


if __name__ == '__main__':
    main()

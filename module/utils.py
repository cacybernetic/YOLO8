"""Utilitaires partagés entre les scripts (train, evaluate, export, infer)."""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from loguru import logger


def setup_logging(level: str = "INFO",
                  show_time: bool = True,
                  log_to_file: bool = True,
                  log_dir: Union[str, Path] = "logs"):
    """Configure loguru avec un format unifié pour tout le projet.

    Deux sinks sont configurés :
      - Console (stderr) : format compact et coloré pour l'humain qui regarde
        l'entraînement en direct.
      - Fichier journalier (`logs/app_YYYY-MM-DD.log`) : format détaillé avec
        chemin module:fonction:ligne, rotation à 500 MB, rétention 30 jours,
        compression .zip des anciens fichiers. Format utile pour analyser un
        run a posteriori (corruption de données, ralentissement, etc.).

    Loguru utilise un logger global (`logger`), il suffit donc de l'appeler
    une fois dans le script d'entrée (train.py, evaluate.py, etc.) pour que
    tous les modules profitent de la même configuration.

    Args:
        level: niveau minimum à afficher ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        show_time: ajoute l'heure en début de ligne console.
        log_to_file: active la persistance dans `log_dir`. Mettre `False` si
            le système de fichiers est read-only (certains conteneurs / CI).
        log_dir: dossier de destination des fichiers de log persistés.
    """
    # On retire le handler par défaut de loguru pour éviter les doublons:
    # par défaut loguru ajoute un handler vers stderr au premier import.
    logger.remove()

    # --- Sink console (stderr) ---
    if show_time:
        fmt_console = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<level>{message}</level>")
    else:
        fmt_console = "<level>{level: <8}</level> | <level>{message}</level>"

    # `colorize=True` active les couleurs ANSI si la sortie est un TTY.
    # `enqueue=False` car le main process est seul à logger ; les workers
    # DataLoader sont muets.
    logger.add(
        sys.stderr,
        format=fmt_console,
        level=level,
        colorize=True,
        enqueue=False,
    )

    # --- Sink fichier (rotation + rétention + compression) ---
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        # Format détaillé pour le fichier: utile en post-mortem pour identifier
        # exactement où un message a été émis (chemin module:fonction:ligne).
        fmt_file = ("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                    "{name}:{function}:{line} | {message}")
        logger.add(
            str(log_path / "app_{time:YYYY-MM-DD}.log"),
            level=level,
            format=fmt_file,
            rotation="500 MB",     # nouveau fichier au-delà de 500 MB
            retention="30 days",   # supprime les logs plus vieux que 30 jours
            compression="zip",     # archive les anciens fichiers en .zip
            enqueue=True,          # thread-safe: utile car le sink fichier peut
                                   # être appelé indirectement depuis un worker
            backtrace=True,        # stack trace complète pour les exceptions
            diagnose=False,        # `False` en prod : évite de fuiter les valeurs
                                   # de variables locales sensibles dans les logs
        )


def print_model_summary(model: nn.Module,
                        input_size=(1, 3, 640, 640),
                        device: Optional[torch.device] = None,
                        depth: int = 4,
                        verbose: int = 1):
    """Affiche un résumé détaillé du modèle via torchinfo.

    Args:
        model: modèle PyTorch (typiquement MyYolo)
        input_size: shape factice pour faire un forward et compter les MACs/params par couche
        device: device sur lequel faire le forward (par défaut: celui du modèle)
        depth: profondeur d'inspection des sous-modules (4 suffit pour voir Backbone/Neck/Head)
        verbose: 0=silencieux, 1=résumé, 2=détaillé

    Si torchinfo n'est pas installé, on tombe sur un résumé minimal natif.
    """
    if device is None:
        device = next(model.parameters()).device

    try:
        from torchinfo import summary
    except ImportError:
        logger.warning("torchinfo non installé. Installez avec: pip install torchinfo")
        _print_native_summary(model)
        return

    # On sauvegarde l'état training/eval de CHAQUE sous-module avant de tout
    # basculer en eval() pour le forward de torchinfo. Sans cela, un simple
    # model.train() final remettrait en mode train les couches qu'on veut
    # garder en eval (ex: BatchNorm figées par freeze_feature_layers).
    training_states = {name: m.training for name, m in model.named_modules()}

    model.eval()
    try:
        # Note: torchinfo.summary() utilise print() en interne; on n'intercepte
        # pas, le tableau de torchinfo reste affiché tel quel.
        summary(
            model,
            input_size=input_size,
            device=device,
            depth=depth,
            verbose=verbose,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            row_settings=("var_names",),
        )
    except Exception as e:
        # torchinfo peut échouer sur certains modèles avec sorties non-tensorielles
        # (la tête de YOLO retourne un tuple en eval). Dans ce cas, fallback.
        logger.warning(f"torchinfo a échoué ({e.__class__.__name__}): {e}")
        logger.info("Fallback sur le résumé natif")
        _print_native_summary(model)
    finally:
        # Restauration fidèle de l'état train/eval de chaque sous-module
        for name, m in model.named_modules():
            if name in training_states:
                m.training = training_states[name]


def _print_native_summary(model: nn.Module):
    """Résumé minimal sans torchinfo: nb total de params + nb trainables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params:     {total:>12,}  ({total/1e6:.3f} M)")
    logger.info(f"Trainable params: {trainable:>12,}  ({trainable/1e6:.3f} M)")
    logger.info(f"Non-trainable:    {total-trainable:>12,}")


def plot_training_history(history: dict, output_path):
    """Trace l'historique des pertes train/val sur 4 subplots.

    Subplots (2x2):
        [0,0] avg_loss : perte totale (train + val)
        [0,1] avg_box  : perte de régression des boites
        [1,0] avg_cls  : perte de classification
        [1,1] avg_dfl  : perte DFL (Distribution Focal Loss)

    Une telle figure est l'outil principal pour diagnostiquer la convergence
    et le sur-apprentissage en cours d'entraînement (cf. checklist du document
    `yolov8_metriques.tex`):
      - Convergence: les courbes train descendent et stagnent.
      - Sur-apprentissage: la courbe val remonte alors que train continue de
        descendre. Si val_loss > 1.5 * train_loss, regularisation insuffisante.
      - Bug de calcul: une courbe constante ou bruitée signale un problème
        (mauvais learning rate, données corrompues, etc.).

    La fonction est tolérante aux longueurs inégales (train et val peuvent
    avoir des nombres différents d'epochs si la val n'est pas faite à chaque
    epoch — les courbes seront simplement plus courtes sur ce canal).

    Args:
        history: dict avec les clés:
            'train_loss', 'train_box', 'train_cls', 'train_dfl' (listes float)
            'val_loss',   'val_box',   'val_cls',   'val_dfl'   (listes float)
            'epochs_train', 'epochs_val' (listes int, 1-indexed pour lisibilité)
        output_path: str ou Path, chemin du PNG à écrire.
    """
    # Import lazy : matplotlib n'est nécessaire que pour cette fonction,
    # Import lazy : matplotlib n'est nécessaire que pour cette fonction,
    # cela évite de plomber le démarrage de train.py si matplotlib n'est pas
    # installé (cas headless minimal). `Path` est déjà importé en haut.
    import matplotlib
    matplotlib.use('Agg')  # backend non-interactif (compatible serveur sans X)
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configuration des 4 subplots (2 lignes x 2 colonnes)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Définition de chaque subplot: (axe, titre, clé_train, clé_val, ylabel)
    panels = [
        (axes[0, 0], 'Perte totale (avg_loss)',         'train_loss', 'val_loss', 'avg_loss'),
        (axes[0, 1], 'Perte boîte (avg_box / CIoU)',     'train_box',  'val_box',  'avg_box'),
        (axes[1, 0], 'Perte classification (avg_cls)',   'train_cls',  'val_cls',  'avg_cls'),
        (axes[1, 1], 'Perte DFL (avg_dfl)',              'train_dfl',  'val_dfl',  'avg_dfl'),
    ]

    epochs_train = history.get('epochs_train', [])
    epochs_val = history.get('epochs_val', [])

    for ax, title, k_train, k_val, ylabel in panels:
        train_vals = history.get(k_train, [])
        val_vals = history.get(k_val, [])

        if train_vals:
            ax.plot(epochs_train[:len(train_vals)], train_vals,
                    color='#1f77b4', linewidth=2, marker='o', markersize=4,
                    label='train')
        if val_vals:
            ax.plot(epochs_val[:len(val_vals)], val_vals,
                    color='#d62728', linewidth=2, marker='s', markersize=4,
                    label='val')

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Ticks d'epoch en entiers (les epochs ne sont pas continues)
        if epochs_train:
            max_epoch = max(epochs_train + epochs_val) if epochs_val else max(epochs_train)
            # Limite raisonnable de ticks pour rester lisible
            n_ticks = min(max_epoch, 15)
            if max_epoch > 0:
                step = max(1, max_epoch // n_ticks)
                ax.set_xticks(list(range(1, max_epoch + 1, step)))

    fig.suptitle("Historique d'entraînement", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

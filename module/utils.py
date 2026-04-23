"""Utilitaires partagés entre les scripts (train, evaluate, export, infer)."""

from typing import Optional

import torch
import torch.nn as nn


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
        print("[summary] torchinfo non installé. "
              "Installez avec: pip install torchinfo")
        _print_native_summary(model)
        return

    was_training = model.training
    model.eval()
    try:
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
        print(f"[summary] torchinfo a échoué ({e.__class__.__name__}): {e}")
        print("[summary] Fallback sur le résumé natif.")
        _print_native_summary(model)
    finally:
        if was_training:
            model.train()


def _print_native_summary(model: nn.Module):
    """Résumé minimal sans torchinfo: nb total de params + nb trainables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[summary] Total params:     {total:>12,}  ({total/1e6:.3f} M)")
    print(f"[summary] Trainable params: {trainable:>12,}  ({trainable/1e6:.3f} M)")
    print(f"[summary] Non-trainable:    {total-trainable:>12,}")

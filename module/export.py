"""
Export d'un modèle YOLOv8 entraîné au format ONNX.

Usage :
    python -m module.export --config module/export.yaml

Le script :
  1. Charge le checkpoint .pt indiqué dans la config
  2. Enveloppe le modèle dans un wrapper qui ne retourne que la sortie d'inférence
     (sans le tuple de sorties brutes utilisé pour la loss en validation)
  3. Exporte en ONNX avec opset configurable, batch dynamique, constant folding
  4. Vérifie le graphe avec onnx.checker
  5. (Optionnel) simplifie le graphe avec onnxsim
  6. (Optionnel) compare la sortie PyTorch avec onnxruntime pour valider l'export

Dépendances :
  - onnx          : vérification du graphe (optionnel mais recommandé)
  - onnxsim       : simplification (optionnel)
  - onnxruntime   : verification numérique (optionnel)

Installer via : pip install onnx onnxsim onnxruntime
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from module.config import ExportConfig, load_export_config
from module.model import MyYolo
from module.utils import print_model_summary, setup_logging


# ---------------------------------------------------------------------------
# Wrapper d'export
# ---------------------------------------------------------------------------

class YoloExportWrapper(nn.Module):
    """Wrapper minimal pour l'export ONNX.

    Le Head de MyYolo retourne en mode eval un tuple (inference_out, raw_outputs).
    Pour un graphe ONNX propre et exploitable, on ne garde que `inference_out` :
      inference_out: (B, 4 + num_classes, num_anchors)
        - 4 premières lignes: (cx, cy, w, h) en coordonnées image (déjà * stride)
        - num_classes lignes suivantes: scores sigmoïdés [0, 1]
    """

    def __init__(self, model: MyYolo):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # En eval, MyYolo retourne (inference_tensor, raw_list)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Chargement des poids
# ---------------------------------------------------------------------------

def load_model(cfg: ExportConfig, device: torch.device) -> MyYolo:
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
        logger.warning(f"{len(missing)} clé(s) manquante(s) (normal si la tête a changé)")
    if unexpected:
        logger.warning(f"{len(unexpected)} clé(s) inattendue(s)")
    logger.info(f"Poids chargés: {weights_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Export ONNX
# ---------------------------------------------------------------------------

def export_to_onnx(wrapper: YoloExportWrapper, dummy_input: torch.Tensor,
                   output_path: Path, cfg: ExportConfig):
    """Effectue l'export ONNX avec les options configurées."""
    input_names = ['images']
    output_names = ['output']

    dynamic_axes = None
    if cfg.dynamic:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch', 2: 'anchors'},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Export: opset={cfg.opset} | dynamic_batch={cfg.dynamic} | half={cfg.half}")
    logger.info(f"Input shape: {tuple(dummy_input.shape)}")

    # Test forward avant export pour avoir un message d'erreur clair si besoin
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    logger.info(f"Output shape (référence PyTorch): {tuple(test_out.shape)}")

    # Force l'exporteur legacy (TorchScript) pour la stabilité. Le nouvel
    # exporteur "dynamo" (PyTorch >= 2.5) nécessite onnxscript comme dépendance
    # supplémentaire et change parfois le nom des sorties; le legacy reste le
    # plus portable pour la production.
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=cfg.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # PyTorch < 2.5 n'a pas le paramètre `dynamo`
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=cfg.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    logger.success(f"Fichier ONNX écrit: {output_path} "
                   f"({output_path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Vérification et simplification
# ---------------------------------------------------------------------------

def check_onnx_graph(onnx_path: Path):
    """Vérifie la validité du graphe ONNX."""
    try:
        import onnx
    except ImportError:
        logger.warning("onnx non installé — check skipped. Installez avec: pip install onnx")
        return None
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    logger.success(f"Graphe ONNX valide "
                   f"(ir_version={model_onnx.ir_version}, "
                   f"opset={model_onnx.opset_import[0].version})")
    return model_onnx


def simplify_onnx(onnx_path: Path, onnx_model=None):
    """Simplifie le graphe ONNX avec onnxsim (si installé)."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        logger.warning("onnxsim non installé — simplify skipped. "
                       "Installez avec: pip install onnxsim")
        return
    if onnx_model is None:
        onnx_model = onnx.load(str(onnx_path))
    try:
        simplified, ok = onnxsim.simplify(onnx_model)
        if not ok:
            logger.warning("onnxsim a signalé un échec — graphe NON remplacé.")
            return
        onnx.save(simplified, str(onnx_path))
        new_size = onnx_path.stat().st_size / 1e6
        logger.success(f"Graphe simplifié et sauvegardé ({new_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Simplify échec: {e}")


def convert_to_fp16(onnx_path: Path):
    """Convertit le modèle ONNX en FP16."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        logger.warning("onnxconverter-common non installé — half skipped. "
                       "Installez avec: pip install onnxconverter-common")
        return
    try:
        model = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
        onnx.save(model_fp16, str(onnx_path))
        new_size = onnx_path.stat().st_size / 1e6
        logger.success(f"Conversion FP16 effectuée ({new_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Half échec: {e}")


def verify_numerical(wrapper: YoloExportWrapper, onnx_path: Path,
                     dummy_input: torch.Tensor, tolerance: float) -> bool:
    """Compare la sortie PyTorch avec celle d'ONNX Runtime sur la même entrée."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime non installé — verify skipped. "
                       "Installez avec: pip install onnxruntime")
        return True

    with torch.no_grad():
        torch_out = wrapper(dummy_input).cpu().numpy()

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_out = session.run(None, {'images': dummy_input.cpu().numpy()})[0]

    if torch_out.shape != onnx_out.shape:
        logger.error(f"Verify ÉCHEC: shapes différentes "
                     f"(torch={torch_out.shape}, onnx={onnx_out.shape})")
        return False

    abs_diff = np.abs(torch_out - onnx_out)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    ok = max_diff <= tolerance

    if ok:
        logger.success(f"Verify OK | max_abs_diff={max_diff:.2e} "
                       f"mean_abs_diff={mean_diff:.2e} (tol={tolerance:.0e})")
    else:
        logger.error(f"Verify ÉCHEC | max_abs_diff={max_diff:.2e} "
                     f"mean_abs_diff={mean_diff:.2e} (tol={tolerance:.0e})")
        logger.warning("L'écart numérique dépasse la tolérance. "
                       "Cela peut venir de simplify=True sur certains opérateurs flottants. "
                       "Réessayez avec simplify=False pour isoler la cause.")
    return ok


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_export(cfg: ExportConfig):
    device = torch.device(cfg.device if torch.cuda.is_available()
                          or not cfg.device.startswith('cuda') else 'cpu')
    logger.info(f"device={device}")

    # 1) Modèle
    model = load_model(cfg, device)
    print_model_summary(model, input_size=(1, 3, cfg.image_size, cfg.image_size),
                        device=device)
    wrapper = YoloExportWrapper(model).to(device).eval()

    # 2) Entrée factice
    dummy_input = torch.zeros(
        1, 3, cfg.image_size, cfg.image_size,
        dtype=torch.float32, device=device,
    )

    # 3) Export
    output_path = Path(cfg.output_path)
    export_to_onnx(wrapper, dummy_input, output_path, cfg)

    # 4) Vérification du graphe
    onnx_model = None
    if cfg.check:
        onnx_model = check_onnx_graph(output_path)

    # 5) Simplification (avant la conversion FP16 pour rester exact)
    if cfg.simplify:
        simplify_onnx(output_path, onnx_model=onnx_model)

    # 6) Vérification numérique (avant FP16 pour éviter les faux négatifs)
    if cfg.verify:
        verify_numerical(wrapper, output_path, dummy_input, cfg.verify_tolerance)

    # 7) Conversion FP16 (optionnelle, fait en dernier)
    if cfg.half:
        convert_to_fp16(output_path)

    logger.success(f"Export terminé: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exporte un modèle YOLOv8 entraîné au format ONNX.",
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers export.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Niveau de log loguru')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    cfg = load_export_config(args.config)
    run_export(cfg)


if __name__ == '__main__':
    main()

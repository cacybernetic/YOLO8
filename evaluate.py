"""
Script d'évaluation d'un modèle YOLOv8 entraîné.

Usage :
    python -m module.evaluate --config module/eval.yaml
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.config import load_eval_config, EvalConfig
from module.dataset import YOLODataset
from module.metrics import MetricAccumulator, non_max_suppression
from module.model import MyYolo
from module.utils import print_model_summary
from train import build_val_targets


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device(cfg.device if torch.cuda.is_available()
                          or not cfg.device.startswith('cuda') else 'cpu')
    print(f"[setup] device={device}")

    # Dataset
    ds = YOLODataset(cfg.dataset_dir, split=cfg.split,
                     image_size=cfg.image_size, augment=False)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=YOLODataset.collate_fn, drop_last=False,
    )
    print(f"[data] {cfg.split}={len(ds)}")

    # Modèle + poids
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
        print(f"[weights] missing keys: {len(missing)}")
    if unexpected:
        print(f"[weights] unexpected keys: {len(unexpected)}")
    print(f"[weights] Chargé: {weights_path}")
    print_model_summary(model, input_size=(1, 3, cfg.image_size, cfg.image_size),
                        device=device)

    model.eval()
    accumulator = MetricAccumulator(device=device)

    pbar = tqdm(loader, desc="[eval]", leave=False, dynamic_ncols=True)
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        out = model(images)
        inference_out = out[0] if isinstance(out, tuple) else out

        preds = non_max_suppression(
            inference_out,
            confidence_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
        )
        targets_per_image = build_val_targets(images, targets, cfg.image_size, device)
        accumulator.update(preds, targets_per_image)

    results = accumulator.compute()
    print("\n=== Résultats ===")
    print(f"  Precision      : {results['precision']:.4f}")
    print(f"  Recall         : {results['recall']:.4f}")
    print(f"  mAP@0.5        : {results['map50']:.4f}")
    print(f"  mAP@0.5:0.95   : {results['map']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Chemin vers eval.yaml')
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    evaluate(cfg)


if __name__ == '__main__':
    main()

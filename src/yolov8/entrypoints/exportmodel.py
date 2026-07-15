"""Export a trained YOLOv8 checkpoint to ONNX.

Usage:
    ylexport --config gpu/configs/export.yaml

Steps: load the .pt checkpoint, export with the stable TorchScript
exporter, check the graph, optionally simplify it (onnxsim), verify
the output against PyTorch, and optionally convert to FP16.
"""

import argparse
from pathlib import Path

import torch
from loguru import logger

from yolov8.config import load_export_config, ExportConfig
from yolov8.devices import resolve_device
from yolov8.logging import (setup_logging, log_model_summary,
                            safe_torch_load)
from yolov8.model import YOLO
from yolov8.onnx_export import (YoloExportWrapper, export_to_onnx,
                                check_onnx_graph, simplify_onnx,
                                convert_to_fp16, verify_numerical)


def load_model(cfg: ExportConfig, device):
    model = YOLO(version=cfg.version, num_classes=cfg.num_classes,
                 input_size=cfg.image_size).to(device)
    weights_path = Path(cfg.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = safe_torch_load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        # Strict load: exporting a partly loaded model would give a
        # silently broken ONNX file.
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"Weights '{weights_path}' do not match the model "
            f"(version={cfg.version}, classes={cfg.num_classes}).\n"
            f"  Detail: {e}") from e
    logger.info(f"Weights loaded: {weights_path}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_export(cfg: ExportConfig):
    device = resolve_device(cfg.device)
    logger.info(f"device={device}")

    model = load_model(cfg, device)
    logger.info("===== model architecture =====")
    log_model_summary(
        model, input_size=(1, 3, cfg.image_size, cfg.image_size),
        device=device)
    wrapper = YoloExportWrapper(model).to(device).eval()

    dummy_input = torch.zeros(1, 3, cfg.image_size, cfg.image_size,
                              dtype=torch.float32, device=device)
    output_path = Path(cfg.output_path)
    export_to_onnx(wrapper, dummy_input, output_path,
                   opset=cfg.opset, dynamic=cfg.dynamic)

    onnx_model = check_onnx_graph(output_path) if cfg.check else None
    if cfg.simplify:
        # Simplify before FP16 so the verify step stays exact.
        simplify_onnx(output_path, onnx_model=onnx_model)
    if cfg.verify:
        verify_numerical(wrapper, output_path, dummy_input,
                         cfg.verify_tolerance)
    if cfg.half:
        convert_to_fp16(output_path)

    logger.info(f"Export finished: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained YOLOv8 model to ONNX.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to export.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    cfg = load_export_config(args.config)
    run_export(cfg)


if __name__ == '__main__':
    main()

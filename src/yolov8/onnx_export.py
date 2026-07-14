"""Export a trained YOLOv8 model to ONNX, with checks."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from yolov8.model import MyYolo


class YoloExportWrapper(nn.Module):
    """Minimal wrapper for the ONNX export.

    In eval mode MyYolo returns (inference_out, raw_outputs). For a
    clean ONNX graph we only keep `inference_out`:
      (B, 4 + num_classes, num_anchors)
        - first 4 rows: (cx, cy, w, h) in image space (already * stride)
        - next rows: class scores after sigmoid, in [0, 1]
    """

    def __init__(self, model: MyYolo):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


def export_to_onnx(wrapper, dummy_input, output_path, opset=17,
                   dynamic=True):
    """Run the ONNX export with the legacy (TorchScript) exporter.

    The legacy exporter stays the most portable choice: the new
    "dynamo" exporter needs onnxscript and sometimes renames outputs.
    """
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch', 2: 'anchors'},
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Export: opset={opset} | dynamic_batch={dynamic}")
    logger.info(f"Input shape: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        test_out = wrapper(dummy_input)
    logger.info(f"Output shape (PyTorch reference): "
                f"{tuple(test_out.shape)}")

    kwargs = dict(
        export_params=True, opset_version=opset,
        do_constant_folding=True, input_names=['images'],
        output_names=['output'], dynamic_axes=dynamic_axes)
    try:
        torch.onnx.export(wrapper, dummy_input, str(output_path),
                          dynamo=False, **kwargs)
    except TypeError:
        # PyTorch < 2.5 has no `dynamo` argument.
        torch.onnx.export(wrapper, dummy_input, str(output_path),
                          **kwargs)
    size_mb = output_path.stat().st_size / 1e6
    logger.success(f"ONNX file written: {output_path} ({size_mb:.2f} MB)")


def check_onnx_graph(onnx_path):
    """Validate the ONNX graph. Returns the model or None."""
    try:
        import onnx
    except ImportError:
        logger.warning("onnx not installed, check skipped "
                       "(pip install onnx)")
        return None
    model_onnx = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_onnx)
    logger.success(f"Valid ONNX graph "
                   f"(ir_version={model_onnx.ir_version}, "
                   f"opset={model_onnx.opset_import[0].version})")
    return model_onnx


def simplify_onnx(onnx_path, onnx_model=None):
    """Simplify the graph with onnxsim when available."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        logger.warning("onnxsim not installed, simplify skipped "
                       "(pip install onnxsim)")
        return
    if onnx_model is None:
        onnx_model = onnx.load(str(onnx_path))
    try:
        simplified, ok = onnxsim.simplify(onnx_model)
        if not ok:
            logger.warning("onnxsim reported a failure; "
                           "graph NOT replaced.")
            return
        onnx.save(simplified, str(onnx_path))
        new_size = Path(onnx_path).stat().st_size / 1e6
        logger.success(f"Graph simplified and saved ({new_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Simplify failed: {e}")


def convert_to_fp16(onnx_path):
    """Convert the ONNX model to FP16."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        logger.warning("onnxconverter-common not installed, half "
                       "skipped (pip install onnxconverter-common)")
        return
    try:
        model = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=False)
        onnx.save(model_fp16, str(onnx_path))
        new_size = Path(onnx_path).stat().st_size / 1e6
        logger.success(f"FP16 conversion done ({new_size:.2f} MB)")
    except Exception as e:
        logger.error(f"Half conversion failed: {e}")


def verify_numerical(wrapper, onnx_path, dummy_input, tolerance):
    """Compare the PyTorch output with ONNX Runtime on the same input."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed, verify skipped "
                       "(pip install onnxruntime)")
        return True

    with torch.no_grad():
        torch_out = wrapper(dummy_input).cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_out = session.run(
        None, {'images': dummy_input.cpu().numpy()})[0]

    if torch_out.shape != onnx_out.shape:
        logger.error(f"Verify FAILED: shapes differ "
                     f"(torch={torch_out.shape}, onnx={onnx_out.shape})")
        return False

    abs_diff = np.abs(torch_out - onnx_out)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    ok = max_diff <= tolerance
    if ok:
        logger.success(f"Verify OK | max_abs_diff={max_diff:.2e} "
                       f"mean_abs_diff={mean_diff:.2e} "
                       f"(tol={tolerance:.0e})")
    else:
        logger.error(f"Verify FAILED | max_abs_diff={max_diff:.2e} "
                     f"mean_abs_diff={mean_diff:.2e} "
                     f"(tol={tolerance:.0e})")
        logger.warning("Try again with simplify=False to find the "
                       "source of the numeric gap.")
    return ok

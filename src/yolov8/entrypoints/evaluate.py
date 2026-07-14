"""Full evaluation entrypoint.

Usage:
    yleval --config gpu/configs/eval.yaml

Outputs (in runs/<run_name>/eval[i]/):
  - results.csv     global metrics (mAP, macro/micro P R F1, losses,
                    best confidence threshold, TP/FP/FN counts)
  - per_class.csv   per class metrics
  - plotes/         PR curve, F1-confidence curve, confusion matrices
  - renders/        example predictions drawn on test images
  - config_used.yaml, logs/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from yolov8.config import load_eval_config, config_to_dict, EvalConfig
from yolov8.dataset import collate_detection_batch
from yolov8.dataset.factory import (build_test_dataset,
                                    resolve_class_names)
from yolov8.devices import resolve_device
from yolov8.logging import (setup_logging, add_file_logging,
                            log_model_summary, safe_torch_load)
from yolov8.lossfn import ComputeLoss
from yolov8.metrics import (non_max_suppression, build_val_targets,
                            box_iou_numpy, match_predictions_to_gt,
                            compute_ap_per_class,
                            find_best_f1_threshold,
                            metrics_at_threshold,
                            build_confusion_matrix,
                            build_per_class_table, build_global_table)
from yolov8.model import MyYolo
from yolov8.plotting import (plot_pr_curves, plot_f1_confidence,
                             plot_confusion_matrix)
from yolov8.training import prepare_run_dir, save_config_used


def load_model(cfg: EvalConfig, num_classes, device):
    """Build the model and load the weights with a strict check."""
    model = MyYolo(version=cfg.model.version, num_classes=num_classes,
                   input_size=cfg.dataset.image_size).to(device)
    weights_path = Path(cfg.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = safe_torch_load(weights_path, map_location=device)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        # Strict load: evaluating a partly loaded model would output
        # absurd metrics without any visible error.
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"Weights '{weights_path}' do not match the model "
            f"(version={cfg.model.version}, classes={num_classes}). "
            f"\n  Detail: {e}") from e
    logger.info(f"Weights loaded: {weights_path}")
    model.eval()
    return model


def tensor_to_bgr(image_tensor):
    """CHW RGB float tensor in [0, 1] -> HWC BGR uint8 image."""
    img = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_predictions(image_tensor, preds, class_names, output_path,
                       conf_threshold=0.25):
    """Draw the predicted boxes on one image and save it."""
    img = tensor_to_bgr(image_tensor)
    for row in preds.cpu().numpy():
        x1, y1, x2, y2, conf, cls = row
        if conf < conf_threshold:
            continue
        c = int(cls)
        color = _class_color(c)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, 2)
        name = class_names[c] if c < len(class_names) else str(c)
        label = f"{name} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)
    cv2.imwrite(str(output_path), img)


def _class_color(c):
    """Stable pseudo random BGR color for a class id."""
    rng = np.random.RandomState(c + 7)
    return tuple(int(v) for v in rng.randint(64, 255, size=3))


@torch.no_grad()
def collect_predictions(model, loader, device, cfg, loss_fn, iou_v,
                        renders_dir, class_names):
    """Single pass over the test set: predictions, losses, matches.

    One pass is enough because the TP/FP flags are stored per IoU
    threshold: the curves are interpolated afterwards without a new
    inference run.
    """
    data = {'tp': [], 'conf': [], 'cls': [], 'target_cls': [],
            'preds_cm': [], 'gts_cm': []}
    losses_sum = {'box': 0.0, 'cls': 0.0, 'dfl': 0.0, 'total': 0.0}
    n_samples = 0
    iou_sum_tp, iou_count_tp = 0.0, 0
    rendered = 0

    pbar = tqdm(loader, desc="[eval]", leave=False, dynamic_ncols=True,
                ascii="░█")
    for images, targets, _paths in pbar:
        images = images.to(device, non_blocking=True)
        out = model(images)
        inference_out, raw_outputs = out if isinstance(out, tuple) \
            else (out, None)

        if raw_outputs is not None:
            lb, lc, ld = loss_fn(raw_outputs, targets)
            bs = images.size(0)
            losses_sum['box'] += float(lb.item()) * bs
            losses_sum['cls'] += float(lc.item()) * bs
            losses_sum['dfl'] += float(ld.item()) * bs
            losses_sum['total'] += float((lb + lc + ld).item()) * bs
            n_samples += bs

        preds_per_image = non_max_suppression(
            inference_out, confidence_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold)
        gts_per_image = build_val_targets(
            images, targets, cfg.dataset.image_size, device)

        for i, (preds, gt) in enumerate(
                zip(preds_per_image, gts_per_image)):
            stats = _match_one_image(preds, gt, iou_v)
            for key, value in stats['store'].items():
                data[key].append(value)
            iou_sum_tp += stats['iou_sum']
            iou_count_tp += stats['iou_count']
            if rendered < cfg.n_renders:
                render_predictions(
                    images[i], preds, class_names,
                    renders_dir / f"render_{rendered:03d}.jpg")
                rendered += 1

    return _finalize_collection(data, losses_sum, n_samples,
                                iou_sum_tp, iou_count_tp, iou_v)


def _match_one_image(preds, gt, iou_v):
    """TP/FP matching of one image. Returns arrays and IoU sums."""
    preds = preds.cpu().numpy()
    gt = gt.cpu().numpy()
    preds_xyxy = preds[:, :4] if preds.size else np.zeros((0, 4))
    pred_conf = preds[:, 4] if preds.size else np.zeros(0)
    pred_cls = preds[:, 5].astype(int) if preds.size \
        else np.zeros(0, dtype=int)
    gts_xyxy = gt[:, 1:5] if gt.size else np.zeros((0, 4))
    gt_cls = gt[:, 0].astype(int) if gt.size \
        else np.zeros(0, dtype=int)

    tp = match_predictions_to_gt(preds_xyxy, pred_cls, pred_conf,
                                 gts_xyxy, gt_cls, iou_v)

    iou_sum, iou_count = 0.0, 0
    if preds_xyxy.shape[0] and gts_xyxy.shape[0] and tp[:, 0].any():
        iou_mat = box_iou_numpy(preds_xyxy, gts_xyxy)
        cls_match = pred_cls[:, None] == gt_cls[None, :]
        iou_mat = np.where(cls_match, iou_mat, 0.0)
        for i in np.where(tp[:, 0])[0]:
            iou_max = float(iou_mat[i].max()) if iou_mat[i].size else 0.0
            if iou_max > 0:
                iou_sum += iou_max
                iou_count += 1

    return {
        'store': {
            'tp': tp, 'conf': pred_conf, 'cls': pred_cls,
            'target_cls': gt_cls,
            'preds_cm': (preds_xyxy, pred_cls, pred_conf),
            'gts_cm': (gts_xyxy, gt_cls),
        },
        'iou_sum': iou_sum, 'iou_count': iou_count,
    }


def _finalize_collection(data, losses_sum, n_samples, iou_sum_tp,
                         iou_count_tp, iou_v):
    if data['tp']:
        tp_matrix = np.concatenate(data['tp'], axis=0)
        conf = np.concatenate(data['conf'], axis=0)
        pred_cls = np.concatenate(data['cls'], axis=0)
    else:
        tp_matrix = np.zeros((0, len(iou_v)), dtype=bool)
        conf = np.zeros(0)
        pred_cls = np.zeros(0, dtype=int)
    target_cls = np.concatenate(data['target_cls'], axis=0) \
        if data['target_cls'] else np.zeros(0, dtype=int)

    losses = {k: (v / n_samples if n_samples else 0.0)
              for k, v in losses_sum.items()}
    iou_mean = iou_sum_tp / iou_count_tp if iou_count_tp else 0.0
    return {
        'tp_matrix': tp_matrix, 'conf': conf, 'pred_cls': pred_cls,
        'target_cls': target_cls, 'all_preds_cm': data['preds_cm'],
        'all_gts_cm': data['gts_cm'], 'losses': losses,
        'iou_mean': iou_mean,
    }


def cls_loss_per_class_proxy(data, num_classes, eps=1e-9):
    """BCE-style per class loss proxy from the NMS confidences.

    TP: -log(conf) (we wanted conf = 1). FP: -log(1 - conf) (we
    wanted conf = 0). A quick way to spot the hard classes.
    """
    sums = np.zeros(num_classes, dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    conf = data['conf']
    pred_cls = data['pred_cls']
    tp_50 = data['tp_matrix'][:, 0] if data['tp_matrix'].size \
        else np.zeros(0, dtype=bool)
    for i in range(len(conf)):
        c = int(pred_cls[i])
        if not 0 <= c < num_classes:
            continue
        if tp_50[i]:
            sums[c] += -float(np.log(max(conf[i], eps)))
        else:
            sums[c] += -float(np.log(max(1.0 - conf[i], eps)))
        counts[c] += 1
    return {c: float(sums[c] / counts[c]) if counts[c] else 0.0
            for c in range(num_classes)}


def evaluate(cfg: EvalConfig, run_dir):
    device = resolve_device(cfg.device)
    logger.info(f"device={device}")
    plotes_dir = Path(run_dir) / 'plotes'
    renders_dir = Path(run_dir) / 'renders'
    renders_dir.mkdir(exist_ok=True)

    test_ds = build_test_dataset(cfg.dataset, seed=cfg.seed)
    class_names = resolve_class_names(cfg.dataset, test_ds)
    num_classes = len(class_names)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_detection_batch, drop_last=False)
    logger.info(f"Data: test={len(test_ds)} | classes={num_classes}")

    model = load_model(cfg, num_classes, device)
    logger.info("===== model architecture =====")
    log_model_summary(
        model, input_size=(1, 3, cfg.dataset.image_size,
                           cfg.dataset.image_size), device=device)
    loss_fn = ComputeLoss(model, cfg.loss.gains())

    # COCO convention: 10 IoU thresholds from 0.5 to 0.95.
    iou_v = np.linspace(0.5, 0.95, 10)
    logger.info("Evaluation loop...")
    data = collect_predictions(model, loader, device, cfg, loss_fn,
                               iou_v, renders_dir, class_names)

    logger.info("Computing per class AP...")
    per_class = compute_ap_per_class(
        data['tp_matrix'], data['conf'], data['pred_cls'],
        data['target_cls'], num_classes=num_classes)

    px = np.linspace(0, 1, 1000)
    best_conf, best_f1 = find_best_f1_threshold(per_class, px)
    logger.info(f"Best confidence threshold: {best_conf:.4f} "
                f"(macro F1 = {best_f1:.4f})")
    metrics_thr = metrics_at_threshold(per_class, px, best_conf)

    counts = _global_counts(data, best_conf)
    _write_tables(run_dir, per_class, metrics_thr, data, counts,
                  best_conf, best_f1, class_names, num_classes)
    _write_figures(plotes_dir, per_class, class_names, data,
                   num_classes, best_conf)
    logger.success(f"Evaluation done. Results in: {run_dir}")


def _global_counts(data, best_conf):
    keep = data['conf'] >= best_conf
    tp_filtered = data['tp_matrix'][keep, 0] if data['tp_matrix'].size \
        else np.zeros(0, dtype=bool)
    n_tp = int(tp_filtered.sum())
    n_gt = int(data['target_cls'].size)
    return {
        'n_gt': n_gt,
        'n_pred': int(data['pred_cls'].size),
        'n_tp': n_tp,
        'n_fp': int((~tp_filtered).sum()),
        'n_fn': max(0, n_gt - n_tp),
    }


def _write_tables(run_dir, per_class, metrics_thr, data, counts,
                  best_conf, best_f1, class_names, num_classes):
    cls_loss = cls_loss_per_class_proxy(data, num_classes)
    df_per_class = build_per_class_table(
        per_class, metrics_thr, class_names,
        cls_loss_per_class=cls_loss)
    df_global = build_global_table(
        per_class, metrics_thr, data['losses'], data['iou_mean'],
        best_conf, best_f1, counts)

    per_class_csv = Path(run_dir) / 'per_class.csv'
    results_csv = Path(run_dir) / 'results.csv'
    df_per_class.to_csv(per_class_csv, index=False)
    df_global.to_csv(results_csv, index=False)
    logger.success(f"CSV written: {results_csv}")
    logger.success(f"CSV written: {per_class_csv}")

    _log_table(df_global)
    if len(df_per_class) > 0:
        logger.info("Top 5 classes by AP@0.5:")
        _log_table(df_per_class.nlargest(5, 'ap50'))
        if len(df_per_class) > 5:
            logger.info("Bottom 5 classes by AP@0.5:")
            _log_table(df_per_class.nsmallest(5, 'ap50'))


def _log_table(df):
    for line in df.to_string(index=False).splitlines():
        logger.info(f"  {line}")


def _write_figures(plotes_dir, per_class, class_names, data,
                   num_classes, best_conf):
    logger.info("Writing figures...")
    plot_pr_curves(per_class, class_names, plotes_dir / 'pr_curve.png')
    plot_f1_confidence(per_class, class_names,
                       plotes_dir / 'f1_confidence.png')
    cm = build_confusion_matrix(
        data['all_preds_cm'], data['all_gts_cm'],
        num_classes=num_classes, iou_threshold=0.45,
        conf_threshold=best_conf)
    plot_confusion_matrix(
        cm, class_names,
        plotes_dir / 'confusion_matrix_normalized.png', normalize=True)
    plot_confusion_matrix(
        cm, class_names, plotes_dir / 'confusion_matrix.png',
        normalize=False)
    logger.success(f"Figures written in: {plotes_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on the full test set.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to eval.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger.info("Starting evaluation pipeline")
    cfg = load_eval_config(args.config)

    run_dir, _ = prepare_run_dir(cfg.output_dir, cfg.run_name,
                                 kind='eval', resume=False)
    add_file_logging(run_dir / 'logs', prefix='eval',
                     level=args.log_level)
    save_config_used(run_dir, config_to_dict(cfg))

    evaluate(cfg, run_dir)


if __name__ == '__main__':
    main()

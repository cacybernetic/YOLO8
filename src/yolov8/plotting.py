"""Plot helpers: training history and evaluation figures.

All plots use the non interactive 'Agg' backend, so they work on
headless servers.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_training_history(history, output_path):
    """Draw the train/val loss history on 4 subplots.

    Subplots (2x2): total loss, box loss, cls loss, dfl loss. These
    curves are the main tool to check convergence and overfitting:
    the val curve going up while the train curve goes down means the
    model is overfitting.

    Args:
        history: dict with the keys
            'train_loss', 'train_box', 'train_cls', 'train_dfl',
            'val_loss', 'val_box', 'val_cls', 'val_dfl' (float lists)
            'epochs_train', 'epochs_val' (1-indexed int lists)
        output_path: destination PNG path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (axes[0, 0], 'Total loss', 'train_loss', 'val_loss'),
        (axes[0, 1], 'Box loss (CIoU)', 'train_box', 'val_box'),
        (axes[1, 0], 'Classification loss', 'train_cls', 'val_cls'),
        (axes[1, 1], 'DFL loss', 'train_dfl', 'val_dfl'),
    ]
    epochs_train = history.get('epochs_train', [])
    epochs_val = history.get('epochs_val', [])

    for ax, title, k_train, k_val in panels:
        _plot_history_panel(ax, title, history.get(k_train, []),
                            history.get(k_val, []),
                            epochs_train, epochs_val)

    fig.suptitle("Training history", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_history_panel(ax, title, train_vals, val_vals,
                        epochs_train, epochs_val):
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
    ax.set_ylabel('loss', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)


def plot_pr_curves(per_class, class_names, output_path,
                   title="Precision-Recall curve"):
    """PR curves per class plus the mean curve (mAP@0.5)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    if not per_class:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center',
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    for c, d in per_class.items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        idx = np.argsort(d['r_curve'])
        ax.plot(d['r_curve'][idx], d['p_curve'][idx], linewidth=1,
                alpha=0.5, label=f"{name} (AP@.5={d['ap50']:.3f})")

    all_p = np.stack([v['p_curve'] for v in per_class.values()], axis=0)
    all_r = np.stack([v['r_curve'] for v in per_class.values()], axis=0)
    mean_p, mean_r = all_p.mean(axis=0), all_r.mean(axis=0)
    idx = np.argsort(mean_r)
    map50 = float(np.mean([v['ap50'] for v in per_class.values()]))
    ax.plot(mean_r[idx], mean_p[idx], 'b-', linewidth=2.5,
            label=f"all classes (mAP@.5={map50:.3f})")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend_or_mean_only(ax, len(per_class))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_f1_confidence(per_class, class_names, output_path,
                       title="F1-Confidence curve"):
    """F1 = f(confidence) with the best threshold marked."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.linspace(0, 1, 1000)
    if not per_class:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    for c, d in per_class.items():
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        ax.plot(x, d['f1_curve'], linewidth=1, alpha=0.5, label=name)

    f1_stack = np.stack(
        [v['f1_curve'] for v in per_class.values()], axis=0)
    f1_mean = f1_stack.mean(axis=0)
    ax.plot(x, f1_mean, 'b-', linewidth=2.5, label='all classes')

    best_idx = int(np.argmax(f1_mean))
    best_conf = float(x[best_idx])
    best_f1 = float(f1_mean[best_idx])
    ax.axvline(x=best_conf, color='r', linestyle='--', linewidth=1.5,
               label=f'best threshold: {best_conf:.3f} '
                     f'(F1={best_f1:.3f})')
    ax.scatter([best_conf], [best_f1], color='r', s=80, zorder=5)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1 score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _legend_or_mean_only(ax, len(per_class), loc='lower center', last=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _legend_or_mean_only(ax, n_classes, loc='lower left', last=1):
    """Full legend for few classes, mean curve only for many."""
    if n_classes <= 20:
        ax.legend(loc=loc, fontsize=7, framealpha=0.9)
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-last:], labels[-last:], loc=loc, fontsize=8)


def plot_confusion_matrix(cm, class_names, output_path, normalize=True,
                          title="Confusion matrix"):
    """Plot the (N+1) x (N+1) confusion matrix (last class: background).

    When normalize is True, each column is divided by its sum:
    cell (i, j) reads "which part of the
    ground truths of class j was predicted as class i".
    """
    nc = len(class_names)
    full_names = list(class_names) + ['background']

    cm_display = cm.astype(np.float64)
    if normalize:
        col_sums = cm_display.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm_display = cm_display / col_sums

    fig_size = max(8, min(20, 0.5 * (nc + 1) + 4))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_display, cmap='Blues', aspect='auto',
                   vmin=0, vmax=1.0 if normalize else cm.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(full_names)))
    ax.set_yticks(np.arange(len(full_names)))
    ax.set_xticklabels(full_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(full_names, fontsize=11)
    ax.set_xlabel('True class', fontsize=12)
    ax.set_ylabel('Predicted class', fontsize=12)
    suffix = ' (normalized)' if normalize else ' (counts)'
    ax.set_title(title + suffix, fontsize=13)

    if nc <= 30:
        _annotate_cells(ax, cm_display, normalize)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _annotate_cells(ax, cm_display, normalize):
    threshold = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            v = cm_display[i, j]
            if v > 0:
                text = f"{v:.2f}" if normalize else f"{int(v)}"
                color = 'white' if v > threshold else 'black'
                ax.text(j, i, text, ha='center', va='center',
                        color=color, fontsize=9)

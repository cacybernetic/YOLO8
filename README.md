<div align="center">

# <img src="logo.png" width="50px" height="25px"/>v8 From Scratch

![](https://img.shields.io/badge/STATUS-stable-brightgreen)
![](https://img.shields.io/badge/Python-3.10-blue)
![](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![](https://img.shields.io/badge/Rust-edition--2024-b7410e)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/latest-2026--07--15-green)

</div>

A complete YOLOv8 object detection pipeline implemented from scratch in
PyTorch — fault-tolerant training, evaluation, fine-tuning, HDF5 dataset
builds, ONNX export, and inference. Bonus: a Rust binary for fast ONNX
inference on static images or live webcam feed.

<br>

**Table of Contents**

- [Description](#description)
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Dataset format](#dataset-format)
- [Usage](#usage)
  - [1. Build HDF5 datasets (optional)](#1-build-hdf5-datasets-optional)
  - [2. Train](#2-train)
  - [3. Evaluate](#3-evaluate)
  - [4. Fine-tune on new classes](#4-fine-tune-on-new-classes)
  - [5. Export to ONNX](#5-export-to-onnx)
  - [6. Run inference](#6-run-inference)
  - [7. Standalone ONNX scripts](#7-standalone-onnx-scripts-predictpy-and-livepy)
  - [8. Run inference in Rust](#8-run-inference-in-rust)
- [Configuration files](#configuration-files)
- [Fault-tolerant training](#fault-tolerant-training)
- [Documentation](#documentation)
- [To contribute](#to-contribute)
- [Licence](#licence)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

---

## Description

This project is a full re-implementation of YOLOv8 in pure PyTorch — no
Ultralytics dependency. It is designed to be readable, hackable, and easy
to run on your own dataset. Every component (backbone, neck, head, loss,
metrics, augmentations, trainer) is written from scratch and documented.

A companion Rust binary lets you run ONNX inference at native speed,
either on a single image or in real-time from a webcam.

## Features

- **Datasets as folders or zip archives**, with a `data.yaml` class list,
  a pre-flight validation scan and a JSON scan cache
  (`train.cache.json`) for instant restarts.
- **HDF5 dataset builds** (`buildh5ds`): pre-compute letterboxed (and
  optionally augmented) samples into `train.h5` / `test.h5` and train
  straight from them (`use_hdf5: true`).
- **Fault-tolerant training**: a checkpoint every `ckpt_step` optimizer
  steps (and during val/test passes) captures the model, optimizer,
  EMA, AMP scaler, dataloader positions, partial meters and RNG states.
  Training resumes mid-epoch without ever seeing a sample twice.
- **Resumable DataLoader adapter**: the shuffle order is a pure function
  of (seed + epoch), so the exact epoch order is rebuilt after a crash.
- **Structured run folders**: `runs/<name>/train`, `train2`, ... and
  `eval`, `eval2`, ... each with `weights/`, `checkpoints/`, `plotes/`,
  `logs/`, `history.csv` and `config_used.yaml`.
- **Disjoint validation / final test**: `val_prob` takes a
  deterministic fraction of the test split for per-epoch validation
  (model selection, early stopping); the final evaluation runs on the
  **held-out remainder**, so the reported test metrics are never
  computed on samples that drove the model selection.
- Rich augmentations: **mosaic** (with `close_mosaic`), HSV jitter,
  affine transforms, MixUp, Cutout, blur, noise, grayscale.
- Cosine and linear LR schedulers with full warm-up (LR, per-group bias
  LR, momentum); SGD / Adam / AdamW.
- **EMA** weights for validation and `best.pt`; **AMP** on CUDA;
  gradient accumulation with end-of-epoch flush; early stopping.
- **Full COCO-style evaluation**: mAP@0.5, mAP@0.5:0.95, macro/micro
  P/R/F1, PR and F1-confidence curves, confusion matrices, prediction
  renders.
- **Export** to ONNX (with optional FP16, graph simplification, and
  numerical verification) and a fully standalone ONNX inference script.
- Unit tests for the model, loss, metrics, dataset, adapter, schedulers
  and the trainer (including mid-epoch resume).

## Project structure

```
.
├── src/yolov8/
│   ├── modules/          # Conv, C2f, SPPF, DFL, Backbone, Neck, Head
│   ├── model.py          # MyYolo (backbone + neck + head)
│   ├── lossfn.py         # TAL assigner + CIoU + DFL + BCE loss
│   ├── dataset/
│   │   ├── sources.py    # folder and zip dataset sources
│   │   ├── scanner.py    # validation scan + JSON cache
│   │   ├── transforms.py # letterbox, tensor conversion
│   │   ├── augment.py    # all augmentations + Augmenter
│   │   ├── yolo_dataset.py
│   │   ├── hdf5_store.py # HDF5 build + read
│   │   ├── adapter.py    # resumable DataLoader adapter
│   │   └── factory.py    # dataset construction from the config
│   ├── metrics/          # NMS, mAP, per-class AP, confusion matrix
│   ├── training/
│   │   ├── optimizers.py # param groups, SGD/Adam/AdamW, freezing
│   │   ├── lr_schedulers.py
│   │   ├── ema.py
│   │   ├── meters.py     # resumable loss meters
│   │   ├── checkpoints.py# naming, rotation, RNG capture
│   │   ├── runs.py       # runs/<name>/train[i] folders
│   │   └── trainer.py    # the fault-tolerant training loop
│   ├── logging.py        # loguru setup + torchinfo summary
│   ├── plotting.py       # history and evaluation figures
│   ├── onnx_export.py
│   ├── config.py         # nested dataclass configs + YAML loaders
│   └── entrypoints/
│       ├── buildds.py    # build HDF5 datasets
│       ├── train.py      # train + val + final test
│       ├── evaluate.py   # full evaluation
│       ├── exportmodel.py# ONNX export
│       ├── inference.py  # standalone ONNX inference
│       └── finetuning.py # build a fine-tunable checkpoint
├── cpu/configs/          # ready-made configs for CPU
├── gpu/configs/          # ready-made configs for NVIDIA CUDA / AMD ROCm
├── tests/                # pytest unit and integration tests
├── docs/
│   ├── en_concepts.md    # beginner-friendly concept guide (English)
│   └── fr_concepts.md    # the same guide in French
└── src/main.rs           # Rust ONNX inference binary
```

## Installation

### Quick install (without cloning)

```bash
pip install git+https://github.com/cacybernetic/YOLO8
# or, faster:
uv pip install git+https://github.com/cacybernetic/YOLO8
```

This registers the CLI tools (`trainyolo8`, `evalyolo8`, `runyolo8`,
`ftyolo8`, `exportw`, `buildh5ds`). Download the configuration files
from [cpu/configs/](cpu/configs/) or [gpu/configs/](gpu/configs/).

### Python — Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
git clone https://github.com/cacybernetic/YOLO8
cd YOLO8
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

> **Note — headless server (no display):**
> ```bash
> sudo apt-get install libgl1-mesa-glx libglib2.0-0
> ```

### Python — Windows

```bash
pip install uv
uv venv --python 3.10
.venv\Scripts\activate
uv pip install -e .
```

### Rust (optional)

Only needed for the Rust ONNX inference binary.

1. Install Rust: [rustup.rs](https://rustup.rs/)
2. `cargo build --release`

The binary is compiled to `target/release/yolov8rust`.

---

## Dataset format

A dataset split is either a **folder** or a **zip archive** with this
layout:

```
train.zip (or train/)
├── images/     # .jpg, .png, .jpeg, ...
├── labels/     # one .txt per image
└── data.yaml   # class names
```

`data.yaml` must contain at least a `names` list (`nc` is optional):

```yaml
nc: 10
names:
- door
- cabinetDoor
- refrigeratorDoor
- window
- chair
- table
- cabinet
- couch
- openedDoor
- pole
```

Each `.txt` label file contains one object per line, values normalized
to [0, 1]:

```
<class_id> <cx> <cy> <w> <h>
```

Train and test are given separately (`train_path`, `test_path`). The
test split is divided into two **disjoint** parts with `val_prob`
(default 0.5): a validation part used for the per-epoch metrics and
the model selection, and a held-out part used for the final
evaluation. Sample caps are available via `max_train_samples` /
`max_test_samples`.

Before anything runs, the dataset is scanned: corrupt images, missing
labels and malformed lines are dropped with a report, and the result is
cached in `train.cache.json` / `test.cache.json` next to the dataset.

---

## Usage

Six commands are installed:

| Command      | Role                                    |
|--------------|-----------------------------------------|
| `buildh5ds`  | Build HDF5 datasets (`train.h5`, `test.h5`) |
| `trainyolo8` | Train a model (train + val + final test) |
| `evalyolo8`  | Evaluate a model on the full test set   |
| `runyolo8`   | Standalone ONNX inference on an image   |
| `ftyolo8`    | Build a fine-tunable checkpoint         |
| `exportw`    | Export to ONNX                          |

Each command takes a single `--config` argument pointing to its YAML
file. Pick the folder matching your hardware: `cpu/configs/` or
`gpu/configs/`.

### 1. Build HDF5 datasets (optional)

```bash
buildh5ds --config gpu/configs/hdf5.yaml
```

Then set `use_hdf5: true` in `train.yaml` to train from `train.h5` /
`test.h5`. Use `augmented_copies: N` to also bake N augmented copies of
each train sample into the file.

### 2. Train

Edit `gpu/configs/train.yaml` (dataset paths, `run_name`), then:

```bash
trainyolo8 --config gpu/configs/train.yaml
```

Everything lands in `runs/<run_name>/train[i]/`:
- `weights/best.pt` (EMA weights of the best epoch) and
  `weights/last.pt`
- `checkpoints/checkpoint_eXXXXcYYYY.pth` — fault-tolerance snapshots
- `plotes/training_history.png`, `history.csv`, `logs/`,
  `config_used.yaml`
- `test_results.csv` — final evaluation on the held-out test part

With `resume: true` (default), restarting the same command reuses the
latest run folder and continues from the newest checkpoint — even in
the middle of an epoch or of a validation pass.

### 3. Evaluate

```bash
evalyolo8 --config gpu/configs/eval.yaml
```

Results go to `runs/<run_name>/eval[i]/`: `results.csv` (global
metrics), `per_class.csv`, `plotes/` (PR curve, F1-confidence curve,
confusion matrices) and `renders/` (example predictions).

### 4. Fine-tune on new classes

**Step 1** — build the fine-tunable checkpoint:

```bash
ftyolo8 --config gpu/configs/finetuning.yaml
```

This transfers the backbone, neck, box branches and DFL, and
re-initializes the classification heads for the new class count.

**Step 2** — train as usual with, in `train.yaml`:

```yaml
model:
  pretrained_weights: weights/finetune_init.pt
  freeze_feature_layers: true   # recommended for small datasets
```

### 5. Export to ONNX

```bash
exportw --config gpu/configs/export.yaml
```

The exported `.onnx` file is numerically verified against the PyTorch
model by default.

### 6. Run inference

`runyolo8` is a fully standalone ONNX inference script (numpy +
opencv + onnxruntime only — you can copy
`src/yolov8/entrypoints/inference.py` anywhere and it keeps working):

```bash
runyolo8 --model weights/best.onnx --nc 10 \
         --image photo.jpg --output result.jpg
```

Options: `--conf 0.25`, `--iou 0.45`, `--size 640`, `--show`,
`--names classes.txt` (one class name per line).

### 7. Standalone ONNX scripts (`predict.py` and `live.py`)

Two extra helper scripts at the project root run a pre-trained ONNX
model without any dependency on the `yolov8` package:

```bash
# Single image (numpy + Pillow only)
python predict.py --model weights/best.onnx --nc 80 \
  --image samples/photo.jpg --output result.jpg

# Webcam or video file (OpenCV)
python live.py --model weights/best.onnx --nc 80 --source 0
```

### 8. Run inference in Rust

```bash
./target/release/yolov8rust \
  --model weights/best.onnx --image photo.jpg \
  --output result.jpg --nc 80 --conf 0.25 --iou 0.45

# Live from webcam
./target/release/yololivers --model weights/best.onnx --source 0 --nc 80
```

---

## Configuration files

All behavior is controlled through the YAML files in `cpu/configs/` and
`gpu/configs/`. The train config is nested:

| Block          | Key fields |
|----------------|------------|
| (top level)    | `run_name`, `output_dir`, `device`, `seed`, `log_interval` |
| `dataset`      | `train_path`, `test_path`, `use_hdf5`, `train_h5`, `test_h5`, `validate`, `cache`, `max_train_samples`, `max_test_samples`, `val_prob`, `image_size`, `augment.*` |
| `model`        | `version` (`n/s/m/l/x`), `pretrained_weights`, `freeze_feature_layers` |
| `optimization` | `epochs`, `batch_size`, `optimizer`, `max_lr`, `scheduler`, `grad_accum`, `amp`, `ema`, `patience` |
| `loss`         | `box_gain`, `cls_gain`, `dfl_gain` |
| `checkpoint`   | `ckpt_step`, `max_checkpoint`, `resume`, `best_metric` |
| `validation`   | `interval`, `conf_threshold`, `iou_threshold` |

Unknown keys are reported with a warning and ignored.

## Fault-tolerant training

The training loop checkpoints every `ckpt_step` optimizer steps into
`checkpoints/checkpoint_e<epoch>c<step>.pth` (one file per save point,
rotated with `max_checkpoint`). Each checkpoint stores:

- model, optimizer, EMA, AMP scaler state;
- the position of the three dataloader adapters (train / val / test);
- the partial loss meters and metric accumulators;
- the RNG states (python, numpy, torch, cuda);
- the training history and the config.

On restart with `resume: true`, the highest-numbered run folder holding
a checkpoint is reused and training continues exactly where it stopped:
mid-train-epoch, mid-validation or mid-final-test, without seeing any
sample twice in the same epoch. This makes multi-day epochs safe
against power or system failures.

## Documentation

- [docs/en_concepts.md](docs/en_concepts.md) — beginner-friendly guide
  to every concept used here (English).
- [docs/fr_concepts.md](docs/fr_concepts.md) — le même guide en
  français.

Run the test suite with:

```bash
make test        # or: pytest tests
```

## To contribute

Contributions are welcome! Please follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add a new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request.

## Licence

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.

## Acknowledgments

This project was built while learning the inner workings of YOLOv8. A
huge thank-you to **[dtdo90](https://github.com/dtdo90)** for the
excellent educational repository
[**dtdo90/yolov8_detection**](https://github.com/dtdo90/yolov8_detection)
and the accompanying
**[YouTube walkthrough](https://www.youtube.com/watch?v=6zQP0L-ph0M)**,
both of which served as the primary reference for understanding the
architecture (backbone, neck, head). Many implementation choices in
this project — the structure of the `Detect` head, and the integration
of the DFL into the box regression — are directly inspired by its work.

If you find this project useful, please consider giving the **dtdo90**
repository a star as a token of appreciation for the educational
content that made it possible.

## References

The implementation is based on the following papers and resources:

### Loss function and assignment strategy

- **TAL — Task-Aligned Assigner** — Feng, C., Zhong, Y., Gao, Y., Scott, M. R.,
  & Huang, W. (2021). *TOOD: Task-Aligned One-stage Object Detection*.
  ICCV 2021.
  [arXiv:2108.07755](https://arxiv.org/abs/2108.07755)
- **DFL — Distribution Focal Loss** — Li, X., Wang, W., Wu, L., Chen, S., Hu, X.,
  Li, J., Tang, J., & Yang, J. (2020). *Generalized Focal Loss: Learning
  Qualified and Distributed Bounding Boxes for Dense Object Detection*.
  NeurIPS 2020.
  [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
- **CIoU Loss** — Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020).
  *Distance-IoU Loss: Faster and Better Learning for Bounding Box
  Regression*. AAAI 2020.
  [arXiv:1911.08287](https://arxiv.org/abs/1911.08287)
- **Focal Loss** — Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.
  (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. Used as the
  reference for the bias initialization of the new classification heads
  during fine-tuning (`b = -log((1-π)/π)` with π=0.01).
  [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

### Architecture components

- **CSPNet** — Wang, C.-Y., Liao, H.-Y. M., Wu, Y.-H., Chen, P.-Y., Hsieh, J.-W.,
  & Yeh, I.-H. (2020). *CSPNet: A New Backbone that can Enhance Learning
  Capability of CNN*. CVPRW 2020. Foundation of the C2f blocks used in the
  backbone.
  [arXiv:1911.11929](https://arxiv.org/abs/1911.11929)
- **PAN — Path Aggregation Network** — Liu, S., Qi, L., Qin, H., Shi, J., &
  Jia, J. (2018). *Path Aggregation Network for Instance Segmentation*.
  CVPR 2018. Used as the basis for the multi-scale neck.
  [arXiv:1803.01534](https://arxiv.org/abs/1803.01534)
- **SPPF** — He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Spatial Pyramid
  Pooling in Deep Convolutional Networks for Visual Recognition*. The Fast
  variant (SPPF) used in this project is the standard YOLOv5/v8 design.
  [arXiv:1406.4729](https://arxiv.org/abs/1406.4729)

### Evaluation metrics

- **COCO evaluation protocol** — Lin, T.-Y., Maire, M., Belongie, S., Hays, J.,
  Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014).
  *Microsoft COCO: Common Objects in Context*. ECCV 2014. Source of the
  101-point AP interpolation and the IoU thresholds 0.5:0.05:0.95.
  [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- **Survey on detection metrics** — Padilla, R., Netto, S. L., & da Silva, E. A. B.
  (2020). *A Survey on Performance Metrics for Object-Detection
  Algorithms*. IWSSIP 2020.
  [DOI](https://doi.org/10.1109/IWSSIP48289.2020.9145130)

### Educational reference

- **dtdo90/yolov8_detection** — DT Do (2024). Implementation of the YOLOv8
  detection model with an accompanying YouTube tutorial.

## Contact

For questions or suggestions:

- **Author**: DOCTOR MOKIRA — dr.mokira@gmail.com
- **Maintainer**: CONSOLE ART CYBERNETIC — ca.cybernetic@gmail.com
- **GitHub**: [cacybernetic/YOLO8](https://github.com/cacybernetic/YOLO8)

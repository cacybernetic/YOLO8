<div align="center">

# <img src="logo.png" width="50px" height="25px"/>v8 From Scratch

![](https://img.shields.io/badge/STATUS-stable-brightgreen)
![](https://img.shields.io/badge/Python-3.10-blue)
![](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![](https://img.shields.io/badge/Rust-edition--2024-b7410e)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/latest-2026--04--27-green)

</div>

A complete YOLOv8 object detection pipeline implemented from scratch in PyTorch — training, evaluation, fine-tuning, ONNX export, and inference. Bonus: a Rust binary for fast ONNX inference on static images or live webcam feed.

<br>

**Table of Contents**

- [Description](#description)
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
  - [Quick install](#quick-install-without-cloning)
  - [Python — Linux](#python--linux)
  - [Python — Windows](#python--windows)
  - [Rust (optional)](#rust-optional)
- [Dataset format](#dataset-format)
- [Usage](#usage)
  - [1. Train](#1-train)
  - [2. Evaluate](#2-evaluate)
  - [3. Fine-tune on new classes](#3-fine-tune-on-new-classes)
  - [4. Export to ONNX](#4-export-to-onnx)
  - [5. Run inference on an image](#5-run-inference-on-an-image)
  - [6. Standalone ONNX scripts (`predict.py` and `live.py`)](#6-standalone-onnx-scripts-predictpy-and-livepy)
  - [7. Run inference in Rust](#7-run-inference-in-rust)
- [Configuration files](#configuration-files)
- [To contribute](#to-contribute)
- [Licence](#licence)
- [Acknowledgments](#acknowledgments)
- [References](#references)
- [Contact](#contact)

---

## Description

This project is a full re-implementation of YOLOv8 in pure PyTorch — no Ultralytics dependency. It is designed to be readable, hackable, and easy to run on your own dataset. Every component (backbone, neck, head, loss, metrics, augmentations) is written from scratch and documented.

A companion Rust binary lets you run ONNX inference at native speed, either on a single image or in real-time from a webcam.

## Features

- **Train from scratch** on any dataset in YOLO format.
- **Fine-tune** a pre-trained model on a new set of classes in a few lines of config.
- **Evaluate** with full COCO-style metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, confusion matrix, PR curves.
- **Export** to ONNX (with optional FP16, graph simplification, and numerical verification).
- **Inference** on images from Python with a futuristic box renderer.
- **Rust binary** for fast ONNX inference on images or live webcam (`src/main.rs`).
- Rich augmentations: HSV jitter, affine transforms, MixUp, Cutout, blur, noise, grayscale.
- Cosine and linear LR schedulers with warm-up.
- Gradient accumulation, automatic checkpoint rotation, training history plots.

## Project structure

```
.
├── yolov8/
│   ├── model.py          # Backbone, Neck, Head, MyYolo
│   ├── lossfn.py         # TAL assigner + CIoU + DFL + BCE loss
│   ├── dataset.py        # YOLODataset with augmentations
│   ├── metrics.py        # NMS, mAP, MetricAccumulator
│   ├── metrics_eval.py   # Full evaluation suite (curves, CSV, confusion matrix)
│   ├── config.py         # Dataclasses + YAML loaders
│   ├── utils.py          # Logging, model summary, history plot
│   └── entrypoints/
│       ├── train.py      # Training loop
│       ├── evaluate.py   # Full evaluation
│       ├── infer.py      # Single-image inference
│       ├── export.py     # ONNX export
│       └── finetuning.py # Build a fine-tunable checkpoint
├── configs/
│   ├── train.yaml
│   ├── eval.yaml
│   ├── infer.yaml
│   ├── export.yaml
│   └── finetune.yaml
└── src/
    └── main.rs           # Rust ONNX inference binary
```

## Installation

### Quick install (without cloning)

You can install the package directly from GitHub using either `pip` or `uv`. This gives you immediate access to all CLI tools (`yltrain`, `yleval`, `ylinfer`, `ylft`, `ylexport`) without downloading the full repository.

**With pip** (works in any Python environment, no extra tools needed):

```bash
pip install git+https://github.com/cacybernetic/YOLO8
```

**With uv** (faster, after installing `uv`):

```bash
uv pip install git+https://github.com/cacybernetic/YOLO8
```

After installation, you can run the commands directly (see [Usage](#usage)) — just make sure you have the required configuration YAML files (download them from the [configs/](configs/) folder if needed).

> **Note for contributors**: if you plan to modify the code or contribute, please follow the full local installation instructions below.


### Python — Linux

**1. Install `uv` (fast Python package manager)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository**

```bash
git clone https://github.com/cacybernetic/YOLO8
cd YOLO8
```

**3. Create a virtual environment with Python 3.10**

```bash
uv venv --python 3.10
source .venv/bin/activate
```

**4. Install the package and its dependencies**

```bash
uv pip install -e .
```

This reads all dependencies from `pyproject.toml` and also registers the command-line tools (`yltrain`, `yleval`, `ylinfer`, `ylft`, `ylexport`) so you can call them directly from your terminal.

> **Note — headless server (no display):** if you are running on a server without a graphical interface, install these system libraries first:
> ```bash
> sudo apt-get install libgl1-mesa-glx libglib2.0-0
> ```

### Python — Windows

1. Download and install Python 3.10 from [python.org](https://www.python.org/downloads/).
2. Open a command prompt inside the project folder.
3. Install `uv`:
   ```bash
   pip install uv
   ```
4. Create the virtual environment:
   ```bash
   uv venv --python 3.10
   .venv\Scripts\activate
   ```
5. Install the package and its dependencies:
   ```bash
   uv pip install -e .
   ```

### Rust (optional)

Only needed if you want to run the Rust ONNX inference binary. Skip this section if you only use the Python scripts.

1. Install Rust: [rustup.rs](https://rustup.rs/)
2. Build the release binary:
   ```bash
   cargo build --release
   ```

The binary will be compiled to `target/release/yolov8rust` (Linux/macOS) or `target\release\yolov8rust.exe` (Windows). It automatically downloads ONNX Runtime on the first build.

---

## Dataset format

Your dataset must follow the standard YOLO folder structure:

```
dataset/
├── train/
│   ├── images/   # .jpg, .png, .jpeg, ...
│   └── labels/   # one .txt per image
└── test/
    ├── images/
    └── labels/
```

Each `.txt` label file contains one object per line:

```
<class_id> <cx> <cy> <w> <h>
```

All values are **normalized** between 0 and 1. Example for a single bounding box of class 0:

```
0 0.512 0.348 0.230 0.415
```

---

## Usage

After installation, five commands are available in your terminal:

| Command | Role |
|---|---|
| `yltrain` | Train a model |
| `yleval` | Evaluate a model |
| `ylinfer` | Run inference on an image |
| `ylft` | Build a fine-tunable checkpoint |
| `ylexport` | Export to ONNX |

Each command takes a single `--config` argument pointing to its YAML file.

### 1. Train

Edit `configs/train.yaml` to point to your dataset and set your number of classes, then run:

```bash
yltrain --config configs/train.yaml
```

Checkpoints are saved in the `checkpoints/` folder. The best model is saved as `checkpoints/best.pt`. A training history plot (loss curves) is regenerated after each epoch at `checkpoints/training_history.png`.

### 2. Evaluate

Edit `configs/eval.yaml` (dataset path, weights path, number of classes), then run:

```bash
yleval --config configs/eval.yaml
```

Results are written to the `results/` folder:
- `per_class.csv` — per-class metrics (Precision, Recall, F1, AP@0.5, AP@0.5:0.95)
- `global.csv` — global metrics (mAP, losses, optimal confidence threshold, …)
- `figures/` — PR curves, F1-confidence curve, confusion matrices

### 3. Fine-tune on new classes

**Step 1** — build the fine-tunable checkpoint:

Edit `configs/finetune.yaml` with the source weights, the old number of classes, and the new number of classes, then run:

```bash
ylft --config configs/finetune.yaml
```

This creates a new `.pt` file with the backbone and neck transferred from the source model, and the classification heads re-initialized for the new classes.

**Step 2** — train as usual:

In `configs/train.yaml`, set `pretrained_weights` to the output of step 1 and `num_classes` to your new class count. Optionally set `freeze_feature_layers: true` to only train the detection head (recommended for small datasets):

```bash
yltrain --config configs/train.yaml
```

### 4. Export to ONNX

Edit `configs/export.yaml`, then run:

```bash
ylexport --config configs/export.yaml
```

The exported `.onnx` file is numerically verified against the PyTorch model by default.

### 5. Run inference on an image

Edit `configs/infer.yaml` (weights, number of classes, class names), then run:

```bash
ylinfer --config configs/infer.yaml --image path/to/image.jpg
```

Useful options:
- `--save output.jpg` — save the annotated image to disk
- `--no-show` — disable the display window (useful on a server)
- `--conf 0.4` — override the confidence threshold
- `--iou 0.5` — override the NMS IoU threshold

### 6. Standalone ONNX scripts (`predict.py` and `live.py`)

Two helper scripts at the project root let you run a pre-trained ONNX model
without any dependency on the `yolov8` package — handy for quick demos,
deployment, or running the model on a machine where you only need
`onnxruntime` and a couple of small libraries.

Both scripts share the same `--model`, `--nc`, `--conf`, `--iou`, and
`--names` options, and accept `--log-level` to control verbosity. If the
model has 80 classes, the standard COCO names are used automatically;
otherwise pass a `--names classes.txt` file (one class name per line).

#### `predict.py` — single image inference

Runs on **CPU only** by default (uses GPU if `onnxruntime-gpu` is installed).
Pure `numpy` + `Pillow` for the image pipeline, no OpenCV or PyTorch needed.

```bash
python predict.py \
  --model  weights/best.onnx \
  --nc     80 \
  --image  samples/photo.jpg \
  --output result.jpg
```

Common options:
- `--conf 0.25` — minimum confidence threshold (default 0.25)
- `--iou 0.45` — NMS IoU threshold (default 0.45)
- `--show` — display the annotated image after inference
- `--names classes.txt` — file with one class name per line

#### `live.py` — real-time webcam or video file

Streams predictions in real time on a webcam feed or video file using OpenCV
for capture and display. Shows a live FPS counter and detection count, and
can record the annotated stream to disk.

```bash
# Webcam (index 0)
python live.py --model weights/best.onnx --nc 80 --source 0

# Video file
python live.py --model weights/best.onnx --nc 80 --source path/to/video.mp4

# Headless mode + save the annotated stream
python live.py --model weights/best.onnx --nc 80 \
  --source path/to/video.mp4 --output annotated.mp4 --no-show
```

`--source` accepts either an integer (webcam index) or a path to a video file.
Press `q` or `ESC` in the display window to quit.

Required Python packages for these two scripts: `numpy`, `onnxruntime`,
`Pillow` (for `predict.py`), `opencv-python` (for `live.py`). They are already
included in the project dependencies, so nothing more to install.

### 7. Run inference in Rust

**On a single image:**

```bash
./target/release/yolov8rust \
  --model  weights/best.onnx \
  --image  photo.jpg         \
  --output result.jpg        \
  --nc     80                \
  --conf   0.25              \
  --iou    0.45
```

**Live from webcam** (uses the `rustcv`-based binary, compiled separately — see `Cargo.toml` at the root):

```bash
./target/release/yololivers \
  --model  weights/best.onnx \
  --source 0                 \
  --nc     80
```

`--source 0` opens the first webcam. Press `q` or `ESC` to quit.

---

## Configuration files

All behavior is controlled through YAML files in `configs/`. The most important fields:

| File | Key fields |
|---|---|
| `train.yaml` | `dataset_dir`, `num_classes`, `version` (`n/s/m/l/x`), `epochs`, `batch_size`, `device` |
| `eval.yaml` | `dataset_dir`, `num_classes`, `weights`, `split` (`test` or `train`) |
| `infer.yaml` | `weights`, `num_classes`, `class_names`, `conf_threshold` |
| `export.yaml` | `weights`, `num_classes`, `output_path`, `simplify`, `half` |
| `finetune.yaml` | `pretrained_weights`, `old_num_classes`, `new_num_classes`, `output_weights` |

All unknown keys in a YAML file are silently ignored, so you can add comments freely.

---

## To contribute

Contributions are welcome! Please follow these steps:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add a new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request.

## Licence

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
 
This project was built while learning the inner workings of YOLOv8. A huge
thank-you to **[dtdo90](https://github.com/dtdo90)** for the excellent
educational repository
[**dtdo90/yolov8_detection**](https://github.com/dtdo90/yolov8_detection)
and the accompanying **[YouTube walkthrough](https://www.youtube.com/watch?v=6zQP0L-ph0M)**,
both of which served as the primary reference for understanding the architecture
(backbone, neck, head).
Many implementation choices in this project — the structure of the `Detect`
head, the integration of the
DFL into the box regression, and the way the training targets are assigned —
are directly inspired by their work.

If you find this project useful, please also consider starring the original
repository as a sign of appreciation for the educational content that made it
possible.

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

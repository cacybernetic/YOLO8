# YOLOv8 From Scratch - Concepts for Beginners (English)

Welcome! This document explains, with simple words, how the model and
the training pipeline of this repository work. You do not need to be
an expert. Take your time, read the examples, and everything will
make sense.

> The same document exists in French: `docs/fr_concepts.md`.

---

## 1. What is object detection?

Imagine you show a photo to a friend and you ask: "What do you see,
and where?". Your friend answers: "A chair here, a table there". That
is object detection: find **what** is in the image (the class) and
**where** it is (the bounding box).

A bounding box is just a rectangle. In the YOLO format, we describe it
with 4 numbers between 0 and 1:

```
<class> <center_x> <center_y> <width> <height>
```

Example: `0 0.5 0.5 0.4 0.4` means "an object of class 0, centered in
the image, taking 40% of the width and 40% of the height".

**Question you may ask: why numbers between 0 and 1?**
Because this way the label does not depend on the image size. The same
label works for a 640x640 image and for a 1920x1080 image.

## 2. The three parts of the model

The model is like a small factory with three workshops:

1. **Backbone** (`src/yolov8/modules/backbone.py`)
   It looks at the image and extracts "features": edges, textures,
   shapes. It works at three zoom levels called P3, P4 and P5. P3 sees
   small details, P5 sees the big picture.

2. **Neck** (`src/yolov8/modules/neck.py`)
   It mixes the three zoom levels together, so each level knows what
   the others saw. This is called a PAN-FPN. Think of three friends
   sharing their notes before an exam.

3. **Head** (`src/yolov8/modules/head.py`)
   It makes the final decision: for many small positions on the image
   (called anchors), it predicts a box and a score for each class.

**Question: what does "anchor-free" mean?**
Old YOLO versions used pre-defined box shapes (anchors) and adjusted
them. YOLOv8 does not need them: each position directly predicts the
distance to the four sides of the box. Fewer settings, simpler
training.

### The special trick: DFL

Instead of predicting one number for "distance to the left side", the
model predicts a small probability distribution over 16 possible
values, and we take the average. This is the Distribution Focal Loss
(DFL) idea. It makes the boxes more precise, like answering "the
distance is probably between 3 and 4, closer to 4" instead of just
"4".

## 3. The loss function (how the model learns)

The loss is a number that says "how wrong is the model right now".
Training means making this number go down. Our loss has three parts
(`src/yolov8/lossfn.py`):

- **Box loss (CIoU)**: are the predicted boxes at the right place,
  with the right size?
- **Classification loss (BCE)**: are the classes right?
- **DFL loss**: are the distance distributions sharp and correct?

Before computing the loss, we must decide which predictions should
match which real objects. This is the job of the **TAL assigner**
(Task-Aligned assigner): for each real object, it picks the 10 best
candidate positions, based on both the class score and the box
overlap.

## 4. The dataset pipeline

### Sources: folder or zip

A dataset split can be a plain folder or a `.zip` archive. Both must
contain:

```
images/    the pictures
labels/    one .txt file per picture
data.yaml  the class names (a `names:` list)
```

### The scan and its cache

Before training, we check every label file and every image. Bad
samples (missing label, broken image, wrong numbers) are dropped with
a warning. The result is saved next to the dataset as
`train.cache.json`, so the next run starts instantly.

**Question: what if I change my dataset?**
The cache stores a fingerprint of the dataset (size, date). If the
dataset changed, the scan runs again automatically.

### The validation split

We do not have a separate `val/` folder. Instead, the config value
`val_prob` (default 0.5) takes a fraction of the **test** set for the
per-epoch validation. The final evaluation still runs on the full
test set at the end of the training.

### HDF5: pre-cooked data

Image decoding and augmentation cost CPU time at every step. With
`buildh5ds` you can "pre-cook" the dataset into `train.h5` and
`test.h5` files: samples are already resized (and optionally
augmented). Training with `use_hdf5: true` then just reads arrays.

### Augmentations

During training we randomly change the images so the model never sees
exactly the same picture twice (`src/yolov8/dataset/augment.py`):

- **Mosaic**: glue 4 images into one big canvas, then crop. The model
  sees objects at many scales and positions.
- **MixUp**: blend two images together.
- **HSV jitter**: change colors a little.
- **Flips, rotations, scale**: move things around.
- **Cutout, blur, noise, grayscale**: make life harder on purpose.

Near the end of training (`close_mosaic`), mosaic and mixup are
turned off so the model finishes on realistic images.

## 5. Fault tolerant training

This is the special power of this pipeline. Imagine your electricity
goes down after 3 days of training. With most projects, you lose the
current epoch. Here, you lose at most a few minutes.

### The DataLoaderAdapter

`src/yolov8/dataset/adapter.py` wraps the PyTorch DataLoader. It
remembers three numbers: the seed, the epoch, and how many batches
were already used in this epoch. The shuffle order only depends on
(seed + epoch), so after a crash we can rebuild the exact same order
and **skip** the batches already done. Every sample is still seen
exactly once per epoch.

### Checkpoints

Every `ckpt_step` optimizer steps, a full snapshot is written:
model weights, optimizer, EMA, AMP scaler, the three loader
positions, the partial loss meters, the partial metric accumulator,
and even the random number generator states. File names look like:

```
checkpoint_e0001c0012.pth   -> epoch 1, step 12
```

Old files are deleted automatically (`max_checkpoint`). On restart
with `resume: true`, the trainer reloads the newest checkpoint and
continues exactly where it was, even in the middle of a validation
pass.

## 6. The training loop, step by step

One epoch looks like this (`src/yolov8/training/trainer.py`):

1. For each batch: forward pass, loss, backward pass.
2. Every `grad_accum` batches: one optimizer step (this simulates a
   bigger batch), then the EMA update.
3. Every `ckpt_step` optimizer steps: write a checkpoint.
4. After the last batch: validation on the val split, metrics table,
   history plot, `last.pt`, and `best.pt` when the chosen metric
   improved.

After the last epoch, a final evaluation runs on the **full** test
set and writes `test_results.csv`.

### Learning rate warmup

At the start, the learning rate is tiny and grows for about 3 epochs.
Why? A freshly created model makes random predictions; a big learning
rate at that moment would push the weights in random directions.
Detail: the bias parameters start with a HIGH learning rate (0.1)
that goes down to the normal value; biases are cheap to move and help
the model calibrate quickly.

### EMA (Exponential Moving Average)

We keep a second, smoothed copy of the weights: at each step,
`ema = 0.9999 * ema + 0.0001 * model`. This smoothed copy is less
noisy and usually scores 1-2 mAP better. Validation and `best.pt`
use the EMA weights.

## 7. The metrics

- **Precision**: among my alarms, how many were real? (few false
  alarms = high precision)
- **Recall**: among the real objects, how many did I find? (few
  missed objects = high recall)
- **F1**: the balance of the two.
- **AP@0.5**: area under the precision-recall curve, when a box is
  "correct" if it overlaps the real box by at least 50% (IoU 0.5).
- **mAP@0.5:0.95**: the same, averaged over 10 stricter and stricter
  overlap thresholds. This is the main COCO number.

The evaluation program also draws the precision-recall curves, the
F1-confidence curve (with the best confidence threshold marked) and
the confusion matrix.

## 8. Where do my files go?

```
runs/
  my_run/
    train/            first training run
    train2/           second one, and so on
      weights/best.pt and last.pt
      checkpoints/    fault tolerance snapshots
      plotes/         training_history.png
      logs/           one log file per start
      history.csv     one row per epoch
      config_used.yaml
      test_results.csv
    eval/, eval2/     evaluation runs
      results.csv, per_class.csv, plotes/, renders/, logs/
```

## 9. The five programs

| Command      | What it does                                   |
|--------------|------------------------------------------------|
| `buildh5ds`  | Pre-cook a dataset into HDF5 files             |
| `trainyolo8` | Train (train + val + final test)               |
| `evalyolo8`  | Full evaluation with curves and matrices       |
| `exportw`    | Export the model to ONNX                       |
| `runyolo8`   | Standalone ONNX inference on one image         |
| `ftyolo8`    | Prepare a checkpoint for new classes           |

Each one takes `--config path/to/file.yaml`. Ready-made config files
live in `cpu/configs/` and `gpu/configs/`.

Happy training!

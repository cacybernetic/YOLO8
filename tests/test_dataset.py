"""Dataset tests: sources, scan cache, labels, HDF5, subsets."""

import json

import numpy as np
import torch

from yolov8.dataset import (make_source, YoloDataset, Hdf5Dataset,
                            build_hdf5, parse_label_text,
                            collate_detection_batch, letterbox,
                            adjust_labels_after_letterbox)


def test_parse_label_text_valid():
    reason, cleaned, rows = parse_label_text("0 0.5 0.5 0.2 0.3\n")
    assert reason is None
    assert not cleaned
    assert rows == [[0.0, 0.5, 0.5, 0.2, 0.3]]


def test_parse_label_text_bad_lines_dropped():
    text = "0 0.5 0.5 0.2 0.3\nnot a label line\n"
    reason, cleaned, rows = parse_label_text(text)
    assert reason is None
    assert cleaned
    assert len(rows) == 1


def test_parse_label_text_out_of_range():
    reason, _, rows = parse_label_text("0 1.5 0.5 0.2 0.3\n")
    assert reason == 'bad_values'
    assert rows is None


def test_parse_label_empty_file_is_valid():
    reason, _, rows = parse_label_text("")
    assert reason is None
    assert rows == []


def test_directory_source(tiny_dataset):
    source = make_source(tiny_dataset['train'])
    images = source.list_images()
    assert len(images) == 6
    assert source.read_names() == ['circle', 'square']
    assert source.read_label_text(images[0]).startswith('0 ')


def test_zip_source(tiny_zip_dataset):
    source = make_source(tiny_zip_dataset['train'])
    images = source.list_images()
    assert len(images) == 6
    assert source.read_names() == ['circle', 'square']
    data = source.read_image_bytes(images[0])
    assert len(data) > 0


def test_scan_cache_roundtrip(tiny_zip_dataset):
    ds1 = YoloDataset(tiny_zip_dataset['train'], image_size=96,
                      verbose=False)
    cache_path = tiny_zip_dataset['root'] / 'train.cache.json'
    assert cache_path.exists()
    with open(cache_path) as f:
        cache = json.load(f)
    assert len(cache['samples']) == 6

    # Second load must come from the cache and give the same samples.
    ds2 = YoloDataset(tiny_zip_dataset['train'], image_size=96,
                      verbose=False)
    assert [s['image'] for s in ds1.samples] == \
           [s['image'] for s in ds2.samples]


def test_dataset_getitem_shapes(tiny_dataset):
    ds = YoloDataset(tiny_dataset['train'], image_size=96,
                     verbose=False)
    img, labels, path = ds[0]
    assert img.shape == (3, 96, 96)
    assert float(img.max()) <= 1.0
    assert labels.shape[1] == 5
    assert path.endswith('.jpg')


def test_max_samples_cap(tiny_dataset):
    ds = YoloDataset(tiny_dataset['train'], image_size=96,
                     max_samples=3, verbose=False)
    assert len(ds) == 3


def test_collate_builds_target_dict(tiny_dataset):
    ds = YoloDataset(tiny_dataset['train'], image_size=96,
                     verbose=False)
    batch = [ds[0], ds[1]]
    images, targets, paths = collate_detection_batch(batch)
    assert images.shape == (2, 3, 96, 96)
    assert set(targets) == {'idx', 'cls', 'box'}
    assert targets['box'].shape[1] == 4
    assert len(paths) == 2


def test_letterbox_label_adjustment():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    labels = np.array([[0, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    boxed, ratio, pad = letterbox(img, new_shape=100)
    out = adjust_labels_after_letterbox(labels, ratio, pad,
                                        (50, 100), 100)
    # The box stays centered and the width is unchanged.
    assert abs(out[0, 1] - 0.5) < 1e-6
    assert abs(out[0, 2] - 0.5) < 1e-6
    assert abs(out[0, 3] - 0.5) < 1e-6
    assert abs(out[0, 4] - 0.25) < 1e-6


def test_hdf5_build_and_read(tiny_dataset, tmp_path):
    ds = YoloDataset(tiny_dataset['train'], image_size=96,
                     verbose=False)
    h5_path = tmp_path / 'train.h5'
    build_hdf5(ds, h5_path, augmented_copies=0, verbose=False)

    h5_ds = Hdf5Dataset(h5_path)
    assert len(h5_ds) == len(ds)
    assert h5_ds.names == ['circle', 'square']
    img, labels, _ = h5_ds[0]
    assert img.shape == (3, 96, 96)
    ref_img, ref_labels, _ = ds[0]
    assert torch.allclose(img, ref_img)
    assert torch.allclose(labels, ref_labels)

"""Dataset package: sources, scanning, transforms, augmentations."""

from .names import parse_data_yaml
from .sources import (SampleSource, DirectorySource, ZipSource,
                      make_source, IMAGE_EXTS)
from .validation import parse_label_text, check_image_bytes, \
    decode_image_bytes
from .scanner import scan_source, load_or_scan, load_cache, save_cache
from .transforms import (letterbox, adjust_labels_after_letterbox,
                         image_to_tensor)
from .augment import DEFAULT_AUGMENT_PARAMS, merge_augment_params, \
    Augmenter
from .yolo_dataset import YoloDataset, collate_detection_batch, subsample
from .hdf5_store import Hdf5Builder, Hdf5Dataset, build_hdf5
from .adapter import DataLoaderAdapter

# Backward compatible alias.
YOLODataset = YoloDataset

__all__ = [
    'parse_data_yaml',
    'SampleSource',
    'DirectorySource',
    'ZipSource',
    'make_source',
    'IMAGE_EXTS',
    'parse_label_text',
    'check_image_bytes',
    'decode_image_bytes',
    'scan_source',
    'load_or_scan',
    'load_cache',
    'save_cache',
    'letterbox',
    'adjust_labels_after_letterbox',
    'image_to_tensor',
    'DEFAULT_AUGMENT_PARAMS',
    'merge_augment_params',
    'Augmenter',
    'YoloDataset',
    'YOLODataset',
    'collate_detection_batch',
    'subsample',
    'Hdf5Builder',
    'Hdf5Dataset',
    'build_hdf5',
    'DataLoaderAdapter',
]

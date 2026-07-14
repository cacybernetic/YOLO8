"""Build datasets and the val split from a DatasetConfig block."""

import torch
from torch.utils.data import Subset
from loguru import logger

from .yolo_dataset import YoloDataset
from .hdf5_store import Hdf5Dataset


def build_train_dataset(ds_cfg, seed=0):
    """Train dataset: HDF5 or raw source, with augmentations."""
    if ds_cfg.use_hdf5:
        logger.info(f"Train data: HDF5 file {ds_cfg.train_h5}")
        return Hdf5Dataset(ds_cfg.train_h5,
                           max_samples=ds_cfg.max_train_samples,
                           seed=seed)
    logger.info(f"Train data: {ds_cfg.train_path}")
    return YoloDataset(
        ds_cfg.train_path, image_size=ds_cfg.image_size,
        augment=ds_cfg.augment.enabled,
        augment_params=ds_cfg.augment.params(),
        max_samples=ds_cfg.max_train_samples,
        use_cache=ds_cfg.cache, validate_images=ds_cfg.validate,
        seed=seed)


def build_test_dataset(ds_cfg, seed=0):
    """Test dataset: HDF5 or raw source, never augmented."""
    if ds_cfg.use_hdf5:
        logger.info(f"Test data: HDF5 file {ds_cfg.test_h5}")
        return Hdf5Dataset(ds_cfg.test_h5,
                           max_samples=ds_cfg.max_test_samples,
                           seed=seed)
    logger.info(f"Test data: {ds_cfg.test_path}")
    return YoloDataset(
        ds_cfg.test_path, image_size=ds_cfg.image_size,
        augment=False, max_samples=ds_cfg.max_test_samples,
        use_cache=ds_cfg.cache, validate_images=ds_cfg.validate,
        seed=seed)


def resolve_class_names(ds_cfg, *datasets):
    """Pick the class names: data.yaml first, then the config fallback.

    Also warns when the train and test datasets disagree.
    """
    found = [d.names for d in datasets if getattr(d, 'names', None)]
    if len(found) >= 2 and found[0] != found[1]:
        logger.warning("Train and test data.yaml class lists differ; "
                       "the train list is used.")
    if found:
        return found[0]
    if ds_cfg.class_names:
        return [str(n) for n in ds_cfg.class_names]
    raise ValueError(
        "No class names found. Add a data.yaml file with a `names` "
        "list at the dataset root, or set dataset.class_names in the "
        "config file.")


def split_val_from_test(test_dataset, val_prob, seed=0):
    """Take a deterministic fraction of the test set for validation.

    Returns a Subset of the test dataset with round(val_prob * n)
    samples. The draw only depends on the seed, so every restart uses
    the same validation images.
    """
    n = len(test_dataset)
    val_count = int(round(max(0.0, min(1.0, val_prob)) * n))
    val_count = max(val_count, 1) if n > 0 else 0
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    indices = sorted(perm[:val_count])
    logger.info(f"Validation split: {len(indices)}/{n} test samples "
                f"(val_prob={val_prob})")
    return Subset(test_dataset, indices)

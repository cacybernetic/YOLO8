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

    Diverging train/test class lists are a fatal error: the class ids
    would not mean the same thing in the two splits and every metric
    would be silently wrong.
    """
    found = [d.names for d in datasets if getattr(d, 'names', None)]
    if len(found) >= 2 and found[0] != found[1]:
        raise ValueError(
            f"Train and test data.yaml class lists differ:\n"
            f"  train: {found[0]}\n  test:  {found[1]}\n"
            f"Fix the data.yaml files so both splits share the same "
            f"class list.")
    if found:
        return found[0]
    if ds_cfg.class_names:
        return [str(n) for n in ds_cfg.class_names]
    raise ValueError(
        "No class names found. Add a data.yaml file with a `names` "
        "list at the dataset root, or set dataset.class_names in the "
        "config file.")


def split_val_from_test(test_dataset, val_prob, seed=0):
    """Split the test set into DISJOINT validation and final-test parts.

    The validation part (round(val_prob * n) samples) drives the model
    selection (best.pt, early stopping); the complement is reserved for
    the final evaluation. Keeping the two disjoint avoids selecting the
    model on the very samples that are then used to report its score.
    The draw only depends on the seed, so every restart uses the same
    split.

    Returns:
        (val_subset, final_test_subset). When val_prob leaves no
        held-out sample, the final test falls back to the FULL test
        set with a leakage warning.
    """
    n = len(test_dataset)
    val_count = int(round(max(0.0, min(1.0, val_prob)) * n))
    val_count = max(val_count, 1) if n > 0 else 0
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_indices = sorted(perm[:val_count])
    test_indices = sorted(perm[val_count:])

    val_subset = Subset(test_dataset, val_indices)
    if test_indices:
        final_test = Subset(test_dataset, test_indices)
    else:
        logger.warning(
            f"val_prob={val_prob} leaves no held-out test sample: the "
            f"final evaluation reuses the full test set, so its "
            f"metrics leak into the model selection. Lower val_prob "
            f"to keep a clean final test.")
        final_test = test_dataset

    logger.info(f"Test split: {len(val_indices)} validation + "
                f"{len(test_indices) or n} final-test samples "
                f"(val_prob={val_prob})")
    return val_subset, final_test

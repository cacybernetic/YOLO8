"""Build ready-to-train HDF5 files from a YOLO dataset.

Usage:
    ylbuild --config gpu/configs/hdf5.yaml

The train split can also store extra augmented copies of each sample
(`augmented_copies`), so training with `use_hdf5: true` reads
pre-computed samples and does far less image work per step.
"""

import argparse

from loguru import logger

from yolov8.config import load_hdf5_build_config, config_to_dict
from yolov8.dataset import YoloDataset, build_hdf5
from yolov8.logging import setup_logging, log_dict


def build_split(path, ds_cfg, output_path, augment, augmented_copies):
    """Scan one split and write its HDF5 file."""
    dataset = YoloDataset(
        path, image_size=ds_cfg.image_size, augment=augment,
        augment_params=ds_cfg.augment.params(),
        use_cache=ds_cfg.cache, validate_images=ds_cfg.validate)
    logger.info(f"Building {output_path} from {len(dataset)} samples "
                f"(augmented copies per sample: "
                f"{augmented_copies if augment else 0})")
    build_hdf5(dataset, output_path,
               augmented_copies=augmented_copies if augment else 0)


def main():
    parser = argparse.ArgumentParser(
        description="Build train.h5 and test.h5 from a YOLO dataset.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to hdf5.yaml')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger.info("Starting HDF5 dataset build")
    cfg = load_hdf5_build_config(args.config)
    log_dict(config_to_dict(cfg))

    ds = cfg.dataset
    if ds.train_path:
        build_split(ds.train_path, ds, ds.train_h5,
                    augment=ds.augment.enabled,
                    augmented_copies=cfg.augmented_copies)
    else:
        logger.info("No train_path set: train split skipped")

    if ds.test_path:
        # The test split is never augmented.
        build_split(ds.test_path, ds, ds.test_h5,
                    augment=False, augmented_copies=0)
    else:
        logger.info("No test_path set: test split skipped")

    logger.info("HDF5 build finished.")


if __name__ == '__main__':
    main()

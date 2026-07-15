"""Scan a dataset source, filter bad samples and cache the result.

The scan result is written next to the source as `<name>.cache.json`,
for example `train.zip` -> `train.cache.json`. The next run loads the
cache instead of scanning again, unless the source has changed.
"""

import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .validation import parse_label_text, check_image_bytes

CACHE_VERSION = 2


def scan_source(source, validate_images=True, strict=False, verbose=True):
    """Scan the source and return the list of valid samples.

    Each sample is a dict: {'image': key, 'labels': [[c,cx,cy,w,h], ...]}.

    When the source ships a data.yaml, the class ids are also checked
    against the class count: an out-of-range id would crash the loss
    assigner much later with an obscure error.

    Args:
        source: a SampleSource.
        validate_images: also decode every image (slow but safe).
        strict: raise on the first invalid sample instead of skipping it.
        verbose: log a scan report at the end.
    """
    image_keys = source.list_images()
    if len(image_keys) == 0:
        raise RuntimeError(f"No image found in source: {source}")

    names = source.read_names()
    num_classes = len(names) if names else None

    samples = []
    stats = {'total': len(image_keys), 'missing_label': 0,
             'empty_label': 0, 'bad_format': 0, 'bad_values': 0,
             'bad_class': 0, 'corrupt_image': 0, 'kept': 0,
             'kept_with_cleaning': 0}
    error_examples = []

    iterator = tqdm(image_keys, desc="validating dataset",
                    disable=not verbose, leave=False, dynamic_ncols=True,
                    ascii="░█")
    for key in iterator:
        sample, reason, cleaned = _scan_one(source, key, validate_images,
                                            num_classes)
        if sample is None:
            stats[reason] += 1
            if len(error_examples) < 5:
                error_examples.append(f"{Path(key).name}: {reason}")
            if strict:
                raise ValueError(f"Invalid sample: {key} ({reason})")
            continue
        samples.append(sample)
        stats['kept'] += 1
        if cleaned:
            stats['kept_with_cleaning'] += 1

    if verbose:
        _log_scan_report(stats, error_examples, validate_images)
    return samples, stats


def _scan_one(source, key, validate_images, num_classes=None):
    """Validate one image + label pair.

    Returns (sample, None, cleaned) or (None, reason, False).
    """
    text = source.read_label_text(key)
    if text is None:
        return None, 'missing_label', False
    reason, cleaned, labels = parse_label_text(text, num_classes)
    if reason is not None:
        return None, reason, False
    if validate_images:
        try:
            data = source.read_image_bytes(key)
        except Exception:
            data = None
        if not check_image_bytes(data):
            return None, 'corrupt_image', False
    return {'image': key, 'labels': labels}, None, cleaned


def _log_scan_report(stats, error_examples, validate_images):
    dropped = stats['total'] - stats['kept']
    logger.info(f"dataset scan: {stats['kept']}/{stats['total']} "
                f"valid samples (dropped: {dropped})")
    if dropped > 0:
        logger.warning(f"  - missing label:  {stats['missing_label']}")
        logger.warning(f"  - empty label:    {stats['empty_label']}")
        logger.warning(f"  - bad format:     {stats['bad_format']}")
        logger.warning(f"  - bad values:     {stats['bad_values']}")
        logger.warning(f"  - bad class id:   {stats['bad_class']}")
        if validate_images:
            logger.warning(f"  - corrupt image:  {stats['corrupt_image']}")
        for e in error_examples:
            logger.warning(f"    example: {e}")
    if stats['kept_with_cleaning'] > 0:
        logger.info(f"  - {stats['kept_with_cleaning']} file(s) kept after "
                    f"dropping malformed lines")


def load_cache(source):
    """Load the scan cache for a source. Return samples or None."""
    path = source.cache_path()
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Unreadable scan cache '{path}': {e}")
        return None
    if data.get('version') != CACHE_VERSION:
        logger.info(f"Scan cache version changed, rescanning: {path}")
        return None
    if data.get('fingerprint') != source.fingerprint():
        logger.info(f"Dataset changed since last scan, rescanning: {path}")
        return None
    return data.get('samples')


def save_cache(source, samples, stats):
    """Write the scan cache next to the source."""
    path = source.cache_path()
    data = {
        'version': CACHE_VERSION,
        'fingerprint': source.fingerprint(),
        'stats': stats,
        'samples': samples,
    }
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Scan cache written: {path}")
    except OSError as e:
        logger.warning(f"Cannot write scan cache '{path}': {e}")


def load_or_scan(source, use_cache=True, validate_images=True,
                 strict=False, verbose=True):
    """Return the valid samples, from the cache when possible."""
    if use_cache:
        samples = load_cache(source)
        if samples is not None:
            logger.info(f"Loaded {len(samples)} samples from scan cache: "
                        f"{source.cache_path().name}")
            return samples
    samples, stats = scan_source(
        source, validate_images=validate_images,
        strict=strict, verbose=verbose)
    if use_cache:
        save_cache(source, samples, stats)
    return samples

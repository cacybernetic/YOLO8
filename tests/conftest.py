"""Shared fixtures: tiny synthetic datasets (folder and zip)."""

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

NAMES = ['circle', 'square']

DATA_YAML = "nc: 2\nnames:\n- circle\n- square\n"


def _make_image(path, size=96, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_split(root, n_images=6):
    """Build a small split folder: images/, labels/, data.yaml."""
    root = Path(root)
    (root / 'images').mkdir(parents=True)
    (root / 'labels').mkdir()
    for i in range(n_images):
        _make_image(root / 'images' / f"img_{i:03d}.jpg", seed=i)
        cls = i % 2
        label = f"{cls} 0.5 0.5 0.4 0.4\n"
        (root / 'labels' / f"img_{i:03d}.txt").write_text(label)
    (root / 'data.yaml').write_text(DATA_YAML)
    return root


@pytest.fixture
def tiny_dataset(tmp_path):
    """Folder dataset with train/ and test/ splits."""
    train = _make_split(tmp_path / 'train', n_images=6)
    test = _make_split(tmp_path / 'test', n_images=4)
    return {'root': tmp_path, 'train': train, 'test': test}


@pytest.fixture
def tiny_zip_dataset(tmp_path):
    """Zip dataset: train.zip and test.zip with the same layout."""
    paths = {}
    for split, n in (('train', 6), ('test', 4)):
        folder = _make_split(tmp_path / f"src_{split}", n_images=n)
        zip_path = tmp_path / f"{split}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for p in sorted(folder.rglob('*')):
                if p.is_file():
                    zf.write(p, p.relative_to(folder).as_posix())
        paths[split] = zip_path
    paths['root'] = tmp_path
    return paths

"""Sample sources: read a YOLO dataset from a folder or a zip archive.

A source gives a uniform view of a dataset split:
  - a list of image keys
  - the bytes of an image
  - the text of the matching label file
  - the class names from data.yaml
  - a fingerprint used to know when the scan cache is stale

Expected layout inside the folder or the zip:
    images/xxx.jpg
    labels/xxx.txt
    data.yaml
An extra top folder (for example `train/images/...`) is detected too.
"""

import os
import zipfile
from pathlib import Path

from .names import parse_data_yaml

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


class SampleSource:
    """Base class for dataset sources."""

    def list_images(self):
        raise NotImplementedError

    def read_image_bytes(self, image_key):
        raise NotImplementedError

    def read_label_text(self, image_key):
        """Return the label text for an image, or None when missing."""
        raise NotImplementedError

    def read_names(self):
        """Return the class names from data.yaml, or None when missing."""
        raise NotImplementedError

    def fingerprint(self):
        raise NotImplementedError

    def cache_path(self):
        raise NotImplementedError

    @staticmethod
    def label_key_for(image_key):
        """Map images/xxx.jpg to labels/xxx.txt (any prefix kept)."""
        head, _, tail = image_key.rpartition('images/')
        stem = Path(tail).stem
        return f"{head}labels/{stem}.txt"


class DirectorySource(SampleSource):
    """Dataset split stored as a plain folder."""

    def __init__(self, root):
        self.root = Path(root)
        self.images_dir = self._find_images_dir()
        self.prefix = str(
            self.images_dir.parent.relative_to(self.root))
        if self.prefix == '.':
            self.prefix = ''

    def _find_images_dir(self):
        """Locate the images/ folder at the root or one level below."""
        direct = self.root / 'images'
        if direct.is_dir():
            return direct
        candidates = sorted(self.root.glob('*/images'))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            f"No images/ folder found under: {self.root}")

    def list_images(self):
        base = self.images_dir.parent
        keys = []
        for p in sorted(self.images_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                rel = p.relative_to(base)
                head = f"{self.prefix}/" if self.prefix else ''
                keys.append(f"{head}{rel.as_posix()}")
        return keys

    def _abs(self, key):
        return self.root / key

    def read_image_bytes(self, image_key):
        return self._abs(image_key).read_bytes()

    def read_label_text(self, image_key):
        path = self._abs(self.label_key_for(image_key))
        if not path.exists():
            return None
        return path.read_text()

    def read_names(self):
        for base in (self.images_dir.parent, self.root):
            path = base / 'data.yaml'
            if path.exists():
                return parse_data_yaml(path.read_text())
        return None

    def fingerprint(self):
        files = list(self.images_dir.iterdir())
        return {
            'kind': 'directory',
            'path': str(self.root.resolve()),
            'n_images': len(files),
            'mtime': self.images_dir.stat().st_mtime,
        }

    def cache_path(self):
        root = self.root.resolve()
        return root.parent / f"{root.name}.cache.json"


class ZipSource(SampleSource):
    """Dataset split stored as a zip archive.

    The zip handle is opened lazily and re-opened after a fork, so the
    source is safe to use inside DataLoader worker processes.
    """

    def __init__(self, zip_path):
        self.path = Path(zip_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Zip archive not found: {self.path}")
        self._handle = None
        self._pid = None
        self.prefix = self._detect_prefix()

    def _zf(self):
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = zipfile.ZipFile(self.path, 'r')
            self._pid = pid
        return self._handle

    def _detect_prefix(self):
        """Find the path prefix that contains the images/ folder."""
        for name in self._zf().namelist():
            head, sep, _ = name.partition('images/')
            if sep and ('/' not in head or head.count('/') == 1):
                return head
        raise FileNotFoundError(
            f"No images/ folder found inside: {self.path}")

    def list_images(self):
        keys = []
        img_prefix = f"{self.prefix}images/"
        for name in sorted(self._zf().namelist()):
            if not name.startswith(img_prefix) or name.endswith('/'):
                continue
            if Path(name).suffix.lower() in IMAGE_EXTS:
                keys.append(name[len(self.prefix):])
        return keys

    def read_image_bytes(self, image_key):
        return self._zf().read(self.prefix + image_key)

    def read_label_text(self, image_key):
        key = self.prefix + self.label_key_for(image_key)
        try:
            return self._zf().read(key).decode('utf-8')
        except KeyError:
            return None

    def read_names(self):
        for key in (f"{self.prefix}data.yaml", 'data.yaml'):
            try:
                text = self._zf().read(key).decode('utf-8')
            except KeyError:
                continue
            return parse_data_yaml(text)
        return None

    def fingerprint(self):
        stat = self.path.stat()
        return {
            'kind': 'zip',
            'path': str(self.path.resolve()),
            'size': stat.st_size,
            'mtime': stat.st_mtime,
        }

    def cache_path(self):
        path = self.path.resolve()
        return path.parent / f"{path.stem}.cache.json"

    def __getstate__(self):
        # Zip handles cannot be pickled; workers re-open the file.
        state = dict(self.__dict__)
        state['_handle'] = None
        state['_pid'] = None
        return state


def make_source(path):
    """Build the right source for a dataset path (folder or .zip)."""
    path = Path(path)
    if path.is_dir():
        return DirectorySource(path)
    if path.suffix.lower() == '.zip':
        return ZipSource(path)
    raise ValueError(
        f"Unsupported dataset path (need a folder or a .zip): {path}")

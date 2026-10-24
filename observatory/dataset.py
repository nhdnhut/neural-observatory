"""MNIST download and IDX parsing using the Python standard library only."""
from pathlib import Path
import gzip
import hashlib
import random
import struct
import urllib.request

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def download(root="data/raw"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        valid = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if not valid:
            print("Downloading", name, flush=True)
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Download checksum mismatch: " + name)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Dataset checksum mismatch: " + name)
    return root


def read_images(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n, h, w = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or h != 28 or w != 28 or len(raw) != 16 + n * h * w:
        raise ValueError("Invalid MNIST image file")
    return raw[16:], n


def read_labels(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049 or len(raw) != 8 + n or max(raw[8:]) > 9:
        raise ValueError("Invalid MNIST label file")
    return list(raw[8:])


def split_indices(labels, seed=2026, validation_per_class=500):
    """Fixed stratified split of official training data; test is never used."""
    rng = random.Random(seed)
    train, val = [], []
    for c in range(10):
        indices = [i for i, y in enumerate(labels) if y == c]
        rng.shuffle(indices)
        val.extend(indices[:validation_per_class])
        train.extend(indices[validation_per_class:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_numpy(root="data/raw"):
    import numpy as np
    root = download(root)
    raw, n = read_images(root / "train-images-idx3-ubyte.gz")
    labels = read_labels(root / "train-labels-idx1-ubyte.gz")
    x = np.frombuffer(raw, dtype=np.uint8).reshape(n, 784).astype(np.float32) / 255
    y = np.asarray(labels, dtype=np.int64)
    train, val = split_indices(labels)
    raw_test, n_test = read_images(root / "t10k-images-idx3-ubyte.gz")
    xt = np.frombuffer(raw_test, dtype=np.uint8).reshape(n_test, 784).astype(np.float32) / 255
    yt = np.asarray(read_labels(root / "t10k-labels-idx1-ubyte.gz"), dtype=np.int64)
    return x[train], y[train], x[val], y[val], xt, yt, train, val


if __name__ == "__main__":
    download()

"""Synthetic production-line images: clean surface versus scratch or spot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_image(defect: bool, rng: np.random.Generator, size: int = 64) -> np.ndarray:
    image = rng.normal(148, 16, (size, size))
    specks = rng.integers(4, 12)
    ys = rng.integers(0, size, specks)
    xs = rng.integers(0, size, specks)
    image[ys, xs] -= rng.uniform(15, 35, size=specks)
    if defect:
        if rng.random() < 0.55:
            y = int(rng.integers(4, size - 4))
            x0 = int(rng.integers(0, 18))
            length = int(rng.integers(28, 48))
            depth = float(rng.uniform(38, 62))
            for step in range(length):
                x = x0 + step
                if x >= size:
                    break
                image[y, x] -= depth
                if y + 1 < size:
                    image[y + 1, x] -= depth * 0.45
        else:
            cy = int(rng.integers(8, size - 8))
            cx = int(rng.integers(8, size - 8))
            radius = int(rng.integers(3, 6))
            yy, xx = np.ogrid[:size, :size]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            image[mask] -= float(rng.uniform(45, 75))
    return np.clip(image, 0, 255).astype(np.uint8)


def features(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    median = np.median(values)
    dark = (median - values) > 28
    dx = np.abs(np.diff(values, axis=1))
    dy = np.abs(np.diff(values, axis=0))
    return np.array(
        [
            values.mean(),
            values.std(),
            values.min(),
            dark.mean(),
            dx.mean(),
            dy.mean(),
            dx.max(),
            np.percentile(median - values, 95),
        ],
        dtype=np.float32,
    )


def build_split(n_train: int = 600, n_test: int = 200, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    total = n_train + n_test
    for index in range(total):
        defect = index % 2 == 1
        images.append(make_image(defect, rng))
        labels.append(int(defect))
    order = rng.permutation(total)
    images = [images[i] for i in order]
    labels = np.asarray(labels, dtype=np.int64)[order]
    return {
        "x_train": images[:n_train],
        "y_train": labels[:n_train],
        "x_test": images[n_train:],
        "y_test": labels[n_train:],
    }


def save_samples(folder: Path, images: list[np.ndarray], labels: np.ndarray, limit: int = 8) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, (image, label) in enumerate(zip(images[:limit], labels[:limit])):
        name = "defect" if label else "ok"
        Image.fromarray(image, mode="L").save(folder / f"{index:02d}_{name}.png")

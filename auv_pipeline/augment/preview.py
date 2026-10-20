"""Render an augmentation preview grid.

Row 1 ("anatomy"): the original frame, then each transform of the profile forced on.
Rows 2+  ("samples"): the full profile applied at its real probabilities, with
Ultralytics-style HSV jitter — i.e. what the model actually sees during training.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from . import _parse_spec, build_transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(sources: list[str], limit: int, rng: random.Random) -> list[Path]:
    paths: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        elif p.exists():
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No images found in: {', '.join(sources)}")
    return rng.sample(paths, min(limit, len(paths)))


def hsv_jitter(img: np.ndarray, h: float, s: float, v: float, rng: np.random.Generator) -> np.ndarray:
    """Same maths as Ultralytics' RandomHSV (multiplicative gains, hue wraps)."""
    if not (h or s or v):
        return img
    r = rng.uniform(-1, 1, 3) * [h, s, v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    scale = size / max(img.shape[:2])
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def _pad(img: np.ndarray, size: int) -> np.ndarray:
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
    return canvas


def make_grid(tiles: list[tuple[np.ndarray, str]], cols: int, title: str) -> np.ndarray:
    size = tiles[0][0].shape[0]
    label_h, title_h, pad = 26, 40, 6
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((title_h + rows * (size + label_h + pad), cols * (size + pad) + pad, 3), 18, np.uint8)
    title = title.replace("—", "-").replace("–", "-").encode("ascii", "ignore").decode()  # putText is ASCII-only
    cv2.putText(grid, title, (pad + 4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
    for i, (img, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        y = title_h + r * (size + label_h + pad)
        x = pad + c * (size + pad)
        cv2.putText(grid, label, (x + 2, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 210, 255), 1, cv2.LINE_AA)
        grid[y + label_h : y + label_h + size, x : x + size] = img
    return grid


def render(
    profile_name: str,
    profile: dict,
    sources: list[str],
    samples: int = 8,
    size: int = 320,
    cols: int = 5,
    seed: int = 0,
) -> np.ndarray:
    import albumentations as A

    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    images = [cv2.imread(str(p)) for p in collect_images(sources, max(1, samples), rng)]
    images = [_resize(im, size) for im in images if im is not None]
    transforms = build_transforms(profile) or []
    names = [_parse_spec(s)[0] for s in profile.get("transforms") or []]
    hsv = profile.get("hsv") or {}

    base = images[0]
    tiles = [(base, "original")]
    for name, t in zip(names, transforms):
        t.set_random_seed(seed)
        tiles.append((t(image=base, force_apply=True)["image"], name))
    if hsv:
        tiles.append((hsv_jitter(base, hsv.get("h", 0), hsv.get("s", 0), hsv.get("v", 0), nprng), "HSV jitter"))
    while len(tiles) % cols:  # finish the anatomy row
        tiles.append((np.full((size, size, 3), 18, np.uint8), ""))

    compose = A.Compose(transforms, seed=seed)
    for i in range(samples):
        img = images[i % len(images)]
        img = hsv_jitter(img, hsv.get("h", 0), hsv.get("s", 0), hsv.get("v", 0), nprng)
        tiles.append((compose(image=img)["image"], f"sample {i + 1}"))

    tiles = [(img if img.shape[:2] == (size, size) else _pad(img, size), label) for img, label in tiles]
    return make_grid(tiles, cols, f"augment profile: {profile_name}  -  {profile.get('description', '')}")

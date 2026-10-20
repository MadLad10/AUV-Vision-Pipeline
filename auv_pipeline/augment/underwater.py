"""Physically-motivated underwater augmentations.

Underwater images follow (approximately) the image-formation model

    I_c = J_c · exp(-β_c · d)  +  B_c · (1 − exp(-β_c · d))
          └── direct signal ──┘    └──── backscatter ─────┘

where J is the scene radiance, d the camera–object range, β_c the per-channel
attenuation coefficient (red is absorbed fastest) and B_c the veiling light.
Each transform below simulates one term of that model, plus the lighting and
particle effects an AUV camera sees in practice. Stock "photo" augmentations
(HSV jitter, brightness/contrast) cover none of these.

All transforms are pixel-level only (bounding boxes are unchanged), operate on
uint8 images in **BGR** order (what Ultralytics feeds Albumentations), and draw
their randomness from Albumentations' seeded generators, so results are
reproducible under ``A.Compose(..., seed=...)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np

# Approximate attenuation coefficients β (1/m) for (R, G, B), Jerlov-style water types.
WATER_TYPES: dict[str, tuple[float, float, float]] = {
    "clear": (0.35, 0.06, 0.04),  # open ocean / clean pool -> blue cast
    "coastal": (0.50, 0.12, 0.20),  # greenish
    "turbid": (0.70, 0.25, 0.45),  # harbour / lake -> green-yellow, murky
}


def _range(value: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    lo, hi = value
    return float(lo), float(hi)


def smooth_noise(h: int, w: int, rng: np.random.Generator, cells: int) -> np.ndarray:
    """Low-frequency noise field in [0, 1] — a cells×cells random grid upsampled bicubically."""
    cells = max(2, int(cells))
    grid = rng.random((cells, cells)).astype(np.float32)
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / max(hi - lo, 1e-6)


def _to_bgr(rgb: Sequence[float]) -> np.ndarray:
    return np.asarray(rgb[::-1], dtype=np.float32)


class WaterAttenuation(A.ImageOnlyTransform):
    """Wavelength-dependent absorption: red fades first, then green (Beer–Lambert).

    The image is attenuated per channel by ``exp(-β·d)`` and then re-exposed so its
    mean brightness is preserved, like a camera's auto-exposure would.
    """

    def __init__(
        self,
        depth: float | Sequence[float] = (0.5, 5.0),
        water: str | Sequence[str] = ("clear", "coastal", "turbid"),
        max_gain: float = 1.6,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.depth = _range(depth)
        self.water = [water] if isinstance(water, str) else list(water)
        unknown = set(self.water) - set(WATER_TYPES)
        if unknown:
            raise ValueError(f"Unknown water type(s) {sorted(unknown)}; choose from {sorted(WATER_TYPES)}")
        self.max_gain = max_gain

    def get_params(self) -> dict:
        return {
            "depth": self.py_random.uniform(*self.depth),
            "water": self.py_random.choice(self.water),
        }

    def apply(self, img: np.ndarray, depth: float = 0.0, water: str = "clear", **params) -> np.ndarray:
        f = img.astype(np.float32)
        out = f * np.exp(-_to_bgr(WATER_TYPES[water]) * depth)
        out *= min(f.mean() / max(float(out.mean()), 1e-3), self.max_gain)
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("depth", "water", "max_gain")


class Backscatter(A.ImageOnlyTransform):
    """Veiling light from forward scattering: ``I = J·t + B·(1 − t)``.

    The transmission ``t`` varies smoothly across the frame (``unevenness``), so haze
    is patchy the way real turbidity is. ``veil_color`` is RGB; ``None`` samples a
    random blue-green water colour each call.
    """

    def __init__(
        self,
        strength: float | Sequence[float] = (0.1, 0.5),
        veil_color: Sequence[int] | None = None,
        unevenness: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.strength = _range(strength)
        self.veil_color = tuple(veil_color) if veil_color is not None else None
        self.unevenness = unevenness

    def get_params(self) -> dict:
        r = self.py_random
        color = self.veil_color or (r.uniform(0, 60), r.uniform(80, 180), r.uniform(100, 210))
        return {
            "strength": r.uniform(*self.strength),
            "color": color,
            "seed": r.randrange(2**31),
        }

    def apply(self, img: np.ndarray, strength: float = 0.0, color=(0, 0, 0), seed: int = 0, **params):
        h, w = img.shape[:2]
        field = smooth_noise(h, w, np.random.default_rng(seed), cells=4)
        haze = strength * (1.0 - self.unevenness + self.unevenness * field)[..., None]
        out = img.astype(np.float32) * (1.0 - haze) + _to_bgr(color) * haze
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("strength", "veil_color", "unevenness")


class Caustics(A.ImageOnlyTransform):
    """Rippling sunlight networks projected onto pool floors and props in shallow water.

    Bright lines appear where two smooth noise fields cross, which gives the familiar
    cellular caustic pattern. ``cells`` controls the pattern scale (more = finer).
    """

    def __init__(
        self,
        strength: float | Sequence[float] = (0.05, 0.25),
        cells: int | Sequence[int] = (6, 16),
        sharpness: float = 12.0,
        p: float = 0.3,
    ):
        super().__init__(p=p)
        self.strength = _range(strength)
        self.cells = tuple(int(c) for c in _range(cells))
        self.sharpness = sharpness

    def get_params(self) -> dict:
        r = self.py_random
        return {
            "strength": r.uniform(*self.strength),
            "cells": r.randint(*self.cells),
            "seed": r.randrange(2**31),
        }

    def apply(self, img: np.ndarray, strength: float = 0.0, cells: int = 8, seed: int = 0, **params):
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed)
        ridge = 1.0 - np.abs(smooth_noise(h, w, rng, cells) - smooth_noise(h, w, rng, cells))
        pattern = (ridge**self.sharpness)[..., None]
        out = img.astype(np.float32) + 255.0 * strength * pattern
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("strength", "cells", "sharpness")


class LightFalloff(A.ImageOnlyTransform):
    """Uneven artificial lighting: a bright cone from the vehicle's lights, dark corners."""

    def __init__(
        self,
        strength: float | Sequence[float] = (0.2, 0.6),
        radius: float | Sequence[float] = (0.5, 1.2),
        p: float = 0.3,
    ):
        super().__init__(p=p)
        self.strength = _range(strength)
        self.radius = _range(radius)

    def get_params(self) -> dict:
        r = self.py_random
        return {
            "strength": r.uniform(*self.strength),
            "radius": r.uniform(*self.radius),
            "center": (r.uniform(0.25, 0.75), r.uniform(0.25, 0.75)),
        }

    def apply(self, img: np.ndarray, strength: float = 0.0, radius: float = 1.0, center=(0.5, 0.5), **params):
        h, w = img.shape[:2]
        s = float(max(h, w))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.hypot(xx / s - center[0] * w / s, yy / s - center[1] * h / s) / radius
        mask = 1.0 - strength * np.clip(dist, 0.0, 1.0) ** 2
        return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("strength", "radius")


class MarineSnow(A.ImageOnlyTransform):
    """Suspended particles lit by the vehicle's lights: soft, out-of-focus bright specks."""

    def __init__(
        self,
        density: float | Sequence[float] = (0.0002, 0.0015),
        size: int | Sequence[int] = (1, 4),
        brightness: float | Sequence[float] = (0.4, 0.9),
        p: float = 0.3,
    ):
        super().__init__(p=p)
        self.density = _range(density)
        self.size = tuple(int(v) for v in _range(size))
        self.brightness = _range(brightness)

    def get_params(self) -> dict:
        r = self.py_random
        return {"density": r.uniform(*self.density), "seed": r.randrange(2**31)}

    def apply(self, img: np.ndarray, density: float = 0.0, seed: int = 0, **params):
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed)
        n = int(density * h * w)
        layer = np.zeros((h, w), dtype=np.float32)
        xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
        radii = rng.integers(self.size[0], self.size[1] + 1, n)
        levels = rng.uniform(*self.brightness, n)
        for x, y, rad, lvl in zip(xs, ys, radii, levels):
            cv2.circle(layer, (int(x), int(y)), int(rad), float(lvl), -1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=1.2)[..., None]
        f = img.astype(np.float32) / 255.0
        out = 1.0 - (1.0 - f) * (1.0 - layer)  # screen blend
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("density", "size", "brightness")


CUSTOM_TRANSFORMS: dict[str, type[A.ImageOnlyTransform]] = {
    cls.__name__: cls for cls in (WaterAttenuation, Backscatter, Caustics, LightFalloff, MarineSnow)
}

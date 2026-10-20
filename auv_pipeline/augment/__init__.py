"""Stage 1 — underwater augmentation profiles: loading, building, and previewing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import CONFIG_DIR, load_yaml

PROFILES_FILE = CONFIG_DIR / "augment" / "underwater.yaml"


def load_profiles(path: Path = PROFILES_FILE) -> dict[str, dict]:
    return load_yaml(path).get("profiles", {})


def get_profile(name: str, path: Path = PROFILES_FILE) -> dict:
    profiles = load_profiles(path)
    if name not in profiles:
        raise KeyError(f"Unknown augment profile '{name}'. Available: {', '.join(profiles)}")
    return profiles[name] or {}


def hsv_overrides(profile: dict) -> dict[str, float]:
    """Profile HSV gains as Ultralytics train kwargs."""
    hsv = profile.get("hsv") or {}
    return {f"hsv_{k}": float(v) for k, v in hsv.items()}


def _parse_spec(spec: Any) -> tuple[str, dict]:
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict) and len(spec) == 1:
        name, kwargs = next(iter(spec.items()))
        return name, dict(kwargs or {})
    raise ValueError(f"Bad transform spec {spec!r}: use 'Name' or {{Name: {{kwargs}}}}")


def describe(profile: dict) -> list[str]:
    """Human-readable transform list, without importing albumentations."""
    out = []
    for spec in profile.get("transforms") or []:
        name, kwargs = _parse_spec(spec)
        p = kwargs.get("p")
        out.append(f"{name}" + (f" (p={p})" if p is not None else ""))
    return out


def build_transforms(profile: dict) -> list | None:
    """Instantiate a profile's transforms. Returns None for a profile with no transforms."""
    specs = profile.get("transforms") or []
    if not specs:
        return None
    import albumentations as A

    from .underwater import CUSTOM_TRANSFORMS

    transforms = []
    for spec in specs:
        name, kwargs = _parse_spec(spec)
        cls = CUSTOM_TRANSFORMS.get(name) or getattr(A, name, None)
        if cls is None:
            raise ValueError(f"Unknown transform '{name}' (not in underwater.py or albumentations)")
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()}
        transforms.append(cls(**kwargs))
    return transforms

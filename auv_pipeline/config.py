"""Preset loading and override merging.

A training preset (configs/train/*.yaml) looks like::

    extends: _base.yaml          # optional, resolved relative to this file
    description: ...
    model: yolo26n.pt
    name: yolo26n_custom
    augment: strong              # profile from configs/augment/underwater.yaml
    train: {...}                 # passed straight to Ultralytics model.train()
    export: {formats: [onnx, engine], half: true}

Merge order (later wins):
    _base.yaml < preset < augment profile HSV gains < --hyp file < CLI flags < --set
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
TRAIN_PRESET_DIR = CONFIG_DIR / "train"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Lists are replaced, not appended."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, (*_seen, path)))}")
    data = load_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return deep_merge(load_with_extends(path.parent / parent, (*_seen, path)), data)


def list_presets(directory: Path = TRAIN_PRESET_DIR) -> dict[str, str]:
    """Return ``{preset_name: description}`` for every non-underscore YAML in ``directory``."""
    presets = {}
    for p in sorted(directory.glob("*.yaml")):
        if not p.name.startswith("_"):
            presets[p.stem] = load_yaml(p).get("description", "")
    return presets


def resolve_preset(name_or_path: str, directory: Path = TRAIN_PRESET_DIR) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        return candidate
    named = directory / f"{name_or_path}.yaml"
    if named.exists():
        return named
    available = ", ".join(list_presets(directory)) or "none"
    raise FileNotFoundError(f"Unknown preset '{name_or_path}'. Available: {available}")


def parse_set(items: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=value`` pairs; values are YAML-typed (``0.5`` -> float, ``false`` -> bool)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw) if raw.strip() else None
    return out

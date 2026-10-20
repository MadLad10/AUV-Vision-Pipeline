"""Stage 2 — config-driven YOLO training with underwater augmentation.

The augment profile's transforms are passed to Ultralytics through its native
``augmentations=`` train argument (no monkey-patching), and its HSV gains become
``hsv_h / hsv_s / hsv_v``.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

from . import ui
from .augment import build_transforms, describe, get_profile, hsv_overrides
from .config import load_with_extends, load_yaml, resolve_preset


def build_config(
    preset: str,
    *,
    model: str | None = None,
    name: str | None = None,
    augment: str | None = None,
    hyp: str | None = None,
    overrides: dict | None = None,
    sets: dict | None = None,
) -> dict:
    """Resolve a preset into the final config. See config.py for the merge order."""
    cfg = load_with_extends(resolve_preset(preset))
    cfg["preset"] = preset
    cfg["augment"] = augment or cfg.get("augment", "none")
    profile = get_profile(cfg["augment"])

    train = {**hsv_overrides(profile), **(cfg.get("train") or {})}
    if hyp:
        # Tuned values (tune_yolo.py) must win over the preset; previously they were
        # silently overridden by the hard-coded kwargs passed to model.train().
        train.update(load_yaml(Path(hyp)))
    train.update({k: v for k, v in (overrides or {}).items() if v is not None})
    train.update(sets or {})

    cfg["train"] = train
    cfg["model"] = model or cfg["model"]
    cfg["name"] = name or cfg.get("name") or f"{Path(cfg['model']).stem}_custom"
    cfg.setdefault("export", {})
    cfg["_profile"] = profile
    return cfg


def print_config(cfg: dict, data: str, project: str) -> None:
    t = cfg["train"]
    hsv = "/".join(str(t.get(k, "-")) for k in ("hsv_h", "hsv_s", "hsv_v"))
    rows = [
        ("Preset", cfg["preset"]),
        ("Model", cfg["model"]),
        ("Data", data),
        ("Output", f"{project}/{cfg['name']}"),
        ("Epochs / patience", f"{t.get('epochs')} / {t.get('patience')}"),
        ("Image size / batch", f"{t.get('imgsz')} / {t.get('batch')}"),
        ("Device", t.get("device", "auto")),
        ("Optimizer / lr0 / lrf", f"{t.get('optimizer')} / {t.get('lr0')} / {t.get('lrf')}"),
        ("Augment profile", cfg["augment"]),
        ("HSV gains (h/s/v)", hsv),
        ("Export", ", ".join(cfg["export"].get("formats", [])) or "none"),
    ]
    ui.kv("Training config", rows)
    transforms = describe(cfg["_profile"])
    if transforms:
        ui.table("Underwater augmentations", ("#", "Transform"), list(enumerate(transforms, 1)))
    other = {k: v for k, v in t.items() if k not in _SHOWN}
    if other:
        ui.kv("Other train args", sorted(other.items()))


_SHOWN = {"epochs", "patience", "imgsz", "batch", "device", "optimizer", "lr0", "lrf", "hsv_h", "hsv_s", "hsv_v"}


def _check_native_augmentations() -> None:
    from ultralytics.data.augment import Albumentations

    if "transforms" not in inspect.signature(Albumentations.__init__).parameters:
        raise RuntimeError(
            "This Ultralytics version cannot take custom Albumentations transforms. "
            "Upgrade with: pip install -U 'ultralytics>=8.4'"
        )


def train(cfg: dict, data: str, project: str = "runs/train", resume: str | None = None) -> Path:
    """Train and return the run's save_dir."""
    from ultralytics import YOLO

    if resume:
        model = YOLO(resume)
        model.train(resume=True)
        return Path(model.trainer.save_dir)

    kwargs = dict(cfg["train"])
    kwargs.update(data=str(data), project=str(Path(project).resolve()), name=cfg["name"], exist_ok=False)
    transforms = build_transforms(cfg["_profile"])
    if transforms is not None:
        _check_native_augmentations()
        kwargs["augmentations"] = transforms

    model = YOLO(cfg["model"])
    start = time.time()
    try:
        model.train(**kwargs)
    except KeyboardInterrupt:
        save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
        if save_dir:
            ui.warn(f"Interrupted. Resume with:\n    python -m auv_pipeline train --resume {save_dir}/weights/last.pt")
        raise
    # Use the trainer's real save_dir: with exist_ok=False Ultralytics may have
    # renamed the run (yolo26n_custom2, ...), so a reconstructed path can be wrong.
    save_dir = Path(model.trainer.save_dir)
    ui.success(f"Training finished in {(time.time() - start) / 60:.1f} min -> {save_dir}")
    return save_dir

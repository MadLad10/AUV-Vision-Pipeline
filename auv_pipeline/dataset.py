"""Dataset sanity check, run before every training so a broken export fails in seconds, not hours.

Checks a YOLO ``data.yaml`` the same way Ultralytics resolves it: every split's images,
their label files (``/images/`` -> ``/labels/``), class ids against ``names``, and the
per-class instance count.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import ui
from .config import load_yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass
class SplitStats:
    name: str
    images: int = 0
    missing_labels: int = 0
    background: int = 0  # empty label file: valid negative sample
    bad_lines: int = 0
    instances: Counter = field(default_factory=Counter)


@dataclass
class DatasetReport:
    yaml_path: Path
    root: Path
    names: list[str]
    splits: dict[str, SplitStats] = field(default_factory=dict)
    sources: dict[str, list[Path]] = field(default_factory=dict)  # resolved image dirs / list files
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def has_test(self) -> bool:
        return "test" in self.splits and self.splits["test"].images > 0


def _names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    nc = data.get("nc")
    return [str(i) for i in range(int(nc))] if nc else []


def _root(yaml_path: Path, data: dict) -> Path:
    raw = data.get("path")
    if not raw:
        return yaml_path.parent
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (yaml_path.parent, Path.cwd()):
        if (base / p).exists():
            return (base / p).resolve()
    return (yaml_path.parent / p).resolve()


def _images_for(entry: str, root: Path) -> list[Path]:
    p = Path(entry)
    p = p if p.is_absolute() else root / p
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if p.is_file() and p.suffix == ".txt":  # list-of-images file
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return [Path(ln) if Path(ln).is_absolute() else (root / ln) for ln in lines]
    return []


def label_path(image: Path) -> Path:
    """Ultralytics' img2label_paths: swap the last /images/ for /labels/ and the suffix for .txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    s = str(image)
    return Path(sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def _scan_label(path: Path, nc: int, stats: SplitStats) -> None:
    lines = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        stats.background += 1
        return
    for parts in lines:
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except (ValueError, IndexError):
            stats.bad_lines += 1
            continue
        if len(coords) < 4 or not 0 <= cls < nc or any(v < -0.01 or v > 1.01 for v in coords):
            stats.bad_lines += 1
            continue
        stats.instances[cls] += 1


def inspect_dataset(data_yaml: str | Path) -> DatasetReport:
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        report = DatasetReport(yaml_path, yaml_path.parent, [])
        report.errors.append(f"data.yaml not found: {yaml_path}")
        return report

    data = load_yaml(yaml_path)
    report = DatasetReport(yaml_path, _root(yaml_path, data), _names(data))
    if not report.names:
        report.errors.append("data.yaml has no 'names' (or 'nc')")
    nc = len(report.names)

    for split in SPLITS:
        entries = data.get(split)
        if not entries:
            if split != "test":
                report.errors.append(f"data.yaml has no '{split}' entry")
            continue
        stats = SplitStats(split)
        for entry in entries if isinstance(entries, list) else [entries]:
            src = Path(str(entry))
            report.sources.setdefault(split, []).append(src if src.is_absolute() else report.root / src)
            images = _images_for(str(entry), report.root)
            if not images:
                report.errors.append(f"{split}: no images found at '{entry}' (root {report.root})")
            for img in images:
                stats.images += 1
                lbl = label_path(img)
                if lbl.exists():
                    _scan_label(lbl, nc, stats)
                else:
                    stats.missing_labels += 1
        report.splits[split] = stats

        if stats.bad_lines:
            report.errors.append(
                f"{split}: {stats.bad_lines} malformed label line(s) "
                f"(class id outside 0..{nc - 1}, <4 coords, or coords outside [0, 1])"
            )
        if stats.missing_labels:
            report.warnings.append(
                f"{split}: {stats.missing_labels} image(s) have no label file — treated as background"
            )

    train = report.splits.get("train")
    if train and train.images:
        for idx, name in enumerate(report.names):
            if train.instances[idx] == 0:
                report.warnings.append(f"class '{name}' has no instances in train")
    return report


def print_report(report: DatasetReport) -> None:
    rows = []
    for s in report.splits.values():
        rows.append((s.name, s.images, sum(s.instances.values()), s.background, s.missing_labels))
    try:
        shown = report.yaml_path.relative_to(Path.cwd())
    except ValueError:
        shown = report.yaml_path
    ui.table(f"Dataset · {shown}", ("Split", "Images", "Boxes", "Background", "No label"), rows)

    if report.names and report.splits:
        cols = ["Class", *report.splits]
        class_rows = [
            (f"{i}: {n}", *(s.instances[i] for s in report.splits.values()))
            for i, n in enumerate(report.names)
        ]
        ui.table("Instances per class", cols, class_rows)

    for w in report.warnings:
        ui.warn(w)
    for e in report.errors:
        ui.error(e)
    if report.ok:
        ui.success("Dataset check passed")

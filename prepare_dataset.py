#!/usr/bin/env python3
"""
Merge multiple annotated YOLO directories into a single train/val/test dataset.

Each input directory can be a flat folder (images + .txt labels side by side)
or a CVAT YOLO export (with obj_train_data/ or images/ + labels/ subfolders).

Usage:
    python prepare_dataset.py \
        --input /path/to/folder1 /path/to/folder2 \
        --output /path/to/final_dataset \
        --classes gate torpedo buoy

Output layout (Ultralytics-ready):
    final_dataset/
        images/train/   images/val/   images/test/
        labels/train/   labels/val/   labels/test/
        data.yaml
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge annotated YOLO folders into a training dataset")
    p.add_argument("--input",   required=True, nargs="+", help="One or more annotated directories")
    p.add_argument("--output",  required=True,             help="Output directory for the final dataset")
    p.add_argument("--classes", nargs="+",                 help="Class names (auto-detected if omitted)")
    p.add_argument("--train",   type=float, default=0.85,  help="Train split ratio (default 0.85)")
    p.add_argument("--val",     type=float, default=0.05,  help="Val split ratio (default 0.05)")
    p.add_argument("--test",    type=float, default=0.10,  help="Test split ratio (default 0.10)")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


def detect_classes(input_dir: Path) -> list[str]:
    for name in ("obj.names", "classes.txt", "data.names"):
        for path in input_dir.rglob(name):
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            if lines:
                return lines

    # ── NEW: also check data.yaml ─────────────────────────────────────────
    for path in input_dir.rglob("data.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if names:
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names.keys())]
            return names
        if names:
            return names

    return []

def find_label(img_path: Path, root: Path) -> Path | None:
    # 1. Sibling .txt
    candidate = img_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    # 2. Parallel labels/ directory (CVAT export style)
    parts = list(img_path.parts)
    for segment in ("images", "obj_train_data", "obj_valid_data", "obj_test_data"):
        if segment in parts:
            idx = parts.index(segment)
            parts[idx] = "labels"
            candidate = Path(*parts).with_suffix(".txt")
            if candidate.exists():
                return candidate

    # 3. Global scan within the root
    candidates = list(root.rglob(f"{img_path.stem}.txt"))
    if candidates:
        return candidates[0]

    return None


def collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for img in sorted(input_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = find_label(img, input_dir)
        if label is None:
            skipped += 1
            continue
        pairs.append((img, label))
    if skipped:
        print(f"    [!] {skipped} images skipped (no matching .txt label found)")
    return pairs


def safe_name(img: Path, source_dir: Path, seen: set[str]) -> str:
    """Return a unique filename, prefixing with source dir name on collision."""
    name = img.name
    if name not in seen:
        seen.add(name)
        return name
    # Prefix with the parent folder name to resolve collision
    prefixed = f"{source_dir.name}__{name}"
    if prefixed not in seen:
        seen.add(prefixed)
        return prefixed
    # Last resort: add a counter
    i = 2
    while True:
        numbered = f"{source_dir.name}__{i}__{name}"
        if numbered not in seen:
            seen.add(numbered)
            return numbered
        i += 1


def copy_split(pairs: list[tuple[Path, Path, str]], split: str, out_root: Path) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, lbl, fname in pairs:
        shutil.copy2(img, img_dir / fname)
        shutil.copy2(lbl, lbl_dir / Path(fname).with_suffix(".txt").name)


def write_yaml(out_root: Path, class_names: list[str]) -> Path:
    data = {
        "path":  ".",
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main() -> None:
    args = parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        sys.exit("Error: --train + --val + --test must equal 1.0")

    input_dirs = [Path(d) for d in args.input]
    output_dir = Path(args.output)

    for d in input_dirs:
        if not d.exists():
            sys.exit(f"Error: input directory not found: {d}")

    # ── 1. Collect pairs from all sources ────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Scanning {len(input_dirs)} input director{'y' if len(input_dirs)==1 else 'ies'}")
    print(f"{'─'*55}")

    all_pairs: list[tuple[Path, Path, str]] = []  # (img, lbl, output_filename)
    seen_names: set[str] = set()
    class_names: list[str] = args.classes or []

    for src in input_dirs:
        print(f"\n  [{src.name}]")
        pairs = collect_pairs(src)
        print(f"    Found {len(pairs)} image-label pairs")

        if not class_names:
            detected = detect_classes(src)
            if detected:
                class_names = detected
                print(f"    Classes detected: {detected}")

        for img, lbl in pairs:
            fname = safe_name(img, src, seen_names)
            all_pairs.append((img, lbl, fname))

    if not all_pairs:
        sys.exit("\nError: No valid image-label pairs found across all input directories.")

    print(f"\n{'─'*55}")
    print(f"  Total pairs collected : {len(all_pairs)}")

    # ── 2. Shuffle + split ───────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(all_pairs)

    n       = len(all_pairs)
    n_train = round(n * args.train)
    n_val   = round(n * args.val)

    splits = {
        "train": all_pairs[:n_train],
        "val":   all_pairs[n_train:n_train + n_val],
        "test":  all_pairs[n_train + n_val:],
    }
    print(f"  Split (seed={args.seed})      : train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    # ── 3. Write files ───────────────────────────────────────────────────────
    print(f"\n  Writing to: {output_dir.resolve()}")
    for split_name, split_pairs in splits.items():
        copy_split(split_pairs, split_name, output_dir)
        print(f"    {split_name:<6}: {len(split_pairs)} pairs")

    # ── 4. data.yaml ─────────────────────────────────────────────────────────
    if not class_names:
        print("\n  [!] Could not detect class names — using placeholder ['class0']")
        print("      Edit data.yaml before training.")
        class_names = ["class0"]

    yaml_path = write_yaml(output_dir, class_names)

    print(f"\n{'─'*55}")
    print(f"  Done.")
    print(f"  Classes  : {class_names}")
    print(f"  data.yaml: {yaml_path}")
    print(f"\n  Train with:")
    print(f"    python -m auv_pipeline train --data {yaml_path} --preset yolo26n")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()

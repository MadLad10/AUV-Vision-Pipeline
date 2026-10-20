#!/usr/bin/env python3
"""
Split a CVAT-exported YOLO dataset into train / val / test splits.

Usage:
    python split_dataset.py --input /path/to/cvat_export --output /path/to/dataset

CVAT exports images and labels into flat or nested directories.
This script finds every image-label pair, shuffles deterministically,
splits 85 / 5 / 10, and writes the standard Ultralytics directory layout
plus a ready-to-use data.yaml.
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="85/5/10 YOLO dataset splitter")
    p.add_argument("--input", required=True, nargs="+", help="One or more CVAT YOLO export directories")
    p.add_argument("--output", required=True, help="Output directory for split dataset")
    p.add_argument("--train", type=float, default=0.85)
    p.add_argument("--val",   type=float, default=0.05)
    p.add_argument("--test",  type=float, default=0.10)
    p.add_argument("--seed",  type=int,   default=42, help="Random seed for reproducibility")
    p.add_argument("--no-split", action="store_true",
                   help="Merge only — dump everything into train/, skip val/test split")
    p.add_argument(
        "--classes", nargs="+",
        help="Class names. Auto-detected from obj.names/classes.txt if omitted.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_class_names(input_dir: Path) -> list[str]:
    """Search for CVAT obj.names, classes.txt, or data.yaml and return the class list."""
    for name in ("obj.names", "classes.txt", "data.names"):
        for path in input_dir.rglob(name):
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            if lines:
                return lines
    for path in input_dir.rglob("data.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if names:
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            return list(names)
    return []


def collect_pairs(input_dir: Path, prefix: str = "") -> list[tuple[Path, Path, str, str]]:
    """
    Return (image_path, label_path, out_img_name, out_lbl_name) tuples.

    `prefix` is prepended to filenames to avoid stem collisions when merging
    multiple input directories (e.g. "june30_frame_000450.png").

    Strategy:
      1. Sibling .txt  — same directory as the image.
      2. Sibling labels/ dir — replace 'images' segment with 'labels' in the path.
      3. Global rglob  — last resort scan for a .txt with the same stem.
    """
    pairs: list[tuple[Path, Path, str, str]] = []
    seen_stems: set[str] = set()

    for img_path in sorted(input_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = img_path.stem
        if stem in seen_stems:
            print(f"  [WARN] Duplicate stem '{stem}' within '{input_dir}', skipping {img_path}")
            continue

        # Strategy 1: sibling .txt
        label_path = img_path.with_suffix(".txt")

        # Strategy 2: parallel labels/ directory
        if not label_path.exists():
            parts = list(img_path.parts)
            try:
                idx = parts.index("images")
                parts[idx] = "labels"
                label_path = Path(*parts).with_suffix(".txt")
            except ValueError:
                pass

        # Strategy 3: global scan
        if not label_path.exists():
            candidates = list(input_dir.rglob(f"{stem}.txt"))
            if candidates:
                label_path = candidates[0]

        if not label_path.exists():
            print(f"  [WARN] No label for '{img_path.name}' — skipping")
            continue

        out_stem    = f"{prefix}{stem}" if prefix else stem
        out_img_name = out_stem + img_path.suffix
        out_lbl_name = out_stem + ".txt"
        pairs.append((img_path, label_path, out_img_name, out_lbl_name))
        seen_stems.add(stem)

    return pairs


def copy_pairs(pairs: list[tuple[Path, Path, str, str]], split: str, out_root: Path) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img, lbl, out_img_name, out_lbl_name in pairs:
        shutil.copy2(img, img_dir / out_img_name)
        shutil.copy2(lbl, lbl_dir / out_lbl_name)


def write_yaml(out_root: Path, class_names: list[str]) -> Path:
    data = {
        "path":  str(out_root.resolve()),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--train + --val + --test must equal 1.0 (got {total:.4f})")

    input_dirs = [Path(d) for d in args.input]
    output_dir = Path(args.output)

    for d in input_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Input directory not found: {d}")

    # 1. Collect pairs from all input directories
    use_prefix = len(input_dirs) > 1
    pairs = []
    for input_dir in input_dirs:
        prefix = input_dir.name.replace(" ", "_") + "_" if use_prefix else ""
        print(f"[1/4] Scanning '{input_dir}' for image-label pairs ...")
        p = collect_pairs(input_dir, prefix=prefix)
        print(f"      Found {len(p)} pairs.")
        pairs.extend(p)
    if not pairs:
        raise ValueError("No valid image-label pairs found. Check --input path.")
    print(f"      Total: {len(pairs)} pairs.")

    # 2. Shuffle
    print(f"[2/4] Shuffling with seed={args.seed} ...")
    random.seed(args.seed)
    random.shuffle(pairs)

    if args.no_split:
        splits = {"train": pairs}
        print("      --no-split: all pairs go to train/")
    else:
        n       = len(pairs)
        n_train = round(n * args.train)
        n_val   = round(n * args.val)
        n_test  = n - n_train - n_val

        splits = {
            "train": pairs[:n_train],
            "val":   pairs[n_train : n_train + n_val],
            "test":  pairs[n_train + n_val :],
        }
        print(f"      train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    # 3. Copy files
    print(f"[3/4] Writing to '{output_dir}' ...")
    for split_name, split_pairs in splits.items():
        copy_pairs(split_pairs, split_name, output_dir)
        print(f"      {split_name}: {len(split_pairs)} pairs written")

    # 4. data.yaml
    print("[4/4] Generating data.yaml ...")
    class_names = args.classes or detect_class_names(input_dirs[0])
    if not class_names:
        print("  [WARN] Could not detect class names.")
        print("         Using placeholder ['class0']. Edit data.yaml before training.")
        class_names = ["class0"]

    yaml_path = write_yaml(output_dir, class_names)
    print(f"      Saved: {yaml_path}")

    print(f"\nDone.")
    print(f"  Classes : {class_names}")
    print(f"  Edit {yaml_path} if class names need adjustment.")


if __name__ == "__main__":
    main()

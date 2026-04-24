#!/usr/bin/env python3
"""
Train YOLO26m on a split dataset and export to ONNX.

Same underwater augmentation strategy as train_yolo26.py but uses yolo26m.pt
as the base model. yolo26m is ~8x larger than yolo26n — use a smaller batch
size and expect longer training times.

Intended to run on a training machine (desktop/cloud GPU), NOT on the Jetson.

Usage:
    python train_yolo26m.py --data /path/to/data.yaml
    python train_yolo26m.py --data /path/to/data.yaml --batch 8 --epochs 100
"""

import argparse
from pathlib import Path


def patch_albumentations_underwater() -> None:
    try:
        import albumentations as A
        from ultralytics.data import augment as ult_augment

        underwater_transforms = A.Compose([
            A.Blur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ])

        _original_init = ult_augment.Albumentations.__init__

        def _patched_init(self, p=1.0, **kwargs):
            _original_init(self, p)
            self.transform = underwater_transforms
            print("  [augment] Underwater Albumentations pipeline active")

        ult_augment.Albumentations.__init__ = _patched_init
        print("[augment] Albumentations class patched for underwater training.")

    except ImportError:
        print("[augment] WARNING: albumentations not installed — skipping blur/contrast augmentations.")
    except AttributeError:
        print("[augment] WARNING: Could not patch Albumentations. Proceeding with stock augmentations.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26m for underwater AUV deployment")

    p.add_argument("--data",    required=True, help="Path to data.yaml")
    p.add_argument("--model",   default="yolo26m.pt")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--patience",type=int,   default=20)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=8,   help="Smaller than yolo26n due to model size")
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--device",  default="0")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name",    default="yolo26m_custom")
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    p.set_defaults(export_onnx=True)

    return p.parse_args()


def train(args: argparse.Namespace):
    patch_albumentations_underwater()
    from ultralytics import YOLO

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    print(f"Training on   : {args.data}")
    print(f"Device        : {args.device}  |  imgsz={args.imgsz}  batch={args.batch}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        warmup_epochs=3,
        close_mosaic=10,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
        hsv_h=0.02,
        hsv_s=0.90,
        hsv_v=0.60,
    )


def export(best_pt: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    if args.export_onnx:
        print("\nExporting to ONNX ...")
        model.export(format="onnx", imgsz=args.imgsz, half=True, dynamic=False, simplify=True)
        print(f"  Saved: {best_pt.with_suffix('.onnx')}")


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    train(args)

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\n[WARN] best.pt not found at: {best_pt}")
        return

    print(f"\nBest weights : {best_pt}")
    export(best_pt, args)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()

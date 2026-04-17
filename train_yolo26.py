#!/usr/bin/env python3
"""
Train YOLO26n on a split dataset and export to TensorRT + ONNX.

Augmentation strategy is tuned for AUV / underwater deployment:
  - Heavy HSV shifts: depth causes green/blue dominance and severe brightness variation.
  - Albumentations blur pipeline: AUV motion and suspended particulates cause real-world
    blur that the default Ultralytics augmentations do not cover.
  - RandomBrightnessContrast: simulates depth-dependent lighting falloff.

Intended to run on a training machine (desktop/cloud GPU), NOT on the Jetson.
TensorRT export must be run on a machine with the same TensorRT version as the
Jetson's JetPack installation, or on the Jetson itself.

Usage:
    python train_yolo26.py --data /path/to/dataset/data.yaml
    python train_yolo26.py --data /path/to/data.yaml --imgsz 416 --batch 32
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Underwater augmentation — injected via monkey-patch before training starts
# ---------------------------------------------------------------------------

def patch_albumentations_underwater() -> None:
    """
    Replace Ultralytics' default (near-no-op) Albumentations transforms with a
    pipeline tuned for underwater / AUV conditions.

    Why monkey-patch: Ultralytics does not expose a public API for passing a
    custom A.Compose to model.train(). Patching the class before the trainer
    initialises the dataset is the least-invasive approach; it requires no
    changes to Ultralytics internals and survives version upgrades gracefully
    (it degrades to the stock transforms if albumentations is missing).
    """
    try:
        import albumentations as A
        from ultralytics.data import augment as ult_augment

        underwater_transforms = A.Compose(
            [
                # --- Blur: AUV motion + suspended particulates ----------------
                A.Blur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=(3, 7), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                # --- Lighting / contrast: depth-dependent falloff ------------
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
            ],
        )

        _original_init = ult_augment.Albumentations.__init__

        def _patched_init(self, p=1.0, **kwargs):
            _original_init(self, p)
            self.transform = underwater_transforms
            print(
                "  [augment] Underwater Albumentations pipeline active "
                "(Blur / MotionBlur / GaussianBlur / RandomBrightnessContrast)"
            )

        ult_augment.Albumentations.__init__ = _patched_init
        print("[augment] Albumentations class patched for underwater training.")

    except ImportError:
        print(
            "[augment] WARNING: albumentations not installed — "
            "blur/contrast augmentations will be skipped.\n"
            "          Install with: pip install albumentations"
        )
    except AttributeError:
        print(
            "[augment] WARNING: Could not patch Albumentations "
            "(Ultralytics internal structure may have changed). "
            "Proceeding with stock augmentations."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26n for Jetson Orin Nano deployment")

    # Data / model
    p.add_argument("--data",    required=True, help="Path to data.yaml")
    p.add_argument("--model",   default="yolo26n.pt",
                   help="Ultralytics model string (default: yolo26n.pt)")

    # Training
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--patience",  type=int,   default=20,  help="Early stopping patience")
    p.add_argument("--imgsz",     type=int,   default=640)
    p.add_argument("--batch",     type=int,   default=16,  help="Batch size for training GPU")
    p.add_argument("--workers",   type=int,   default=8)
    p.add_argument("--device",    default="0", help="CUDA device, e.g. 0 / 0,1 / cpu")

    # Output
    p.add_argument("--project",   default="runs/train")
    p.add_argument("--name",      default="yolo26n_custom")

    # Export
    p.add_argument("--no-export-tensorrt", dest="export_tensorrt", action="store_false",
                   help="Skip TensorRT export")
    p.add_argument("--no-export-onnx",     dest="export_onnx",     action="store_false",
                   help="Skip ONNX export")
    p.add_argument("--half", action="store_true", default=True,
                   help="FP16 for exports (recommended for Jetson TensorRT)")
    p.set_defaults(export_tensorrt=True, export_onnx=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    # Patch albumentations BEFORE importing YOLO so the dataset picks it up
    # when the trainer initialises its augmentation pipeline.
    patch_albumentations_underwater()

    from ultralytics import YOLO

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    print(f"Training on   : {args.data}")
    print(f"Device        : {args.device}  |  imgsz={args.imgsz}  batch={args.batch}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,        # early stopping
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        amp=True,                      # mixed precision
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        warmup_epochs=3,
        close_mosaic=10,               # disable mosaic for last N epochs
        save=True,
        save_period=-1,                # only keep best + last
        val=True,
        plots=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
        # --- Underwater HSV augmentation ------------------------------------
        # Defaults are hsv_h=0.015, hsv_s=0.7, hsv_v=0.4.
        # Pushed higher because underwater color shifts dramatically with depth
        # (green/blue dominance, low-contrast scenes, variable lighting).
        hsv_h=0.02,
        hsv_s=0.90,
        hsv_v=0.60,
    )

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(best_pt: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    if args.export_onnx:
        print("\nExporting to ONNX ...")
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            half=args.half,
            dynamic=False,
            simplify=True,
        )
        print(f"  Saved: {best_pt.with_suffix('.onnx')}")

    if args.export_tensorrt:
        print("\nExporting to TensorRT (.engine) ...")
        print(
            "  NOTE: This must run on a machine with TensorRT installed.\n"
            "        For Jetson: run this step directly on the Orin Nano\n"
            "        (or on a Linux x86 host with the same TRT version as JetPack)."
        )
        model.export(
            format="engine",
            imgsz=args.imgsz,
            half=args.half,
            device=args.device,
        )
        print(f"  Saved: {best_pt.with_suffix('.engine')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    train(args)

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\n[WARN] best.pt not found at expected path: {best_pt}")
        print("       Verify --project and --name match your training run.")
        return

    print(f"\nBest weights  : {best_pt}")
    export(best_pt, args)
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()

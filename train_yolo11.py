#!/usr/bin/env python3
"""
Train YOLOv11n on a split dataset and export to TensorRT + ONNX.

Usage:
    python train_yolo11.py --data /path/to/dataset/data.yaml
    python train_yolo11.py --data /path/to/data.yaml --imgsz 416 --batch 32
"""

import argparse
from pathlib import Path


def patch_albumentations_underwater() -> None:
    try:
        import albumentations as A
        from ultralytics.data import augment as ult_augment

        underwater_transforms = A.Compose(
            [
                # --- Brightness/contrast: depth-dependent falloff ------------
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.6
                ),
                A.RandomGamma(gamma_limit=(60, 140), p=0.4),
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
    p = argparse.ArgumentParser(description="Train YOLOv11n for Jetson Orin Nano deployment")

    p.add_argument("--data",    required=True, help="Path to data.yaml")
    p.add_argument("--cfg",     default=None,  help="Path to best_hyperparameters.yaml from tuning")
    p.add_argument("--model",   default="yolo11n.pt",
                   help="Ultralytics model string (default: yolo11n.pt)")

    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--patience",  type=int,   default=20,  help="Early stopping patience")
    p.add_argument("--imgsz",     type=int,   default=640)
    p.add_argument("--batch",     type=int,   default=16)
    p.add_argument("--workers",   type=int,   default=8)
    p.add_argument("--device",    default="0", help="CUDA device, e.g. 0 / 0,1 / cpu")

    p.add_argument("--project",   default="runs/train")
    p.add_argument("--name",      default="yolo11n_custom")

    p.add_argument("--no-export-tensorrt", dest="export_tensorrt", action="store_false",
                   help="Skip TensorRT export")
    p.add_argument("--no-export-onnx",     dest="export_onnx",     action="store_false",
                   help="Skip ONNX export")
    p.add_argument("--half", action="store_true", default=True,
                   help="FP16 for exports (recommended for Jetson TensorRT)")
    p.set_defaults(export_tensorrt=True, export_onnx=True)

    return p.parse_args()


def train(args: argparse.Namespace):
    patch_albumentations_underwater()

    from ultralytics import YOLO

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    print(f"Training on   : {args.data}")
    print(f"Device        : {args.device}  |  imgsz={args.imgsz}  batch={args.batch}")

    if args.cfg:
        print(f"Hyperparams   : {args.cfg}")

    results = model.train(
        cfg=args.cfg or None,
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

    return results


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

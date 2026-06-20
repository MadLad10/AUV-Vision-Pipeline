#!/usr/bin/env python3
"""
Hyperparameter tuning for YOLO models using Ultralytics built-in evolutionary tuner.

Tunes all key training and augmentation hyperparameters and saves the best
config to a YAML file ready to pass to the training script.

Usage:
    python tune_yolo.py --data /path/to/data.yaml
    python tune_yolo.py --data /path/to/data.yaml --model yolo11n.pt --iterations 100
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO hyperparameter tuning")
    p.add_argument("--data",       required=True, help="Path to data.yaml")
    p.add_argument("--model",      default="yolo11n.pt", help="Base model (default: yolo11n.pt)")
    p.add_argument("--epochs",     type=int, default=30,  help="Epochs per trial (keep low, default 30)")
    p.add_argument("--iterations", type=int, default=50,  help="Number of tuning trials (default 50)")
    p.add_argument("--device",     default="0",           help="CUDA device")
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--project",    default="runs/tune")
    p.add_argument("--name",       default="tune_result")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    from ultralytics import YOLO

    print(f"Model      : {args.model}")
    print(f"Data       : {args.data}")
    print(f"Trials     : {args.iterations}  |  Epochs/trial: {args.epochs}")
    print(f"Device     : {args.device}\n")

    model = YOLO(args.model)

    result = model.tune(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        plots=True,
        save=True,
        project=args.project,
        name=args.name,
        # Hyperparameter search space — all key params
        space={
            "lr0":            (1e-5, 1e-1),   # initial learning rate
            "lrf":            (0.01, 1.0),    # final lr = lr0 * lrf
            "momentum":       (0.6,  0.98),
            "weight_decay":   (0.0,  0.001),
            "warmup_epochs":  (0.0,  5.0),
            "warmup_momentum":(0.0,  0.95),
            "box":            (1.0,  20.0),   # box loss gain
            "cls":            (0.2,  4.0),    # cls loss gain
            "dfl":            (0.4,  6.0),    # dfl loss gain
            "hsv_h":          (0.0,  0.1),    # hue augmentation
            "hsv_s":          (0.0,  0.9),    # saturation augmentation
            "hsv_v":          (0.0,  0.9),    # value augmentation
            "degrees":        (0.0,  45.0),   # rotation
            "translate":      (0.0,  0.9),    # translation
            "scale":          (0.0,  0.95),   # scale
            "shear":          (0.0,  10.0),
            "perspective":    (0.0,  0.001),
            "flipud":         (0.0,  1.0),
            "fliplr":         (0.0,  1.0),
            "mosaic":         (0.0,  1.0),
            "mixup":          (0.0,  1.0),
        },
    )

    best_yaml = Path(args.project) / args.name / "best_hyperparameters.yaml"
    print(f"\nTuning complete.")
    print(f"Best hyperparameters saved to: {best_yaml}")
    print(f"\nTrain with best params:")
    print(f"  python train_yolo11.py --data {args.data} --epochs 200 --cfg {best_yaml}")


if __name__ == "__main__":
    main()

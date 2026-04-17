#!/usr/bin/env python3
"""
Structured channel pruning of YOLO26n using Torch-Pruning (BNScaleImportance).

Structured pruning removes entire channels (not individual weights), so the
resulting model is genuinely smaller and faster — not just sparse. This is
required for real speedup on Jetson TensorRT without sparsity support.

Pipeline:
  1. Load best.pt
  2. Prune channels via BN-scale importance (L1 on BatchNorm gamma)
  3. Save pruned checkpoint
  4. Fine-tune for recovery epochs
  5. Export pruned model to ONNX + TensorRT

Usage:
    python prune_yolo26.py \
        --weights runs/train/yolo26n_custom/weights/best.pt \
        --data    /path/to/dataset/data.yaml

    python prune_yolo26.py \
        --weights runs/train/yolo26n_custom/weights/best.pt \
        --data    /path/to/dataset/data.yaml \
        --pruning-ratio 0.40 \
        --finetune-epochs 30
"""

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Structured channel pruning for YOLO26n")

    p.add_argument("--weights",          required=True, help="Path to best.pt (base model)")
    p.add_argument("--data",             required=True, help="Path to data.yaml")
    p.add_argument("--pruning-ratio",    type=float, default=0.30,
                   help="Channel pruning ratio, e.g. 0.30 removes 30%% of channels (default: 0.30)")
    p.add_argument("--finetune-epochs",  type=int,   default=25,
                   help="Fine-tuning epochs after pruning (default: 25)")
    p.add_argument("--imgsz",            type=int,   default=640)
    p.add_argument("--batch",            type=int,   default=16)
    p.add_argument("--device",           default="0")
    p.add_argument("--project",          default="runs/prune")
    p.add_argument("--name",             default="yolo26n_pruned")
    p.add_argument("--half",             action="store_true", default=True,
                   help="FP16 for re-export (recommended for Jetson)")
    p.add_argument("--no-export-tensorrt", dest="export_tensorrt", action="store_false")
    p.add_argument("--no-export-onnx",     dest="export_onnx",     action="store_false")
    p.set_defaults(export_tensorrt=True, export_onnx=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – Prune
# ---------------------------------------------------------------------------

def _ignored_layers(model_nn: torch.nn.Module) -> list:
    """Return layers that must NOT be pruned (detection head changes output shape)."""
    IGNORE_CLASSES = {"Detect", "Segment", "Pose", "OBB", "RTDETRDecoder", "v10Detect"}
    ignored = []
    for m in model_nn.modules():
        if m.__class__.__name__ in IGNORE_CLASSES:
            ignored.append(m)
    return ignored


def prune_model(
    weights_path: Path,
    pruning_ratio: float,
    imgsz: int,
    device_str: str,
) -> tuple:
    """
    Load the YOLO model, apply structured channel pruning in-place, and return
    the updated YOLO wrapper together with pre/post statistics.
    """
    import torch_pruning as tp
    from ultralytics import YOLO

    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    print(f"Loading weights : {weights_path}")
    yolo     = YOLO(str(weights_path))
    model_nn = yolo.model.eval().to(device)

    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(device)

    base_macs, base_params = tp.utils.count_ops_and_params(model_nn, example_inputs)
    print(f"\nBase model  —  params: {base_params/1e6:.2f}M  |  MACs: {base_macs/1e9:.2f}G")

    ignored = _ignored_layers(model_nn)
    print(f"  Ignoring {len(ignored)} head layer(s) from pruning")

    pruner = tp.pruner.MetaPruner(
        model=model_nn,
        example_inputs=example_inputs,
        importance=tp.importance.BNScaleImportance(),
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored,
        round_to=8,      # keep channel counts as multiples of 8 (Tensor Core alignment)
    )

    print(f"\nPruning  (ratio={pruning_ratio}) ...")
    for group in pruner.step(interactive=True):
        group.prune()

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model_nn, example_inputs)
    param_reduction = (1 - pruned_params / base_params) * 100
    mac_reduction   = (1 - pruned_macs   / base_macs)   * 100
    print(
        f"Pruned model —  params: {pruned_params/1e6:.2f}M ({param_reduction:.1f}% reduction)"
        f"  |  MACs: {pruned_macs/1e9:.2f}G ({mac_reduction:.1f}% reduction)"
    )

    stats = dict(
        base_params=base_params, base_macs=base_macs,
        pruned_params=pruned_params, pruned_macs=pruned_macs,
    )
    return yolo, model_nn, stats


def save_pruned_checkpoint(yolo, model_nn: torch.nn.Module, out_path: Path) -> None:
    """
    Persist the pruned model in the Ultralytics checkpoint format so it can be
    reloaded with YOLO(path) for fine-tuning.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep a float32 copy for the checkpoint (training stability)
    model_cpu = copy.deepcopy(model_nn).float().cpu()

    ckpt = {
        "epoch":        -1,
        "best_fitness": None,
        "model":        model_cpu,
        "ema":          None,
        "updates":      0,
        "optimizer":    None,
        "train_args":   {},
        "date":         None,
        "version":      None,
    }
    torch.save(ckpt, out_path)
    print(f"Pruned checkpoint saved : {out_path}")


# ---------------------------------------------------------------------------
# Step 2 – Fine-tune
# ---------------------------------------------------------------------------

def finetune(
    pruned_ckpt: Path,
    data: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> None:
    from ultralytics import YOLO

    print(f"\nFine-tuning pruned model for {epochs} epochs ...")
    model = YOLO(str(pruned_ckpt))
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=True,
        lr0=1e-4,          # intentionally lower than initial training LR
        lrf=1e-3,
        warmup_epochs=1,
        patience=10,
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        project=project,
        name=name,
        exist_ok=False,
    )


# ---------------------------------------------------------------------------
# Step 3 – Re-export
# ---------------------------------------------------------------------------

def export_pruned(best_pt: Path, imgsz: int, half: bool, device: str,
                  export_onnx: bool, export_tensorrt: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    if export_onnx:
        print("\nExporting pruned model to ONNX ...")
        model.export(format="onnx", imgsz=imgsz, half=half, simplify=True)
        print(f"  Saved: {best_pt.with_suffix('.onnx')}")

    if export_tensorrt:
        print("\nExporting pruned model to TensorRT (.engine) ...")
        print(
            "  NOTE: Run this on Jetson or a host with matching TensorRT/JetPack version."
        )
        model.export(format="engine", imgsz=imgsz, half=half, device=device)
        print(f"  Saved: {best_pt.with_suffix('.engine')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    # -- Prune ----------------------------------------------------------------
    yolo, pruned_nn, stats = prune_model(
        weights_path, args.pruning_ratio, args.imgsz, args.device
    )

    pruned_ckpt_path = Path(args.project) / args.name / "weights" / "pruned_init.pt"
    save_pruned_checkpoint(yolo, pruned_nn, pruned_ckpt_path)

    # -- Fine-tune ------------------------------------------------------------
    finetune(
        pruned_ckpt=pruned_ckpt_path,
        data=args.data,
        epochs=args.finetune_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name + "_ft",
    )

    best_pt = Path(args.project) / (args.name + "_ft") / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"\n[WARN] Fine-tuned best.pt not found at: {best_pt}")
        print("       Check --project / --name arguments.")
        return

    print(f"\nFine-tuned pruned weights : {best_pt}")

    # -- Re-export ------------------------------------------------------------
    export_pruned(
        best_pt, args.imgsz, args.half, args.device,
        args.export_onnx, args.export_tensorrt,
    )

    print("\nPruning pipeline complete.")
    print(f"  Param reduction : {(1 - stats['pruned_params']/stats['base_params'])*100:.1f}%")
    print(f"  MAC  reduction  : {(1 - stats['pruned_macs']/stats['base_macs'])*100:.1f}%")


if __name__ == "__main__":
    main()

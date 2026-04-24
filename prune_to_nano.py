#!/usr/bin/env python3
"""
Iterative structured pruning: yolo26m → yolo26n equivalent (~2.4 M params).

yolo26m starts at ~20 M params. Two rounds of 40% channel pruning reduce
conv layer params to roughly (0.6 * 0.6)^2 ≈ 13% of the original, bringing
the model to ~2-3 M params — comparable to yolo26n.

Each round:
  1. Prune channels by the per-round ratio using BNScaleImportance
  2. Save intermediate checkpoint
  3. Fine-tune for recovery epochs

After all rounds the final model is exported to ONNX.

Usage:
    python prune_to_nano.py \
        --weights runs/train/yolo26m_custom/weights/best.pt \
        --data    /path/to/data.yaml

    # Custom schedule: 3 rounds of 35% pruning, 20 epochs each
    python prune_to_nano.py \
        --weights best.pt --data data.yaml \
        --rounds 3 --ratio-per-round 0.35 --finetune-epochs 20
"""

import argparse
import copy
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative prune yolo26m → nano equivalent")

    p.add_argument("--weights",           required=True, help="Path to yolo26m best.pt")
    p.add_argument("--data",              required=True, help="Path to data.yaml")
    p.add_argument("--rounds",            type=int,   default=2,
                   help="Number of prune+finetune rounds (default: 2)")
    p.add_argument("--ratio-per-round",   type=float, default=0.40,
                   help="Channel pruning ratio per round (default: 0.40 = 40%%)")
    p.add_argument("--finetune-epochs",   type=int,   default=20,
                   help="Fine-tuning epochs after each round (default: 20)")
    p.add_argument("--imgsz",             type=int,   default=640)
    p.add_argument("--batch",             type=int,   default=16)
    p.add_argument("--device",            default="0")
    p.add_argument("--project",           default="runs/prune_to_nano")
    p.add_argument("--name",              default="yolo26m_to_nano")
    p.add_argument("--no-export-onnx",    dest="export_onnx", action="store_false")
    p.set_defaults(export_onnx=True)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ignored_layers(model_nn: torch.nn.Module) -> list:
    IGNORE = {"Detect", "Segment", "Pose", "OBB", "RTDETRDecoder", "v10Detect"}
    return [m for m in model_nn.modules() if m.__class__.__name__ in IGNORE]


def count_params_macs(model_nn, imgsz, device):
    import torch_pruning as tp
    x = torch.randn(1, 3, imgsz, imgsz).to(device)
    macs, params = tp.utils.count_ops_and_params(model_nn, x)
    return params, macs


def prune_one_round(model_nn, ratio, imgsz, device):
    import torch_pruning as tp

    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(device)
    ignored        = _ignored_layers(model_nn)

    pruner = tp.pruner.MetaPruner(
        model          = model_nn,
        example_inputs = example_inputs,
        importance     = tp.importance.BNScaleImportance(),
        pruning_ratio  = ratio,
        ignored_layers = ignored,
        round_to       = 1,
    )
    for group in pruner.step(interactive=True):
        group.prune()


def save_checkpoint(yolo, model_nn, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": -1, "best_fitness": None,
        "model": copy.deepcopy(model_nn).float().cpu(),
        "ema": None, "updates": 0, "optimizer": None,
        "train_args": {}, "date": None, "version": None,
    }
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def finetune(ckpt_path: Path, args, round_name: str) -> Path:
    from ultralytics import YOLO

    print(f"\nFine-tuning ({round_name}) for {args.finetune_epochs} epochs ...")
    model = YOLO(str(ckpt_path))
    model.train(
        data          = args.data,
        epochs        = args.finetune_epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        device        = args.device,
        amp           = True,
        lr0           = 5e-5,
        lrf           = 1e-3,
        warmup_epochs = 1,
        patience      = 10,
        save          = True,
        save_period   = -1,
        val           = True,
        plots         = True,
        project       = args.project,
        name          = round_name,
        exist_ok      = False,
    )
    best = Path(args.project) / round_name / "weights" / "best.pt"
    return best


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

    from ultralytics import YOLO

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    print(f"Loading: {weights_path}")
    yolo     = YOLO(str(weights_path))
    model_nn = yolo.model.eval().to(device)

    base_params, base_macs = count_params_macs(model_nn, args.imgsz, device)
    print(f"\nBase model  — params: {base_params/1e6:.2f}M  |  MACs: {base_macs/1e9:.2f}G")
    print(f"Target      — params: ~2.4M (yolo26n equivalent)\n")

    current_weights = weights_path

    for rnd in range(1, args.rounds + 1):
        print(f"{'='*60}")
        print(f"Round {rnd}/{args.rounds}  —  pruning {args.ratio_per_round*100:.0f}% of channels")
        print(f"{'='*60}")

        # Reload from last checkpoint
        yolo     = YOLO(str(current_weights))
        model_nn = yolo.model.eval().to(device)

        before_params, _ = count_params_macs(model_nn, args.imgsz, device)

        prune_one_round(model_nn, args.ratio_per_round, args.imgsz, device)

        after_params, after_macs = count_params_macs(model_nn, args.imgsz, device)
        reduction = (1 - after_params / before_params) * 100
        print(f"  After pruning — params: {after_params/1e6:.2f}M "
              f"({reduction:.1f}% reduction this round)  |  MACs: {after_macs/1e9:.2f}G")

        ckpt_path = Path(args.project) / f"round{rnd}_pruned" / "weights" / "pruned_init.pt"
        save_checkpoint(yolo, model_nn, ckpt_path)

        round_name    = f"{args.name}_r{rnd}"
        current_weights = finetune(ckpt_path, args, round_name)

        if not current_weights.exists():
            print(f"[WARN] Fine-tuned weights not found at {current_weights}. Stopping.")
            break

        print(f"\nRound {rnd} complete. Weights: {current_weights}")

    # Final summary
    print(f"\n{'='*60}")
    print("Iterative pruning complete.")
    final_yolo     = YOLO(str(current_weights))
    final_model_nn = final_yolo.model.eval().to(device)
    final_params, final_macs = count_params_macs(final_model_nn, args.imgsz, device)
    total_reduction = (1 - final_params / base_params) * 100
    print(f"  Original  : {base_params/1e6:.2f}M params")
    print(f"  Final     : {final_params/1e6:.2f}M params  ({total_reduction:.1f}% total reduction)")
    print(f"  MACs      : {final_macs/1e9:.2f}G")
    print(f"  yolo26n   : ~2.4M params / 5.2G MACs  (reference)")

    # Export
    if args.export_onnx:
        print("\nExporting final pruned model to ONNX ...")
        final_yolo.export(format="onnx", imgsz=args.imgsz, half=True, simplify=True)
        print(f"  Saved: {current_weights.with_suffix('.onnx')}")

    print(f"\nFinal weights: {current_weights}")


if __name__ == "__main__":
    main()

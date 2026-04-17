#!/usr/bin/env python3
"""
Benchmark base vs pruned YOLO26n.

Metrics reported:
  mAP50, mAP50-95, params (M), MACs (G), file size (MB),
  mean inference latency (ms), FPS.

Usage:
    python benchmark.py \
        --base   runs/train/yolo26n_custom/weights/best.pt \
        --pruned runs/prune/yolo26n_pruned_ft/weights/best.pt \
        --data   /path/to/dataset/data.yaml

    # CPU-only machine:
    python benchmark.py --base ... --pruned ... --data ... --device cpu
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark base vs pruned YOLO26n")
    p.add_argument("--base",    required=True, help="Base model weights (best.pt)")
    p.add_argument("--pruned",  required=True, help="Pruned model weights (best.pt)")
    p.add_argument("--data",    required=True, help="Path to data.yaml")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=1, help="Batch size for latency test")
    p.add_argument("--warmup",  type=int,   default=50,  help="Warmup iterations")
    p.add_argument("--iters",   type=int,   default=200, help="Measurement iterations")
    p.add_argument("--device",  default="0")
    p.add_argument("--conf",    type=float, default=0.001, help="Confidence threshold for val")
    p.add_argument("--iou",     type=float, default=0.60,  help="IoU threshold for val NMS")
    return p.parse_args()


# ---------------------------------------------------------------------------
# mAP evaluation
# ---------------------------------------------------------------------------

def evaluate_map(weights: Path, data: str, imgsz: int, conf: float,
                 iou: float, device: str) -> tuple[float, float]:
    from ultralytics import YOLO

    model   = YOLO(str(weights))
    metrics = model.val(
        data=data, imgsz=imgsz, conf=conf, iou=iou,
        device=device, verbose=False,
    )
    return float(metrics.box.map50), float(metrics.box.map)


# ---------------------------------------------------------------------------
# Param / MACs counting
# ---------------------------------------------------------------------------

def count_params_macs(model_nn: torch.nn.Module, imgsz: int,
                      device: torch.device) -> tuple[int, float | None]:
    try:
        import torch_pruning as tp
        dummy  = torch.randn(1, 3, imgsz, imgsz).to(device)
        macs, params = tp.utils.count_ops_and_params(model_nn, dummy)
        return int(params), float(macs)
    except Exception as exc:
        print(f"  [WARN] torch_pruning count failed ({exc}); using parameter count only.")
        return sum(p.numel() for p in model_nn.parameters()), None


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    model_nn: torch.nn.Module,
    imgsz: int,
    batch: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for pure forward-pass latency."""
    model_nn.eval()
    dummy = torch.randn(batch, 3, imgsz, imgsz).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model_nn(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                starts[i].record()
                _ = model_nn(dummy)
                ends[i].record()
            torch.cuda.synchronize()
            latencies = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        else:
            latencies = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model_nn(dummy)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


# ---------------------------------------------------------------------------
# Full benchmark for one model
# ---------------------------------------------------------------------------

def benchmark_one(label: str, weights: Path, args: argparse.Namespace,
                  device: torch.device) -> dict:
    from ultralytics import YOLO

    print(f"\n{'=' * 52}")
    print(f"  {label}")
    print(f"{'=' * 52}")

    print("  Evaluating mAP on test split ...")
    map50, map5095 = evaluate_map(weights, args.data, args.imgsz,
                                   args.conf, args.iou, args.device)

    yolo     = YOLO(str(weights))
    model_nn = yolo.model.to(device).eval()

    params, macs = count_params_macs(model_nn, args.imgsz, device)
    size_mb      = weights.stat().st_size / (1024 ** 2)

    print("  Measuring inference latency ...")
    lat_mean, lat_std = measure_latency(
        model_nn, args.imgsz, args.batch, args.warmup, args.iters, device
    )
    fps = 1000.0 / lat_mean

    result = dict(
        label=label,
        mAP50=map50,
        mAP5095=map5095,
        params_M=params / 1e6,
        macs_G=macs / 1e9 if macs else None,
        size_mb=size_mb,
        lat_mean_ms=lat_mean,
        lat_std_ms=lat_std,
        fps=fps,
    )

    print(f"  mAP50        : {map50:.4f}")
    print(f"  mAP50-95     : {map5095:.4f}")
    print(f"  Params       : {params/1e6:.2f} M")
    if macs:
        print(f"  MACs         : {macs/1e9:.2f} G")
    print(f"  File size    : {size_mb:.1f} MB")
    print(f"  Latency      : {lat_mean:.2f} ± {lat_std:.2f} ms")
    print(f"  FPS          : {fps:.1f}")

    return result


# ---------------------------------------------------------------------------
# Side-by-side comparison table
# ---------------------------------------------------------------------------

def print_comparison(base: dict, pruned: dict) -> None:
    def fmt(v: float | None, decimals: int = 3) -> str:
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    def delta(bv: float | None, pv: float | None, higher_is_better: bool = False) -> str:
        if bv is None or pv is None or abs(bv) < 1e-12:
            return "N/A"
        pct = (pv - bv) / abs(bv) * 100
        sign = "+" if pct >= 0 else ""
        # Colour interpretation: accuracy metrics want higher; resource metrics want lower
        arrow = ("▲" if pct >= 0 else "▼") if higher_is_better else ("▼" if pct <= 0 else "▲")
        return f"{sign}{pct:.1f}% {arrow}"

    rows = [
        ("mAP50",         "mAP50",      3, True),
        ("mAP50-95",      "mAP5095",    3, True),
        ("Params (M)",    "params_M",   2, False),
        ("MACs (G)",      "macs_G",     2, False),
        ("Size (MB)",     "size_mb",    1, False),
        ("Latency (ms)",  "lat_mean_ms",2, False),
        ("FPS",           "fps",        1, True),
    ]

    print(f"\n{'=' * 62}")
    print(f"  {'COMPARISON SUMMARY':^58}")
    print(f"{'=' * 62}")
    print(f"  {'Metric':<18} {'Base':>12} {'Pruned':>12} {'Delta':>14}")
    print(f"  {'-'*58}")
    for display, key, dec, hib in rows:
        bv = base.get(key)
        pv = pruned.get(key)
        print(
            f"  {display:<18} {fmt(bv, dec):>12} {fmt(pv, dec):>12} {delta(bv, pv, hib):>14}"
        )
    print(f"{'=' * 62}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_path   = Path(args.base)
    pruned_path = Path(args.pruned)

    for p in (base_path, pruned_path, Path(args.data)):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)

    base_results   = benchmark_one("BASE MODEL",   base_path,   args, device)
    pruned_results = benchmark_one("PRUNED MODEL", pruned_path, args, device)

    print_comparison(base_results, pruned_results)


if __name__ == "__main__":
    main()

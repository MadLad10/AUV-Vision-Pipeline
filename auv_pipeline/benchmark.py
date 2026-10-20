"""Stage 5 — compare models side by side: mAP, params, MACs, size, latency, FPS.

The first model is the reference; every other column shows its delta. Works on
.pt weights and on exported .onnx / .engine files (params/MACs are .pt-only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import ui

# (label, key, decimals, higher_is_better)
ROWS = [
    ("mAP50", "mAP50", 3, True),
    ("mAP50-95", "mAP50-95", 3, True),
    ("Params (M)", "params_M", 2, False),
    ("GMACs", "macs_G", 2, False),
    ("Size (MB)", "size_mb", 1, False),
    ("Latency (ms)", "latency_ms", 2, False),
    ("FPS", "fps", 1, True),
]


def _torch_latency(model_nn, imgsz: int, warmup: int, iters: int, device) -> float:
    import numpy as np
    import torch

    model_nn.eval()
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model_nn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for s, e in zip(starts, ends):
                s.record()
                model_nn(x)
                e.record()
            torch.cuda.synchronize()
            return float(np.mean([s.elapsed_time(e) for s, e in zip(starts, ends)]))
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model_nn(x)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(times))


def _predict_latency(yolo, imgsz: int, warmup: int, iters: int, device) -> float:
    """Inference-only latency through Ultralytics' predictor (for .onnx / .engine)."""
    import numpy as np

    frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    times = []
    for i in range(warmup + iters):
        r = yolo.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
        if i >= warmup:
            times.append(r.speed["inference"])
    return float(np.mean(times))


def benchmark_one(weights: Path, data: str, *, imgsz: int, device: str | None, split: str,
                  warmup: int, iters: int, conf: float, iou: float) -> dict:
    import torch
    from ultralytics import YOLO

    from .prune import _device
    from .results import evaluate

    dev = _device(device)
    ui.section(f"Benchmark · {weights}")
    row: dict = {"model": str(weights), "size_mb": _size_mb(weights)}
    row.update(evaluate(weights, data, split=split, imgsz=imgsz, conf=conf, iou=iou, device=device, plots=False))

    yolo = YOLO(str(weights))
    if weights.suffix == ".pt":
        try:
            from .prune import count_params_macs

            params, macs = count_params_macs(yolo.model.to(dev), imgsz, dev)
            row.update(params_M=params / 1e6, macs_G=macs / 1e9)
        except ImportError:
            row["params_M"] = sum(p.numel() for p in yolo.model.parameters()) / 1e6
        with torch.no_grad():
            row["latency_ms"] = _torch_latency(yolo.model.to(dev), imgsz, warmup, iters, dev)
    else:
        row["latency_ms"] = _predict_latency(yolo, imgsz, warmup, iters, device)
    row["fps"] = 1000.0 / row["latency_ms"]
    return row


def _size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20
    return path.stat().st_size / 2**20


def _fmt(v, d: int) -> str:
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"


def _delta(ref, v, higher_is_better: bool) -> str:
    if not isinstance(ref, (int, float)) or not isinstance(v, (int, float)) or abs(ref) < 1e-12:
        return ""
    pct = (v - ref) / abs(ref) * 100
    good = pct >= 0 if higher_is_better else pct <= 0
    return f" ({pct:+.1f}% {'▲' if pct >= 0 else '▼'}{'' if good else '!'})"


def comparison_rows(results: list[dict]) -> list[list[str]]:
    ref = results[0]
    return [
        [label, *(_fmt(r.get(key), d) + (_delta(ref.get(key), r.get(key), hib) if i else "")
                  for i, r in enumerate(results))]
        for label, key, d, hib in ROWS
    ]


def to_markdown(results: list[dict]) -> str:
    names = [r.get("label") or Path(r["model"]).name for r in results]
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += ["| " + " | ".join(row) + " |" for row in comparison_rows(results)]
    return "\n".join(lines)


def benchmark(models: list[Path], data: str, *, labels: list[str] | None = None, save: Path | None = None,
              **kwargs) -> list[dict]:
    results = [benchmark_one(Path(m), data, **kwargs) for m in models]
    for r, label in zip(results, labels or []):
        r["label"] = label
    headers = [r.get("label") or Path(r["model"]).name for r in results]
    ui.table("Comparison (deltas vs first model)", ("Metric", *headers), comparison_rows(results))
    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        if save.suffix == ".md":
            save.write_text(to_markdown(results) + "\n")
        else:
            save.write_text(json.dumps(results, indent=2))
        ui.success(f"Saved {save}")
    return results

"""Stage 4 — export to ONNX / TensorRT for the Jetson Orin Nano.

TensorRT engines are tied to the TensorRT version that built them, so build the
``engine`` on the Jetson itself (or a host with the same JetPack TensorRT).
INT8 needs calibration images: pass the dataset's data.yaml.
"""

from __future__ import annotations

from pathlib import Path

from . import ui

FORMATS = ("onnx", "engine", "torchscript", "openvino")


def export(
    weights: Path,
    formats: list[str],
    imgsz: int = 640,
    half: bool = True,
    int8: bool = False,
    data: str | None = None,
    device: str | None = None,
    workspace: float | None = None,
) -> list[tuple[str, Path]]:
    import torch
    from ultralytics import YOLO

    if int8 and not data:
        raise ValueError("INT8 export needs --data (calibration images)")
    has_cuda = torch.cuda.is_available()
    artifacts: list[tuple[str, Path]] = []

    for fmt in formats:
        kwargs: dict = {"format": fmt, "imgsz": imgsz}
        if fmt == "engine":
            if not has_cuda:
                ui.warn(
                    "TensorRT export skipped: no CUDA device here. On the Jetson run:\n"
                    f"    python -m auv_pipeline export --weights {weights} --formats engine"
                )
                continue
            kwargs.update(device=device if device is not None else 0, half=half and not int8)
            if int8:
                kwargs.update(int8=True, data=data)
            if workspace:
                kwargs["workspace"] = workspace
        elif fmt == "onnx":
            # FP16 ONNX only works when exporting on a GPU
            kwargs.update(simplify=True, dynamic=False, half=half and has_cuda)
            if has_cuda:
                kwargs["device"] = device if device is not None else 0

        ui.info(f"Exporting {weights.name} -> {fmt} ({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
        out = YOLO(str(weights)).export(**kwargs)
        artifacts.append((fmt, Path(out)))
        ui.success(f"{fmt}: {out}")

    if artifacts:
        ui.table(
            "Exported artifacts",
            ("Format", "File", "Size (MB)"),
            [(f, p, f"{_size_mb(p):.1f}") for f, p in artifacts],
        )
    return artifacts


def _size_mb(path: Path) -> float:
    if path.is_dir():  # openvino / saved_model exports are directories
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20
    return path.stat().st_size / 2**20 if path.exists() else 0.0

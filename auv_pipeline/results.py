"""Reading Ultralytics run directories, evaluating weights, and writing run summaries."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import ui
from .config import REPO_ROOT, load_yaml

METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def read_results_csv(run_dir: Path) -> dict | None:
    """Best epoch (by mAP50-95) from a run's results.csv, or None if it has none."""
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get(METRIC_COLUMNS["mAP50-95"], 0) or 0))
    out = {"best_epoch": int(float(best["epoch"])), "epochs_run": len(rows)}
    out.update({k: float(best[col]) for k, col in METRIC_COLUMNS.items() if col in best})
    if "time" in rows[-1]:
        out["train_time_s"] = float(rows[-1]["time"])
    return out


def run_report(run_dir: Path) -> dict:
    """Metrics + key training args for a finished run directory (e.g. saved_runs/Torpedo)."""
    run_dir = Path(run_dir)
    report: dict = {"run": run_dir.name, "metrics": read_results_csv(run_dir) or {}}
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        args = load_yaml(args_file)
        report["args"] = {k: args.get(k) for k in ("model", "epochs", "imgsz", "batch", "optimizer", "patience")}
    return report


def print_run_report(report: dict, markdown: bool = False) -> None:
    m, a = report.get("metrics", {}), report.get("args", {})
    rows = [
        ("Model", a.get("model", "?")),
        ("Image size", a.get("imgsz", "?")),
        ("Epochs run / max", f"{m.get('epochs_run', '?')} / {a.get('epochs', '?')}"),
        ("Best epoch", m.get("best_epoch", "?")),
        ("Precision", f"{m['precision']:.3f}" if "precision" in m else "?"),
        ("Recall", f"{m['recall']:.3f}" if "recall" in m else "?"),
        ("mAP50", f"{m['mAP50']:.3f}" if "mAP50" in m else "?"),
        ("mAP50-95", f"{m['mAP50-95']:.3f}" if "mAP50-95" in m else "?"),
    ]
    if "train_time_s" in m:
        rows.append(("Train time", f"{m['train_time_s'] / 60:.1f} min"))
    if markdown:
        print(f"| {report['run']} | |\n|---|---|")
        for k, v in rows:
            print(f"| {k} | {v} |")
    else:
        ui.kv(f"Run · {report['run']}", rows)


def evaluate(weights: Path, data: str, split: str = "val", **kwargs) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=data, split=split, verbose=False, **kwargs)
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }


def git_info() -> dict:
    def _git(*cmd: str) -> str:
        try:
            return subprocess.run(
                ["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def write_summary(path: Path, payload: dict) -> Path:
    payload = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_info(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

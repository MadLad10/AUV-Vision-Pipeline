"""Command line: ``python -m auv_pipeline <stage> ...``"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__, ui
from .config import list_presets, parse_set

EXAMPLES = """\
examples:
  python -m auv_pipeline check     --data data.yaml
  python -m auv_pipeline augment   --profile physics --images dataset/images/train
  python -m auv_pipeline train     --data data.yaml --preset yolo26n --augment physics
  python -m auv_pipeline prune     --weights best.pt --data data.yaml --ratio 0.3
  python -m auv_pipeline prune     --weights yolo26m_best.pt --data data.yaml --rounds 2 --ratio 0.4
  python -m auv_pipeline export    --weights best.pt --formats engine --int8 --data data.yaml
  python -m auv_pipeline benchmark --models base.pt pruned.pt --data data.yaml
  python -m auv_pipeline report    saved_runs/Torpedo
  python -m auv_pipeline run       --data data.yaml --preset yolo26n
"""


# ---------------------------------------------------------------------------
# Argument groups shared by several stages
# ---------------------------------------------------------------------------

def _train_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("training overrides (default: from the preset)")
    g.add_argument("--model", help="weights / model name, e.g. yolo26n.pt or a previous best.pt")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch", type=int)
    g.add_argument("--imgsz", type=int)
    g.add_argument("--patience", type=int)
    g.add_argument("--workers", type=int)
    g.add_argument("--freeze", type=int, help="freeze the first N layers")
    g.add_argument("--device", help="0 | 0,1 | cpu (default: auto)")
    g.add_argument("--hyp", "--cfg", dest="hyp", help="best_hyperparameters.yaml from tune_yolo.py")
    g.add_argument("--set", nargs="+", action="extend", metavar="KEY=VALUE", default=[],
                   help="any Ultralytics train arg, e.g. --set mosaic=0.5 cos_lr=true")


def _export_args(p: argparse.ArgumentParser, default_formats: list[str] | None) -> None:
    g = p.add_argument_group("export")
    g.add_argument("--formats", nargs="+", default=default_formats, metavar="FMT",
                   help="onnx engine torchscript openvino (default: from the preset)")
    g.add_argument("--no-half", dest="half", action="store_false", default=None, help="FP32 instead of FP16")
    g.add_argument("--int8", action="store_true", help="TensorRT INT8 (calibrates on --data)")
    g.add_argument("--workspace", type=float, help="TensorRT builder workspace (GB)")


def _overrides(args) -> dict:
    return {k: getattr(args, k) for k in ("epochs", "batch", "imgsz", "patience", "workers", "freeze", "device")}


def _cfg_from_args(args):
    from .train import build_config

    return build_config(
        args.preset, model=args.model, name=getattr(args, "name", None), augment=args.augment,
        hyp=args.hyp, overrides=_overrides(args), sets=parse_set(args.set),
    )


def _dataset_ok(data: str):
    from .dataset import inspect_dataset, print_report

    report = inspect_dataset(data)
    print_report(report)
    return report


# ---------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------

def cmd_presets(args) -> int:
    from .augment import load_profiles

    ui.table("Training presets (configs/train/)", ("Preset", "Description"), list_presets().items())
    ui.table("Augment profiles (configs/augment/underwater.yaml)", ("Profile", "Description"),
             [(k, (v or {}).get("description", "")) for k, v in load_profiles().items()])
    return 0


def cmd_check(args) -> int:
    ui.banner("Dataset check", args.data)
    return 0 if _dataset_ok(args.data).ok else 1


def cmd_augment(args) -> int:
    import cv2

    from .augment import describe, get_profile
    from .augment.preview import render

    profile = get_profile(args.profile)
    ui.banner(f"Augment profile · {args.profile}", profile.get("description", ""))
    ui.table("Transforms", ("#", "Transform"), list(enumerate(describe(profile), 1)))
    grid = render(args.profile, profile, args.images, args.samples, args.size, args.cols, args.seed)
    out = Path(args.out or f"runs/augment/{args.profile}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    ui.success(f"Preview saved to {out}")
    if args.show:
        cv2.imshow(f"augment: {args.profile}", grid)
        cv2.waitKey(0)
    return 0


def cmd_train(args) -> int:
    from .export import export
    from .results import evaluate, read_results_csv, write_summary
    from .train import print_config, train

    if args.resume:
        ui.banner("Resuming training", args.resume)
        train({}, "", resume=args.resume)
        return 0
    if not args.data:
        ui.error("--data is required (or --resume)")
        return 2

    cfg = _cfg_from_args(args)
    ui.banner("AUV vision pipeline", f"train · preset {args.preset} · augment {cfg['augment']}")
    report = _dataset_ok(args.data)
    print_config(cfg, args.data, args.project)
    if not report.ok:
        return 1
    if args.dry_run:
        ui.success("Dry run: config and dataset OK, nothing trained")
        return 0

    save_dir = train(cfg, args.data, args.project)
    best = save_dir / "weights" / "best.pt"
    summary: dict = {"preset": args.preset, "augment": cfg["augment"], "data": args.data,
                     "model": cfg["model"], "save_dir": str(save_dir), "val": read_results_csv(save_dir)}
    imgsz = cfg["train"].get("imgsz", 640)
    device = cfg["train"].get("device")

    if report.has_test and not args.no_test:
        ui.section("Held-out test split")
        summary["test"] = evaluate(best, args.data, split="test", imgsz=imgsz, device=device,
                                   project=str(save_dir), name="test_eval")

    formats = [] if args.no_export else (args.formats or cfg["export"].get("formats", []))
    half = cfg["export"].get("half", True) if args.half is None else args.half
    if formats:
        artifacts = export(best, formats, imgsz=imgsz, half=half, int8=args.int8, data=args.data,
                           device=device, workspace=args.workspace)
        summary["exports"] = {f: str(p) for f, p in artifacts}

    path = write_summary(save_dir / "run_summary.json", summary)
    _print_metrics(summary)
    ui.success(f"Run summary: {path}")
    return 0


def _print_metrics(summary: dict) -> None:
    rows = []
    for split in ("val", "test"):
        m = summary.get(split) or {}
        if m:
            keys = ("precision", "recall", "mAP50", "mAP50-95")
            rows.append((split, *(f"{m[k]:.3f}" if k in m else "—" for k in keys)))
    if rows:
        ui.table("Results", ("Split", "Precision", "Recall", "mAP50", "mAP50-95"), rows)


def cmd_prune(args) -> int:
    from .export import export
    from .prune import prune

    ui.banner("Structured pruning", f"{args.weights} · {args.rounds} round(s) × {args.ratio:.0%}")
    results = prune(
        Path(args.weights), args.data, ratio=args.ratio, rounds=args.rounds,
        finetune_epochs=args.finetune_epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        lr0=args.lr0, augment=args.augment, round_to=args.round_to, project=args.project, name=args.name,
    )
    if args.formats:
        export(results[-1].weights, args.formats, imgsz=args.imgsz, half=args.half is not False,
               int8=args.int8, data=args.data, device=args.device, workspace=args.workspace)
    return 0


def cmd_export(args) -> int:
    from .export import export

    ui.banner("Export", f"{args.weights} → {', '.join(args.formats)}")
    export(Path(args.weights), args.formats, imgsz=args.imgsz, half=args.half is not False, int8=args.int8,
           data=args.data, device=args.device, workspace=args.workspace)
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import benchmark

    ui.banner("Benchmark", f"{len(args.models)} model(s) on {args.data} [{args.split}]")
    benchmark([Path(m) for m in args.models], args.data, labels=args.labels, save=args.save,
              imgsz=args.imgsz, device=args.device, split=args.split, warmup=args.warmup,
              iters=args.iters, conf=args.conf, iou=args.iou)
    return 0


def cmd_report(args) -> int:
    from .results import print_run_report, run_report

    for run_dir in args.runs:
        print_run_report(run_report(Path(run_dir)), markdown=args.markdown)
    return 0


def cmd_run(args) -> int:
    """All stages end to end: check → augment preview → train → prune → export → benchmark."""
    import cv2

    from .augment.preview import render
    from .benchmark import benchmark
    from .export import export
    from .prune import prune
    from .results import read_results_csv, write_summary
    from .train import print_config, train

    cfg = _cfg_from_args(args)
    root = Path(args.project) / f"{cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    skip = set(args.skip)
    total = 6
    ui.banner("AUV vision pipeline — full run", f"preset {args.preset} · output {root}")

    ui.stage(1, total, "Dataset check")
    report = _dataset_ok(args.data)
    if not report.ok:
        return 1

    ui.stage(2, total, f"Underwater augmentation · {cfg['augment']}")
    print_config(cfg, args.data, str(root))
    sources = [str(s) for s in report.sources.get("train", [])]
    if cfg["_profile"].get("transforms") and sources:
        root.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "augment_preview.jpg"), render(cfg["augment"], cfg["_profile"], sources))
        ui.success(f"Preview: {root / 'augment_preview.jpg'}")
    if args.dry_run:
        ui.success("Dry run: stopping before training")
        return 0

    ui.stage(3, total, "Training")
    save_dir = train(cfg, args.data, str(root))
    best = save_dir / "weights" / "best.pt"
    imgsz = cfg["train"].get("imgsz", 640)
    device = cfg["train"].get("device")
    summary: dict = {"preset": args.preset, "augment": cfg["augment"], "train": str(save_dir),
                     "val": read_results_csv(save_dir)}

    deploy = best
    ui.stage(4, total, "Pruning")
    if "prune" in skip:
        ui.info("skipped")
    else:
        rounds = prune(best, args.data, ratio=args.prune_ratio, rounds=args.rounds,
                       finetune_epochs=args.finetune_epochs, imgsz=imgsz, batch=cfg["train"].get("batch", 16),
                       device=device, augment=cfg["augment"], project=str(root), name="prune")
        deploy = rounds[-1].weights
        summary["prune"] = [{"round": r.round, "params": r.params, "macs": r.macs, "weights": str(r.weights)}
                            for r in rounds]

    ui.stage(5, total, "Export (ONNX / TensorRT)")
    formats = args.formats or cfg["export"].get("formats", [])
    if "export" in skip or not formats:
        ui.info("skipped")
    else:
        artifacts = export(deploy, formats, imgsz=imgsz, half=cfg["export"].get("half", True), int8=args.int8,
                           data=args.data, device=device)
        summary["exports"] = {f: str(p) for f, p in artifacts}

    ui.stage(6, total, "Benchmark")
    if "benchmark" in skip or deploy == best:
        ui.info("skipped" if "benchmark" in skip else "nothing to compare (pruning skipped)")
    else:
        benchmark([best, deploy], args.data, labels=["trained", "pruned"], save=root / "benchmark.md",
                  imgsz=imgsz, device=device, split="test" if report.has_test else "val",
                  warmup=20, iters=100, conf=0.001, iou=0.6)

    path = write_summary(root / "pipeline_summary.json", summary)
    ui.success(f"Pipeline complete → {path}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auv_pipeline",
        description="AUV vision pipeline: underwater augmentation → training → pruning → TensorRT",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"auv_pipeline {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<stage>")

    p = sub.add_parser("presets", help="list training presets and augment profiles")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("check", help="dataset sanity check")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("augment", help="stage 1: preview an underwater augmentation profile")
    p.add_argument("--profile", default="physics")
    p.add_argument("--images", nargs="+", required=True, help="image files or directories")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--size", type=int, default=320, help="tile size (px)")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output image (default: runs/augment/<profile>.jpg)")
    p.add_argument("--show", action="store_true", help="also open a window")
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("train", help="stage 2: train with a preset + underwater augmentation")
    p.add_argument("--data", help="data.yaml")
    p.add_argument("--preset", default="yolo26n", help="name in configs/train/ or a YAML path")
    p.add_argument("--augment", help="augment profile (default: from the preset)")
    p.add_argument("--name", help="run name (default: from the preset)")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--resume", metavar="LAST_PT", help="resume an interrupted run from weights/last.pt")
    p.add_argument("--no-test", action="store_true", help="skip the held-out test-split evaluation")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="check dataset + print config, don't train")
    _train_overrides(p)
    _export_args(p, None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("prune", help="stage 3: structured pruning + recovery fine-tune")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--ratio", type=float, default=0.30, help="channels removed per round (default 0.30)")
    p.add_argument("--rounds", type=int, default=1, help=">1 for iterative pruning (e.g. yolo26m → nano)")
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--lr0", type=float, help="fine-tune LR (default 1e-4, or 5e-5 when iterative)")
    p.add_argument("--augment", help="augment profile for fine-tuning (default: Ultralytics stock)")
    p.add_argument("--round-to", type=int, default=8, help="round channel counts to a multiple of N")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device")
    p.add_argument("--project", default="runs/prune")
    p.add_argument("--name", default="pruned")
    _export_args(p, ["onnx", "engine"])
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("export", help="stage 4: ONNX / TensorRT export")
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device")
    p.add_argument("--data", help="data.yaml (INT8 calibration)")
    _export_args(p, ["onnx", "engine"])
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("benchmark", help="stage 5: compare models (mAP / size / latency)")
    p.add_argument("--models", nargs="+", required=True, help="first model is the reference")
    p.add_argument("--labels", nargs="+", help="display names, one per model")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.60)
    p.add_argument("--save", help="write results to .md or .json")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("report", help="summarise finished run directories")
    p.add_argument("runs", nargs="+")
    p.add_argument("--markdown", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", help="all stages end to end")
    p.add_argument("--data", required=True)
    p.add_argument("--preset", default="yolo26n")
    p.add_argument("--augment")
    p.add_argument("--project", default="runs/pipeline")
    p.add_argument("--prune-ratio", type=float, default=0.30)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--finetune-epochs", type=int, default=25)
    p.add_argument("--formats", nargs="+", metavar="FMT")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--skip", nargs="+", default=[], choices=("prune", "export", "benchmark"))
    p.add_argument("--dry-run", action="store_true")
    _train_overrides(p)
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        ui.error(str(exc).strip("'\""))
        return 2


if __name__ == "__main__":
    sys.exit(main())

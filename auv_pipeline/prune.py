"""Stage 3 — structured channel pruning (Torch-Pruning, BN-scale importance) + recovery fine-tune.

Structured pruning removes whole channels, so the network is genuinely smaller and
faster on TensorRT (which has no use for unstructured sparsity).

    rounds=1  single-shot prune + fine-tune        (e.g. yolo26n, 30 %)
    rounds>1  iterative prune + fine-tune per round (e.g. yolo26m -> nano-size, 2 x 40 %)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from . import ui
from .augment import build_transforms, get_profile, hsv_overrides

# The detection head's output shape must not change.
IGNORE_CLASSES = {"Detect", "Segment", "Pose", "OBB", "RTDETRDecoder", "v10Detect"}


@dataclass
class RoundResult:
    round: int
    ratio: float
    params: int
    macs: float
    weights: Path


def _device(spec: str | None):
    import torch

    if spec is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(f"cuda:{spec}" if str(spec).isdigit() else spec)


def count_params_macs(model_nn, imgsz: int, device) -> tuple[int, float]:
    import torch
    import torch_pruning as tp

    macs, params = tp.utils.count_ops_and_params(model_nn, torch.randn(1, 3, imgsz, imgsz).to(device))
    return int(params), float(macs)


def prune_once(model_nn, ratio: float, imgsz: int, device, round_to: int = 8) -> None:
    """Prune ``ratio`` of channels in place, keeping the detection head intact."""
    import torch
    import torch_pruning as tp

    for p in model_nn.parameters():  # dependency tracing needs autograd
        p.requires_grad_(True)
    pruner = tp.pruner.MetaPruner(
        model=model_nn,
        example_inputs=torch.randn(1, 3, imgsz, imgsz).to(device),
        importance=tp.importance.BNScaleImportance(),
        pruning_ratio=ratio,
        ignored_layers=[m for m in model_nn.modules() if m.__class__.__name__ in IGNORE_CLASSES],
        round_to=round_to,  # channel counts as multiples of 8 -> Tensor Core friendly
    )
    for group in pruner.step(interactive=True):
        group.prune()


def save_pruned_checkpoint(model_nn, path: Path) -> None:
    """Ultralytics-format checkpoint of the pruned (not yet fine-tuned) model."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": -1, "best_fitness": None, "model": copy.deepcopy(model_nn).float().cpu(),
            "ema": None, "updates": 0, "optimizer": None, "train_args": {}, "date": None, "version": None,
        },
        path,
    )


def pruned_trainer(pruned_nn):
    """A DetectionTrainer that fine-tunes ``pruned_nn`` itself.

    Ultralytics' ``Model.train()`` calls ``trainer.get_model(cfg=model.yaml, weights=model)``,
    which builds a *fresh full-size* network from the original YAML and copies across
    only the weights whose shapes still match — silently undoing the pruning. Returning
    the pruned module here keeps the smaller architecture through fine-tuning and into
    best.pt.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class PrunedDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pruned_nn

    return PrunedDetectionTrainer


def prune(
    weights: Path,
    data: str,
    *,
    ratio: float = 0.30,
    rounds: int = 1,
    finetune_epochs: int = 25,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    lr0: float | None = None,
    augment: str | None = None,
    round_to: int = 8,
    project: str = "runs/prune",
    name: str = "pruned",
) -> list[RoundResult]:
    from ultralytics import YOLO

    dev = _device(device)
    project_dir = Path(project).resolve()
    lr0 = lr0 if lr0 is not None else (1e-4 if rounds == 1 else 5e-5)  # well below the training LR

    finetune_args: dict = {}
    if augment:
        profile = get_profile(augment)
        finetune_args.update(hsv_overrides(profile))
        transforms = build_transforms(profile)
        if transforms is not None:
            finetune_args["augmentations"] = transforms

    yolo = YOLO(str(weights))
    base_params, base_macs = count_params_macs(yolo.model.eval().to(dev), imgsz, dev)
    results = [RoundResult(0, 0.0, base_params, base_macs, Path(weights))]
    ui.info(f"Base model: {base_params / 1e6:.2f} M params, {base_macs / 1e9:.2f} GMACs")

    current = Path(weights)
    for rnd in range(1, rounds + 1):
        ui.section(f"Prune round {rnd}/{rounds} — {ratio:.0%} of channels")
        yolo = YOLO(str(current))
        model_nn = yolo.model.eval().to(dev)
        before, _ = count_params_macs(model_nn, imgsz, dev)
        prune_once(model_nn, ratio, imgsz, dev, round_to)
        params, macs = count_params_macs(model_nn, imgsz, dev)
        ui.info(f"Pruned: {params / 1e6:.2f} M params ({1 - params / before:.1%} this round), {macs / 1e9:.2f} GMACs")
        save_pruned_checkpoint(model_nn, project_dir / f"{name}_r{rnd}_init.pt")

        ui.info(f"Fine-tuning {finetune_epochs} epochs (AdamW, lr0={lr0})")
        model_nn.cpu().float()
        yolo.train(
            trainer=pruned_trainer(model_nn),
            data=data, epochs=finetune_epochs, imgsz=imgsz, batch=batch,
            device=device, amp=True,
            # explicit optimizer: with Ultralytics' default 'auto', lr0 is ignored
            optimizer="AdamW", lr0=lr0, lrf=1e-3, warmup_epochs=1, patience=10,
            plots=True, project=str(project_dir), name=f"{name}_r{rnd}", exist_ok=False,
            **finetune_args,
        )
        current = Path(yolo.trainer.best if yolo.trainer.best.exists() else yolo.trainer.last)
        final_params, final_macs = count_params_macs(YOLO(str(current)).model.eval().to(dev), imgsz, dev)
        if final_params > params * 1.01:
            ui.warn(f"Fine-tuned model grew back to {final_params / 1e6:.2f} M params — pruning was lost")
        results.append(RoundResult(rnd, ratio, final_params, final_macs, current))

    ui.table(
        "Pruning summary",
        ("Round", "Params (M)", "GMACs", "vs base", "Weights"),
        [
            (r.round or "base", f"{r.params / 1e6:.2f}", f"{r.macs / 1e9:.2f}",
             f"{r.params / base_params - 1:+.1%}" if r.round else "—", r.weights)
            for r in results
        ],
    )
    return results

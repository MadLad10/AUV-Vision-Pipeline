# YOLO26n Training & Pruning Pipeline
### Jetson Orin Nano 8GB — Real-Time Object Detection

---

## Why YOLO26n?

| Factor | Reasoning |
|---|---|
| **Hardware** | Jetson Orin Nano 8GB has ~40 TOPS (INT8). YOLO26s runs, but headroom matters for tracking, preprocessing, and other pipeline stages running concurrently. |
| **Real-time** | YOLO26n hits 30+ FPS comfortably on Orin Nano at 640 px with TensorRT FP16. YOLO26s drops to 15–20 FPS — risky for strict real-time requirements. |
| **Accuracy trade-off** | On custom datasets with sufficient data, YOLO26n closes most of the gap to YOLO26s after proper training. Accuracy loss is typically 2–4 mAP points — acceptable for most applications. |
| **Post-pruning** | Pruning YOLO26s down to YOLO26n-size adds complexity. Starting with YOLO26n and pruning it 30% further yields an even lighter model without the added difficulty. |
| **Memory** | 8 GB is shared between GPU and system RAM. YOLO26n leaves room for the rest of your application stack. |

**When to reconsider:** If YOLO26n mAP is too low for your use case after training, switch to YOLO26s with aggressive pruning + INT8 quantization.

> **Model string note:** This pipeline uses `yolo26n.pt` (Ultralytics naming convention).
> Verify the exact string in the [Ultralytics docs](https://docs.ultralytics.com) before running.
> If YOLO26 is not yet listed, substitute the latest available nano variant (e.g. `yolo12n.pt`).

---

## Pipeline Overview

```
[CVAT annotation] → split_dataset.py → train_yolo26.py → prune_yolo26.py → benchmark.py → Jetson deploy
```

---

## Step 0 — Prerequisites

### Training machine (desktop / cloud GPU)

```bash
# 1. Clone / enter this directory
cd yolo26_pipeline

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Step 1 — Export from CVAT

1. Open your CVAT project → **Export dataset**
2. Format: **YOLO 1.1** (produces `obj_train_data/` with images + `.txt` labels and `obj.names`)
3. Extract the zip; note the path — this is your `--input` for the next step.

---

## Step 2 — Split the Dataset

```bash
python split_dataset.py \
    --input  /path/to/cvat_export \
    --output /path/to/dataset
```

Output structure:
```
dataset/
├── images/
│   ├── train/   # 85 %
│   ├── val/     #  5 %
│   └── test/    # 10 %
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

**Verify** `data.yaml` — especially `names:` (class list). Edit manually if auto-detection missed any class.

Key options:
```
--train 0.85   # default
--val   0.05   # default
--test  0.10   # default
--seed  42     # change for a different shuffle
```

---

## Step 3 — Train YOLO26n

```bash
python train_yolo26.py \
    --data /path/to/dataset/data.yaml
```

Results land in `runs/train/yolo26n_custom/`.

Common overrides:
```
--imgsz 416         # reduce image size for faster training / smaller model
--batch 32          # increase if you have a large GPU
--epochs 150        # more epochs for small datasets
--device 0,1        # multi-GPU
```

TensorRT export runs automatically after training. If you don't have TensorRT on the
training machine, skip it and run the export on the Jetson (see Step 6).

```bash
# Skip TensorRT on training machine, export only ONNX
python train_yolo26.py --data /path/to/data.yaml --no-export-tensorrt
```

---

## Step 4 — Prune the Model

```bash
python prune_yolo26.py \
    --weights runs/train/yolo26n_custom/weights/best.pt \
    --data    /path/to/dataset/data.yaml
```

This runs three sub-steps automatically:
1. **Prune** — removes 30% of channels (structured, not sparse)
2. **Fine-tune** — 25 recovery epochs at a lower LR
3. **Re-export** — ONNX + TensorRT

Key options:
```
--pruning-ratio   0.30   # start here; try 0.40 if FPS is still not enough
--finetune-epochs 25     # increase to 40 if mAP drops more than 3 points
```

Outputs land in:
```
runs/prune/yolo26n_pruned_ft/weights/best.pt
runs/prune/yolo26n_pruned_ft/weights/best.onnx
runs/prune/yolo26n_pruned_ft/weights/best.engine   (if TensorRT available)
```

---

## Step 5 — Benchmark

```bash
python benchmark.py \
    --base   runs/train/yolo26n_custom/weights/best.pt \
    --pruned runs/prune/yolo26n_pruned_ft/weights/best.pt \
    --data   /path/to/dataset/data.yaml
```

Expected output example:
```
==============================================================
  COMPARISON SUMMARY
==============================================================
  Metric               Base       Pruned        Delta
  ----------------------------------------------------------
  mAP50               0.847        0.831      -1.9% ▼
  mAP50-95            0.612        0.594      -2.9% ▼
  Params (M)           3.20         2.24      -30.0% ▼
  MACs (G)             8.10         5.67      -30.0% ▼
  Size (MB)            6.5          4.6       -29.2% ▼
  Latency (ms)        12.40         8.60      -30.6% ▼
  FPS                  80.6        116.3      +44.3% ▲
==============================================================
```

If mAP drop > 4 points: increase `--finetune-epochs` and re-run pruning.
If FPS gain < 20%: increase `--pruning-ratio` to 0.40.

---

## Step 6 — Jetson Orin Nano Deployment

### 6a. Flash JetPack

Download **JetPack 6.x** from NVIDIA's [SDK Manager](https://developer.nvidia.com/sdk-manager).
Flash to the Jetson Orin Nano following the official guide.

### 6b. Install PyTorch for Jetson

Do **not** use standard PyPI wheels — they lack Jetson CUDA support.

```bash
# On the Jetson:
sudo apt-get update && sudo apt-get install -y python3-pip

# Install NVIDIA's Jetson PyTorch wheel (check for the latest at):
# https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson/jetson-orin/
pip3 install --no-cache \
    https://developer.download.nvidia.com/compute/redist/jp/v60/<torch_wheel_for_jp6>.whl

pip3 install torchvision ultralytics
```

### 6c. TensorRT Export on Jetson

If you skipped TensorRT export on the training machine:

```bash
# Copy best.pt to Jetson, then:
python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine', imgsz=640, half=True, device=0)
"
```

This produces `best.engine` — the file you use for real-time inference.

### 6d. Real-Time Inference

```python
from ultralytics import YOLO

model = YOLO("best.engine")   # TensorRT engine, loaded once

for frame in camera_stream():
    results = model(frame, imgsz=640, conf=0.4, iou=0.5)
    annotated = results[0].plot()
    display(annotated)
```

For maximum FPS:
- Use `imgsz=416` or `imgsz=320` if 640 is still too slow after pruning.
- Enable INT8 quantization in the export step for another ~2× speedup (requires calibration data).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No valid image-label pairs found` | Check that CVAT exported `.txt` files alongside images, or in a sibling `labels/` directory. |
| `yolo26n.pt` download fails | Verify the model string in Ultralytics docs. Substitute with the latest nano variant if needed. |
| mAP drops > 5 pts after pruning | Increase `--finetune-epochs` to 40 and lower `--pruning-ratio` to 0.20. |
| TensorRT export fails on training machine | Run export on Jetson directly (Step 6c). |
| OOM during training | Halve `--batch`; enable `--imgsz 416`. |
| Pruning crashes with shape mismatch | Ensure `torch-pruning>=1.4.0`; confirm detection head is in `IGNORE_CLASSES`. |

---

## File Summary

| File | Purpose |
|---|---|
| `split_dataset.py` | 85/5/10 split + `data.yaml` generation |
| `train_yolo26.py` | Training + ONNX/TensorRT export |
| `prune_yolo26.py` | Structured channel pruning + fine-tune + re-export |
| `benchmark.py` | Side-by-side mAP / latency / FPS comparison |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | This document |

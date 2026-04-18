# Bounding Box Flickering — Fixes & Additions

> Append this section to the main README after "Step 6 — Jetson Orin Nano Deployment"
> or keep it as a standalone troubleshooting supplement.

---

## Why the Boxes Flicker

Single-frame detectors like YOLO26n process each frame independently. Between
consecutive frames, small changes in lighting, motion blur, JPEG/H.264 noise,
or (underwater) turbidity and scattering shift predictions by a few pixels and
occasionally drop detections entirely. The result is visible jitter and
blink on/off behaviour on the output video.

Flickering is **not** a bug in YOLO26 — it's inherent to per-frame detection
and affects every YOLO version. The fixes below are stackable, ordered from
highest impact to lowest.

---

## Fix Layer 1 — Use a Tracker Instead of Raw Prediction

Switch from `model.predict()` to `model.track()`. Ultralytics ships built-in
BoT-SORT and ByteTrack support. The tracker maintains a Kalman filter per
object, so missed or noisy detections no longer cause the box to disappear
or jump.

Replace Step 6d inference loop with:

```python
from ultralytics import YOLO

model = YOLO("best.engine")   # pruned TensorRT engine

for frame in camera_stream():
    results = model.track(
        frame,
        persist=True,                 # REQUIRED — keeps Kalman state
        tracker="bytetrack.yaml",     # or "botsort.yaml"
        conf=0.25,                    # lower than 0.4 — ByteTrack uses low-conf boxes
        iou=0.5,
        imgsz=640,
        verbose=False,
    )
    annotated = results[0].plot()
    display(annotated)
```

**Key parameter changes:**

| Parameter | Old | New | Why |
|---|---|---|---|
| `conf` | 0.4 | 0.25 | ByteTrack's second-pass associates low-confidence detections. Higher threshold throws away boxes the tracker would have used. |
| `persist` | (not set) | `True` | Without this, tracker state resets every frame — zero benefit. |

**Choosing a tracker:**
- **ByteTrack** — faster, simpler. Start here.
- **BoT-SORT** — slower, but has camera-motion compensation + ReID. Use for
  moving-camera scenarios (ROV, AUV, handheld).

### Tune `bytetrack.yaml`

Located at `ultralytics/cfg/trackers/bytetrack.yaml`:

```yaml
track_high_thresh: 0.25    # first-association threshold
track_low_thresh:  0.1     # second-pass threshold, keep low
new_track_thresh:  0.25    # don't spawn duplicate IDs
track_buffer:      60      # hold tracks 60 frames (2s @ 30fps) through occlusion
match_thresh:      0.8
```

Increasing `track_buffer` is what eliminates the on/off blink — the box stays
drawn for 2 seconds even if the detector briefly loses the object.

---

## Fix Layer 2 — EMA Smoothing on Box Coordinates

Even with a tracker, box corners still wiggle ±3–5 px frame-to-frame.
Apply an exponential moving average per track ID:

```python
from collections import defaultdict
import numpy as np

class BoxSmoother:
    """EMA on xyxy coordinates per track ID."""
    def __init__(self, alpha=0.6):
        self.alpha = alpha       # higher = more responsive, lower = smoother
        self.state = {}          # id -> np.array([x1, y1, x2, y2])

    def update(self, track_id, box_xyxy):
        box = np.asarray(box_xyxy, dtype=np.float32)
        if track_id not in self.state:
            self.state[track_id] = box
        else:
            self.state[track_id] = (
                self.alpha * box + (1 - self.alpha) * self.state[track_id]
            )
        return self.state[track_id]

    def prune(self, active_ids):
        self.state = {k: v for k, v in self.state.items() if k in active_ids}


smoother = BoxSmoother(alpha=0.6)

for frame in camera_stream():
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=0.25, iou=0.5, verbose=False)
    r = results[0]
    if r.boxes.id is None:
        continue
    ids      = r.boxes.id.int().cpu().tolist()
    boxes    = r.boxes.xyxy.cpu().numpy()
    smoothed = [smoother.update(i, b) for i, b in zip(ids, boxes)]
    smoother.prune(set(ids))
    # draw `smoothed` instead of raw boxes
```

Tuning `alpha`:

| `alpha` | Behavior |
|---|---|
| 0.4 | Very smooth, slight lag on fast motion |
| 0.6 | **Default** — good balance |
| 0.75 | Responsive, minor residual jitter |

### Optional: adaptive alpha

For mixed static/fast-moving targets, scale `alpha` by how far the box moved
since the last frame — small movement → stronger smoothing, large movement →
weaker smoothing (so fast targets don't lag).

---

## Fix Layer 3 — Underwater-Specific Preprocessing

Underwater flicker has an extra cause: low contrast, color shift, and
scattering noise. Two cheap preprocessing steps that stabilize the backbone:

### 3a. CLAHE (contrast enhancement)

```python
import cv2

def underwater_preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

Cost: ~1 ms per frame on Jetson Orin Nano.

> **Important:** if you apply CLAHE at inference, you must also apply it
> during training (re-run Step 3 on CLAHE-preprocessed images). Otherwise
> the train/inference distribution mismatch will hurt mAP.

### 3b. Mild Gaussian blur

Counterintuitive but effective — suppresses high-frequency noise that
destabilizes feature maps:

```python
frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=2.0)
```

Apply **before** inference. Measure mAP on your val set before and after —
keep it only if mAP is stable or improves.

---

## Recommended Application Order

1. **Start with Layer 1** (tracker + lowered `conf` + `track_buffer=60`).
   Fixes ~80% of flicker in ~30 minutes of work.
2. If boxes still wiggle → **add Layer 2** (`BoxSmoother`, 10 more minutes).
3. If underwater and still unstable → **add Layer 3a** (CLAHE), then retrain
   Step 3 with CLAHE applied to the training set.
4. Only if all the above fail → retrain with stronger video-aware augmentation
   (brightness/contrast/Gaussian-noise jitter per image) so the model itself
   becomes more invariant frame-to-frame.

---

## Pruning and Temporal Stability

Pruning at 30% can degrade temporal stability more than the mAP number
suggests — the remaining channels become more sensitive to input
perturbations, which amplifies frame-to-frame jitter.

If the above layers don't fully fix flicker, benchmark the **unpruned** model
as a control. If unpruned is noticeably more stable:

```
--pruning-ratio   0.20   # down from 0.30
--finetune-epochs 35     # up from 25 for better recovery
```

---

## References

| Paper / Repo | Relevance |
|---|---|
| [ByteTrack (Zhang et al., 2021)](https://arxiv.org/abs/2110.06864) | Core tracker paper. Explains why lowering `conf` helps. |
| [de Langis et al., "An Analysis of Deep Object Detectors For Diver Detection"](https://arxiv.org/abs/2012.05701) | Underwater-specific. Shows highest-mAP model is not the most temporally stable. Public 105k-frame dataset. |
| [DyAqua-YOLO (2025)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1678417/full) | Underwater-specific architecture that addresses box misalignment / localization drift across YOLO versions. |
| [Towards Stable 3D Object Detection (Lu et al., 2024)](https://arxiv.org/abs/2407.04305) | Introduces Prediction Consistency Learning (PCL) — training-time strategy for detection stability. Adaptable to 2D YOLO. |
| [BoxMOT](https://github.com/mikel-brostrom/boxmot) | Multi-tracker library (ByteTrack, BoT-SORT, OC-SORT, StrongSORT, DeepOCSORT). |
| [NVIDIA DeepStream + NvDCF](https://forums.developer.nvidia.com/t/bounding-box-flickering-in-yolov11-nvdcf-tracking-deepstream-6-3-on-jetson-xavier-agx/324831) | Jetson-native hardware-accelerated tracker path. |

---

## Troubleshooting (Flicker-Specific)

| Symptom | Fix |
|---|---|
| Boxes blink on/off every few seconds | Increase `track_buffer` to 60–90; lower `conf` to 0.2. |
| Box corners jitter ±3–5 px | Add `BoxSmoother` with `alpha=0.6`. |
| Box lags behind fast-moving target | Raise `alpha` to 0.75 or use adaptive alpha. |
| ID keeps switching on same object | Switch tracker to BoT-SORT; enable `with_reid: True`. |
| Flicker only in murky / low-light frames | Add CLAHE preprocessing and retrain. |
| Stable before pruning, flickery after | Lower `--pruning-ratio` to 0.20; raise `--finetune-epochs` to 35. |
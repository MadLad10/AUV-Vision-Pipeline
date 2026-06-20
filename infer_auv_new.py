#!/usr/bin/env python3
from ultralytics import YOLO
import cv2
import numpy as np

# ── Change these ──────────────────────────────────────────────────────────────
WEIGHTS = r"/home/yaad/Downloads/gate_updated_june4.pt"
SOURCE  = r"/home/yaad/Downloads/9may_gate_torpedo.webm"
CONF    = 0.25
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = [(0,255,0),(255,80,80),(0,200,255),(255,0,200),(0,165,255)]

class BoxSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = {}

    def update(self, track_id, box):
        box = np.asarray(box, dtype=np.float32)
        if track_id not in self.state:
            self.state[track_id] = box
        else:
            self.state[track_id] = self.alpha * box + (1 - self.alpha) * self.state[track_id]
        return self.state[track_id]

    def prune(self, active_ids):
        self.state = {k: v for k, v in self.state.items() if k in active_ids}


model    = YOLO(WEIGHTS)
smoother = BoxSmoother()
cap      = cv2.VideoCapture(SOURCE)

print("Running — press q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=CONF, verbose=False, imgsz=640, half=True)[0]
    boxes   = results.boxes
    out     = frame.copy()

    if boxes.id is not None:
        smoother.prune(set(boxes.id.int().cpu().tolist()))

    for i, box in enumerate(boxes):
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        if boxes.id is not None:
            tid = int(boxes.id[i])
            x1, y1, x2, y2 = map(int, smoother.update(tid, box.xyxy[0].tolist()))
            label = f"{results.names[cls]} {conf:.2f} id{tid}"
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{results.names[cls]} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.imshow("AUV Inference", out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")

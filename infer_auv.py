#!/usr/bin/env python3
"""
AUV inference — single-panel display with optional inpainting.

Modes:
  --mode raw       : run YOLO on the raw frame only (default)
  --mode inpaint   : run YOLO on the inpainted frame only
  --mode compare   : side-by-side RAW vs INPAINTED

Controls:
    q  — quit
    s  — save a snapshot of the current frame as a PNG

Usage:
    python infer_auv.py --source /path/to/dive.mp4 --weights best.pt
    python infer_auv.py --source dive.mp4 --weights best.pt --mode inpaint
    python infer_auv.py --source dive.mp4 --weights best.pt --mode compare --save out.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AUV YOLO inference")
    p.add_argument("--source",  required=True, help="Video path or camera index")
    p.add_argument("--weights", required=True, help="Model weights: .pt or .engine")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--mode",    choices=["raw", "inpaint", "compare"], default="raw")
    p.add_argument("--sat-thresh",     type=int, default=50)
    p.add_argument("--dark-thresh",    type=int, default=40)
    p.add_argument("--bright-thresh",  type=int, default=230)
    p.add_argument("--dilate-ksize",   type=int, default=5)
    p.add_argument("--inpaint-radius", type=int, default=3)
    p.add_argument("--save", default=None, help="Output video path (e.g. out.mp4)")
    p.add_argument("--fps",  type=float, default=None, help="Override output FPS (useful when source FPS metadata is wrong)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

def build_degraded_mask(frame_bgr, sat_thresh, dark_thresh, bright_thresh, dilate_ksize):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s < sat_thresh) | (v < dark_thresh) | (v > bright_thresh)).astype(np.uint8) * 255
    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask   = cv2.dilate(mask, kernel, iterations=1)
    return mask


def inpaint_frame(frame_bgr, mask, inpaint_radius):
    if not mask.any():
        return frame_bgr.copy()
    return cv2.inpaint(frame_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

LABEL_H   = 30
FONT      = cv2.FONT_HERSHEY_SIMPLEX

CLASS_COLORS = [
    (0,   255,   0),   # green
    (255,  80,  80),   # blue
    (0,   200, 255),   # yellow
    (255,   0, 200),   # pink
    (0,   165, 255),   # orange
    (180,   0, 255),   # purple
    (255, 255,   0),   # cyan
    (0,   255, 180),   # mint
]

def class_color(cls: int):
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


class BoxSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = {}

    def update(self, track_id, box_xyxy):
        box = np.asarray(box_xyxy, dtype=np.float32)
        if track_id not in self.state:
            self.state[track_id] = box
        else:
            self.state[track_id] = self.alpha * box + (1 - self.alpha) * self.state[track_id]
        return self.state[track_id]

    def prune(self, active_ids):
        self.state = {k: v for k, v in self.state.items() if k in active_ids}


def draw_detections(frame, results, smoother=None):
    out = frame.copy()
    r   = results[0]
    has_ids = r.boxes.id is not None

    if has_ids and smoother is not None:
        ids  = r.boxes.id.int().cpu().tolist()
        smoother.prune(set(ids))

    for i, box in enumerate(r.boxes):
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

        if has_ids and smoother is not None:
            track_id = int(r.boxes.id[i])
            x1, y1, x2, y2 = map(int, smoother.update(track_id, box.xyxy[0].tolist()))
            label = f"{r.names[cls]} {conf:.2f} id{track_id}"
        else:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{r.names[cls]} {conf:.2f}"

        color = class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 8, 18)), FONT, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(y1 - 8, 18)), FONT, 0.9, color, 2, cv2.LINE_AA)
    return out


def make_panel(frame, label, fps, color):
    h, w   = frame.shape[:2]
    panel  = np.zeros((h + LABEL_H, w, 3), dtype=np.uint8)
    panel[LABEL_H:, :] = frame
    panel[:LABEL_H, :] = (30, 30, 30)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.75, 2)
    cv2.putText(panel, label, ((w - tw) // 2, (LABEL_H + th) // 2), FONT, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS: {fps:.1f}", (8, LABEL_H + 22), FONT, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSMeter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self, elapsed_s):
        if elapsed_s > 0:
            self._times.append(elapsed_s)

    @property
    def fps(self):
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    from ultralytics import YOLO

    print(f"Loading model : {args.weights}")
    model = YOLO(args.weights)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_out = args.fps if args.fps else (fps_src if fps_src <= 120 else 30.0)
    w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source : {args.source}  ({w_src}×{h_src} @ {fps_src:.1f} FPS)  mode={args.mode}")
    if fps_src != fps_out:
        print(f"Output FPS clamped to {fps_out:.1f} (source reported {fps_src:.1f})")

    compare = args.mode == "compare"
    out_w   = w_src * 2 if compare else w_src
    out_h   = h_src + LABEL_H

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (out_w, out_h))
        print(f"Saving to : {args.save}")

    fps_meter    = FPSMeter()
    fps_meter2   = FPSMeter()
    smoother     = BoxSmoother(alpha=0.6)
    smoother2    = BoxSmoother(alpha=0.6)
    snapshot_idx = 0

    display = True
    if args.save:
        try:
            cv2.namedWindow("AUV Inference", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except cv2.error:
            display = False
    else:
        cv2.namedWindow("AUV Inference", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("\nRunning — press  q  to quit,  s  to snapshot\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mode in ("inpaint", "compare"):
            mask     = build_degraded_mask(frame, args.sat_thresh, args.dark_thresh,
                                           args.bright_thresh, args.dilate_ksize)
            restored = inpaint_frame(frame, mask, args.inpaint_radius)

        if args.mode == "raw":
            t0  = time.perf_counter()
            res = model.track(frame, imgsz=args.imgsz, conf=args.conf, persist=True, verbose=False, tracker="bytetrack.yaml")
            fps_meter.tick(time.perf_counter() - t0)
            combined = make_panel(draw_detections(frame, res, smoother), "RAW", fps_meter.fps, (100, 210, 255))

        elif args.mode == "inpaint":
            t0  = time.perf_counter()
            res = model.track(restored, imgsz=args.imgsz, conf=args.conf, persist=True, verbose=False, tracker="bytetrack.yaml")
            fps_meter.tick(time.perf_counter() - t0)
            combined = make_panel(draw_detections(restored, res, smoother), "INPAINTED", fps_meter.fps, (100, 255, 160))

        else:  # compare
            t0  = time.perf_counter()
            r_raw = model.track(frame,    imgsz=args.imgsz, conf=args.conf, persist=True, verbose=False)
            fps_meter.tick(time.perf_counter() - t0)

            t0  = time.perf_counter()
            r_inp = model.track(restored, imgsz=args.imgsz, conf=args.conf, persist=True, verbose=False, tracker="bytetrack.yaml")
            fps_meter2.tick(time.perf_counter() - t0)

            p_raw = make_panel(draw_detections(frame,    r_raw, smoother),  "RAW",       fps_meter.fps,  (100, 210, 255))
            p_inp = make_panel(draw_detections(restored, r_inp, smoother2), "INPAINTED", fps_meter2.fps, (100, 255, 160))
            combined = cv2.hconcat([p_raw, p_inp])

        if display:
            cv2.imshow("AUV Inference", combined)
        if writer:
            writer.write(combined)

        if display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap = f"snapshot_{snapshot_idx:04d}.png"
                cv2.imwrite(snap, combined)
                print(f"  Snapshot saved: {snap}")
                snapshot_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\nVideo saved: {args.save}")
    if display:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

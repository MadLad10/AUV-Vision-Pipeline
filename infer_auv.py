#!/usr/bin/env python3
"""
AUV side-by-side inference: RAW vs cv2.inpaint-restored frames.

Runs YOLO independently on both the raw and inpainted frame for each video
frame, then displays them side by side in a single resizable window.

Use this script to visually decide whether inpainting improves detection for
your specific dive conditions before committing to one pipeline.

FOR DEPLOYMENT: once you have chosen a pipeline (raw or inpainted), run
inference with only that branch — running YOLO twice per frame is intentional
here for comparison purposes only and will roughly halve your FPS.

Controls:
    q  — quit
    s  — save a snapshot of the current side-by-side frame as a PNG

Usage:
    # Video file
    python infer_auv.py --source /path/to/dive.mp4 --weights best.pt

    # Live camera (index 0)
    python infer_auv.py --source 0 --weights best.engine

    # Tune mask thresholds for a specific dive
    python infer_auv.py --source dive.mp4 --weights best.pt \\
        --sat-thresh 40 --dark-thresh 30 --bright-thresh 220

    # Save the side-by-side output
    python infer_auv.py --source dive.mp4 --weights best.pt --save output.mp4
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
    p = argparse.ArgumentParser(
        description="Side-by-side AUV inference: RAW vs inpainted"
    )
    p.add_argument("--source",  required=True,
                   help="Video path or camera index (e.g. 0)")
    p.add_argument("--weights", required=True,
                   help="Model weights: .pt or .engine")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--imgsz",   type=int,   default=640)

    # Inpainting mask thresholds (HSV space, 0-255)
    p.add_argument("--sat-thresh",    type=int, default=50,
                   help="Pixels with S < this are 'degraded' (default: 50)")
    p.add_argument("--dark-thresh",   type=int, default=40,
                   help="Pixels with V < this are 'dark/degraded' (default: 40)")
    p.add_argument("--bright-thresh", type=int, default=230,
                   help="Pixels with V > this are 'backscatter' (default: 230)")
    p.add_argument("--dilate-ksize",  type=int, default=5,
                   help="Mask dilation kernel size in pixels (default: 5)")
    p.add_argument("--inpaint-radius", type=int, default=3,
                   help="cv2.inpaint radius (default: 3)")

    p.add_argument("--save", default=None,
                   help="Optional output path for side-by-side video (e.g. out.mp4)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

def build_degraded_mask(
    frame_bgr: np.ndarray,
    sat_thresh: int,
    dark_thresh: int,
    bright_thresh: int,
    dilate_ksize: int,
) -> np.ndarray:
    """
    Return a uint8 mask (255 = degraded, 0 = ok) for regions that are:
      - low saturation  (fog / backscatter desaturation)
      - very dark       (deep / shadowed regions)
      - very bright     (backscatter highlights)

    The mask is dilated so inpainting can blend smoothly at region edges.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s   = hsv[:, :, 1]
    v   = hsv[:, :, 2]

    mask = (
        (s < sat_thresh)    |
        (v < dark_thresh)   |
        (v > bright_thresh)
    ).astype(np.uint8) * 255

    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def inpaint_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int,
) -> np.ndarray:
    """Apply TELEA inpainting. If the mask is all-zero, returns the original frame."""
    if not mask.any():
        return frame_bgr.copy()
    return cv2.inpaint(frame_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

LABEL_H    = 30   # px reserved at the top of each panel for the pipeline label
FPS_FONT   = cv2.FONT_HERSHEY_SIMPLEX
FPS_SCALE  = 0.65
FPS_THICK  = 2
LBL_SCALE  = 0.75
LBL_THICK  = 2


def draw_detections(frame: np.ndarray, results) -> np.ndarray:
    """Draw YOLO bounding boxes + labels onto a copy of frame."""
    out = frame.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = f"{results[0].names[cls]} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                    FPS_FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def make_panel(
    frame: np.ndarray,
    label: str,
    fps: float,
    label_color: tuple[int, int, int],
) -> np.ndarray:
    """Add a pipeline label bar at the top and FPS overlay."""
    h, w = frame.shape[:2]
    panel = np.zeros((h + LABEL_H, w, 3), dtype=np.uint8)
    panel[LABEL_H:, :] = frame

    # Header bar background
    panel[:LABEL_H, :] = (30, 30, 30)

    # Pipeline label (centred)
    (tw, th), _ = cv2.getTextSize(label, FPS_FONT, LBL_SCALE, LBL_THICK)
    cv2.putText(panel, label,
                ((w - tw) // 2, (LABEL_H + th) // 2),
                FPS_FONT, LBL_SCALE, label_color, LBL_THICK, cv2.LINE_AA)

    # FPS (top-left of image area)
    fps_str = f"FPS: {fps:.1f}"
    cv2.putText(panel, fps_str, (8, LABEL_H + 22),
                FPS_FONT, FPS_SCALE, (0, 220, 255), FPS_THICK, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Rolling FPS tracker
# ---------------------------------------------------------------------------

class FPSMeter:
    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, elapsed_s: float) -> None:
        if elapsed_s > 0:
            self._times.append(elapsed_s)

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def open_source(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def main() -> None:
    args = parse_args()

    # Lazy import so the script fails fast on bad args before loading torch
    from ultralytics import YOLO

    print(f"Loading model : {args.weights}")
    model = YOLO(args.weights)

    cap      = open_source(args.source)
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source        : {args.source}  ({w_src}×{h_src} @ {fps_src:.1f} FPS)")

    # Optional output writer (side-by-side dimensions = 2*w_src × h_src + LABEL_H)
    writer: cv2.VideoWriter | None = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        out_w   = w_src * 2
        out_h   = h_src + LABEL_H
        writer  = cv2.VideoWriter(str(out_path), fourcc, fps_src, (out_w, out_h))
        print(f"Saving to     : {out_path}")

    fps_raw      = FPSMeter()
    fps_inpainted = FPSMeter()
    snapshot_idx  = 0

    cv2.namedWindow("AUV Inference", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    print("\nRunning — press  q  to quit,  s  to save snapshot\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # -- Inpainting mask + restored frame --------------------------------
        mask      = build_degraded_mask(
            frame,
            sat_thresh=args.sat_thresh,
            dark_thresh=args.dark_thresh,
            bright_thresh=args.bright_thresh,
            dilate_ksize=args.dilate_ksize,
        )
        restored  = inpaint_frame(frame, mask, args.inpaint_radius)

        # -- RAW pipeline -----------------------------------------------------
        t0_raw    = time.perf_counter()
        raw_res   = model(frame,    imgsz=args.imgsz, conf=args.conf, verbose=False)
        fps_raw.tick(time.perf_counter() - t0_raw)
        raw_drawn = draw_detections(frame, raw_res)

        # -- INPAINTED pipeline -----------------------------------------------
        t0_inp    = time.perf_counter()
        inp_res   = model(restored, imgsz=args.imgsz, conf=args.conf, verbose=False)
        fps_inpainted.tick(time.perf_counter() - t0_inp)
        inp_drawn = draw_detections(restored, inp_res)

        # -- Compose side-by-side display -------------------------------------
        panel_raw = make_panel(raw_drawn, "RAW",       fps_raw.fps,       (100, 210, 255))
        panel_inp = make_panel(inp_drawn, "INPAINTED",  fps_inpainted.fps, (100, 255, 160))

        combined  = cv2.hconcat([panel_raw, panel_inp])

        cv2.imshow("AUV Inference", combined)

        if writer is not None:
            writer.write(combined)

        # -- Key handling -----------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = f"snapshot_{snapshot_idx:04d}.png"
            cv2.imwrite(snap_path, combined)
            print(f"  Snapshot saved: {snap_path}")
            snapshot_idx += 1

    # -- Cleanup --------------------------------------------------------------
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nVideo saved: {args.save}")
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

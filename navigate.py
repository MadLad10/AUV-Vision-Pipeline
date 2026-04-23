#!/usr/bin/env python3
"""
AUV Gate Navigation — Visual Decision Overlay

Runs YOLO detection and displays a continuous navigation command on screen
based on the visual_navigation decision logic (err_x, aspect, fill, FSM).

States:
  SEARCHING      — no gate detected
  ALIGNING       — gate visible but off-center
  APPROACH       — gate centered, moving forward
  RECOVER_LEFT   — only right bar visible, move left
  RECOVER_RIGHT  — only left bar visible, move right
  OVERSHOOT      — single bar too close, back up
  PASS_THROUGH   — gate fills frame, commit

Usage:
    python navigate.py --weights best.pt --source video.mkv
    python navigate.py --weights best.pt --source 0          # live camera
    python navigate.py --weights best.pt --source video.mkv --save out.mp4
"""

import argparse
import csv
import math
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AUV Gate Navigation Overlay")
    p.add_argument("--weights",         required=True)
    p.add_argument("--source",          required=True)
    p.add_argument("--conf",            type=float, default=0.50)
    p.add_argument("--imgsz",           type=int,   default=640)
    p.add_argument("--device",          default="0")
    p.add_argument("--save",            default=None,  help="Output video path")
    p.add_argument("--log",             default=None,  help="CSV log path")
    # Decision thresholds
    p.add_argument("--err-center",      type=float, default=0.10)
    p.add_argument("--aspect-full",     type=float, default=1.30)
    p.add_argument("--aspect-bar",      type=float, default=0.50)
    p.add_argument("--fill-passthru",   type=float, default=0.60)
    p.add_argument("--fill-overshoot",  type=float, default=0.25)
    p.add_argument("--ema-alpha",       type=float, default=0.30)
    p.add_argument("--hysteresis",      type=int,   default=4)
    p.add_argument("--search-frames",   type=int,   default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colors & fonts
# ---------------------------------------------------------------------------

FONT     = cv2.FONT_HERSHEY_SIMPLEX
MONO     = cv2.FONT_HERSHEY_PLAIN

C_GREEN  = (0,   220,  60)
C_LGREEN = (0,   255, 120)
C_YELLOW = (0,   210, 255)
C_ORANGE = (0,   150, 255)
C_RED    = (0,    50, 255)
C_GRAY   = (140, 140, 140)
C_WHITE  = (230, 230, 230)
C_CYAN   = (255, 210,   0)
C_BLACK  = (0,     0,   0)

STATE_STYLE = {
    "SEARCHING":       ("SEARCHING",              C_GRAY),
    "ALIGNING_LEFT":   ("YAW LEFT",               C_YELLOW),
    "ALIGNING_RIGHT":  ("YAW RIGHT",              C_YELLOW),
    "APPROACH":        ("ALIGNED  →  FORWARD",    C_GREEN),
    "RECOVER_LEFT":    ("LATERAL LEFT",            C_ORANGE),
    "RECOVER_RIGHT":   ("LATERAL RIGHT",           C_ORANGE),
    "OVERSHOOT_LEFT":  ("BACK UP + LATERAL RIGHT", C_RED),
    "OVERSHOOT_RIGHT": ("BACK UP + LATERAL LEFT",  C_RED),
    "OVERSHOOT":       ("BACK UP",                C_RED),
    "PASS_THROUGH":    ("PASS THROUGH",           C_LGREEN),
}


# ---------------------------------------------------------------------------
# FPS meter
# ---------------------------------------------------------------------------

class FPSMeter:
    def __init__(self, window=30):
        self._t = deque(maxlen=window)

    def tick(self, dt):
        if dt > 0:
            self._t.append(dt)

    @property
    def fps(self):
        return 1.0 / (sum(self._t) / len(self._t)) if self._t else 0.0


# ---------------------------------------------------------------------------
# FSM with hysteresis
# ---------------------------------------------------------------------------

class NavigationFSM:
    def __init__(self, hysteresis: int, search_frames: int):
        self.k             = hysteresis
        self.search_frames = search_frames
        self.state         = "SEARCHING"
        self._candidate    = "SEARCHING"
        self._count        = 0
        self._no_det       = 0

    def update(self, candidate: str) -> str:
        if candidate == "SEARCHING":
            self._no_det += 1
            if self._no_det >= self.search_frames:
                self.state      = "SEARCHING"
                self._candidate = "SEARCHING"
                self._count     = 0
        else:
            self._no_det = 0
            if candidate == self._candidate:
                self._count += 1
                if self._count >= self.k:
                    self.state = candidate
            else:
                self._candidate = candidate
                self._count     = 1
        return self.state


# ---------------------------------------------------------------------------
# Per-frame features
# ---------------------------------------------------------------------------

def compute_features(x1, y1, x2, y2, iw, ih):
    bbox_cx = (x1 + x2) / 2
    bbox_w  = x2 - x1
    bbox_h  = y2 - y1
    err_x   = (bbox_cx - iw / 2) / (iw / 2)
    aspect  = bbox_w / max(bbox_h, 1)
    fill    = (bbox_w * bbox_h) / (iw * ih)
    return err_x, aspect, fill, int(bbox_cx)


def decide(err_x, aspect, fill, args) -> str:
    full_gate = aspect >= args.aspect_full
    bar_only  = aspect <= args.aspect_bar

    if full_gate:
        if fill >= args.fill_passthru:
            return "PASS_THROUGH"
        if abs(err_x) < args.err_center:
            return "APPROACH"
        return "ALIGNING_LEFT" if err_x < 0 else "ALIGNING_RIGHT"

    if bar_only:
        if fill >= args.fill_overshoot:
            # Too close — back up with lateral correction
            if abs(err_x) < args.err_center:
                return "OVERSHOOT"           # centered but too close
            return "OVERSHOOT_LEFT" if err_x < 0 else "OVERSHOOT_RIGHT"
        # Bar on left side of frame → right bar visible → move left
        return "RECOVER_LEFT" if err_x < 0 else "RECOVER_RIGHT"

    # Ambiguous aspect — treat like full gate logic
    if fill >= args.fill_passthru:
        return "PASS_THROUGH"
    if abs(err_x) < args.err_center:
        return "APPROACH"
    return "ALIGNING_LEFT" if err_x < 0 else "ALIGNING_RIGHT"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_banner(frame, state, h, w):
    label, color = STATE_STYLE.get(state, ("UNKNOWN", C_GRAY))
    cv2.rectangle(frame, (0, 0), (w, 55), C_BLACK, -1)
    cv2.rectangle(frame, (0, 0), (w, 55), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 1.2, 2)
    cv2.putText(frame, label, ((w - tw) // 2, 38),
                FONT, 1.2, color, 2, cv2.LINE_AA)


def draw_reference_lines(frame, bbox_cx, iw, ih, err_x):
    # Center line
    cv2.line(frame, (iw // 2, 60), (iw // 2, ih - 40), C_WHITE, 1, cv2.LINE_AA)

    # Dead-zone band (middle 20%)
    dz = int(iw * 0.10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (iw // 2 - dz, 60), (iw // 2 + dz, ih - 40), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Gate center line
    if abs(err_x) < 0.10:
        col = C_GREEN
    elif abs(err_x) < 0.30:
        col = C_YELLOW
    else:
        col = C_RED
    cv2.line(frame, (bbox_cx, 60), (bbox_cx, ih - 40), col, 2, cv2.LINE_AA)


def draw_direction_arrow(frame, state, iw, ih):
    mx, my = iw // 2, ih // 2
    arm    = 60

    arrows = {
        "ALIGNING_LEFT":   ((mx - arm, my),       (mx - arm * 3, my)),
        "ALIGNING_RIGHT":  ((mx + arm, my),        (mx + arm * 3, my)),
        "APPROACH":        ((mx, my + arm),         (mx, my - arm * 2)),
        "RECOVER_LEFT":    ((mx - arm, my),        (mx - arm * 3, my)),
        "RECOVER_RIGHT":   ((mx + arm, my),         (mx + arm * 3, my)),
        "OVERSHOOT":       ((mx, my - arm),         (mx, my + arm * 2)),
        "OVERSHOOT_LEFT":  ((mx, my - arm),         (mx, my + arm * 2)),
        "OVERSHOOT_RIGHT": ((mx, my - arm),         (mx, my + arm * 2)),
        "PASS_THROUGH":    ((mx, my + arm),         (mx, my - arm * 2)),
    }

    if state in arrows:
        _, color = STATE_STYLE[state]
        start, end = arrows[state]
        cv2.arrowedLine(frame, start, end, color, 4, cv2.LINE_AA, tipLength=0.3)
        # Extra lateral arrow for directional overshoot
        if state == "OVERSHOOT_LEFT":
            cv2.arrowedLine(frame, (mx + arm, my), (mx + arm * 3, my),
                            color, 4, cv2.LINE_AA, tipLength=0.3)
        elif state == "OVERSHOOT_RIGHT":
            cv2.arrowedLine(frame, (mx - arm, my), (mx - arm * 3, my),
                            color, 4, cv2.LINE_AA, tipLength=0.3)


def draw_debug_strip(frame, state, err_x, aspect, fill, conf, track_id, fps, ih, iw):
    txt = (f"state:{state}  yaw:{err_x:+.2f}  aspect:{aspect:.2f}"
           f"  fill:{fill:.2f}  conf:{conf:.2f}  id:{track_id}  fps:{fps:.1f}")
    cv2.rectangle(frame, (0, ih - 28), (iw, ih), C_BLACK, -1)
    cv2.putText(frame, txt, (6, ih - 8), MONO, 1.0, C_GRAY, 1, cv2.LINE_AA)


def draw_crosshair(frame, cx, cy):
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), C_CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), C_CYAN, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, C_CYAN, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    from ultralytics import YOLO

    print(f"Loading model : {args.weights}")
    model = YOLO(args.weights)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    iw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source : {args.source}  ({iw}×{ih} @ {fps_src:.1f} fps)")

    ffmpeg_proc = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_proc = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{iw}x{ih}", "-pix_fmt", "bgr24", "-r", str(fps_src),
            "-i", "pipe:",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            args.save,
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(f"Saving to : {args.save}")

    log_file = log_writer = None
    if args.log:
        log_file   = open(args.log, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp", "state", "err_x", "aspect",
                              "fill", "conf", "track_id"])
        print(f"Logging to : {args.log}")

    fsm       = NavigationFSM(args.hysteresis, args.search_frames)
    fps_meter = FPSMeter()
    err_smooth = 0.0
    snap_idx   = 0

    cv2.namedWindow("AUV Navigation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("\nRunning — q: quit  s: snapshot\n")

    while True:
        t0         = time.perf_counter()
        ok, frame  = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=args.conf,
                              imgsz=args.imgsz, device=args.device, verbose=False)

        r       = results[0]
        has_ids = r.boxes.id is not None

        # Pick primary gate detection (largest area)
        best       = None
        best_area  = 0
        best_conf  = 0.0
        best_tid   = -1

        for i, box in enumerate(r.boxes):
            cls  = int(box.cls[0])
            name = r.names[cls]
            if "gate" not in name.lower():
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best      = (x1, y1, x2, y2)
                best_area = area
                best_conf = float(box.conf[0])
                best_tid  = int(r.boxes.id[i]) if has_ids else -1

        # Also draw all boxes
        out = frame.copy()
        for i, box in enumerate(r.boxes):
            cls  = int(box.cls[0])
            name = r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            color = C_GREEN if "gate" in name.lower() else C_ORANGE
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)),
                        FONT, 0.45, color, 1, cv2.LINE_AA)

        # Compute features and decide
        if best:
            x1, y1, x2, y2    = best
            err_x, aspect, fill, bbox_cx = compute_features(x1, y1, x2, y2, iw, ih)
            err_smooth = args.ema_alpha * err_x + (1 - args.ema_alpha) * err_smooth
            candidate  = decide(err_smooth, aspect, fill, args)
            state      = fsm.update(candidate)

            draw_reference_lines(out, bbox_cx, iw, ih, err_smooth)
            draw_crosshair(out, iw // 2, ih // 2)
        else:
            err_x = aspect = fill = 0.0
            bbox_cx = iw // 2
            state = fsm.update("SEARCHING")
            draw_crosshair(out, iw // 2, ih // 2)

        # Overlay
        draw_banner(out, state, ih, iw)
        draw_direction_arrow(out, state, iw, ih)
        draw_debug_strip(out, state, err_smooth, aspect, fill,
                         best_conf, best_tid, fps_meter.fps, ih, iw)

        # FPS
        fps_meter.tick(time.perf_counter() - t0)

        # Log
        if log_writer and best:
            log_writer.writerow([time.time(), state, f"{err_smooth:.3f}",
                                 f"{aspect:.3f}", f"{fill:.3f}",
                                 f"{best_conf:.3f}", best_tid])

        cv2.imshow("AUV Navigation", out)
        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(out.tobytes())

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap = f"nav_snap_{snap_idx:04d}.png"
            cv2.imwrite(snap, out)
            print(f"  Snapshot: {snap}")
            snap_idx += 1

    cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"Video saved: {args.save}")
    if log_file:
        log_file.close()
        print(f"Log saved: {args.log}")
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

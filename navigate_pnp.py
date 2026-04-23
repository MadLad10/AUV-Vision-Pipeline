#!/usr/bin/env python3
"""
AUV Gate Navigation — PnP Pose Estimation

Gate is 3 m wide × 1.5 m tall (known physical dimensions).
solvePnP maps the YOLO bounding box corners to the 3-D gate model,
giving real-world distance (m), yaw (°), and pitch (°) instead of
fill-ratio / aspect proxies.

Navigation thresholds are in physical units:
  --yaw-center    deg   gate is "centred" if |yaw| < this
  --dist-passthru m     commit to pass-through when closer than this
  --dist-overshoot m    back-up threshold (single-bar mode)

Camera FOV is estimated from the frame width; provide --cam-fov for accuracy.

Usage:
    python navigate_pnp.py --weights best.pt --source video.mkv --save out.mp4
    python navigate_pnp.py --weights best.pt --source 0 --cam-fov 90
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
    p = argparse.ArgumentParser(description="AUV Gate Navigation — PnP")
    p.add_argument("--weights",         required=True)
    p.add_argument("--source",          required=True)
    p.add_argument("--conf",            type=float, default=0.50)
    p.add_argument("--imgsz",           type=int,   default=640)
    p.add_argument("--device",          default="mps")
    p.add_argument("--save",            default=None)
    p.add_argument("--log",             default=None)
    # Gate physical size
    p.add_argument("--gate-width",      type=float, default=3.0,  help="Gate width  (m)")
    p.add_argument("--gate-height",     type=float, default=1.5,  help="Gate height (m)")
    p.add_argument("--cam-fov",         type=float, default=60.0, help="Horizontal FOV (deg)")
    # Decision thresholds (physical units)
    p.add_argument("--yaw-center",      type=float, default=5.0,  help="Centred band (deg)")
    p.add_argument("--dist-passthru",   type=float, default=0.8,  help="Pass-through dist (m)")
    p.add_argument("--dist-overshoot",  type=float, default=1.5,  help="Too-close dist (m)")
    p.add_argument("--aspect-bar",      type=float, default=0.50, help="aspect < this → single bar")
    # Smoothing / FSM
    p.add_argument("--ema-alpha",       type=float, default=0.30)
    p.add_argument("--hysteresis",      type=int,   default=4)
    p.add_argument("--search-frames",   type=int,   default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colors & fonts
# ---------------------------------------------------------------------------

FONT    = cv2.FONT_HERSHEY_SIMPLEX
MONO    = cv2.FONT_HERSHEY_PLAIN

C_GREEN  = (0,  220,  60)
C_LGREEN = (0,  255, 120)
C_YELLOW = (0,  210, 255)
C_ORANGE = (0,  150, 255)
C_RED    = (0,   50, 255)
C_GRAY   = (140, 140, 140)
C_WHITE  = (230, 230, 230)
C_CYAN   = (255, 210,   0)
C_BLACK  = (0,     0,   0)

STATE_STYLE = {
    "SEARCHING":       ("SEARCHING",               C_GRAY),
    "ALIGNING_LEFT":   ("YAW LEFT",                C_YELLOW),
    "ALIGNING_RIGHT":  ("YAW RIGHT",               C_YELLOW),
    "APPROACH":        ("ALIGNED  →  FORWARD",     C_GREEN),
    "RECOVER_LEFT":    ("LATERAL LEFT",             C_ORANGE),
    "RECOVER_RIGHT":   ("LATERAL RIGHT",            C_ORANGE),
    "OVERSHOOT_LEFT":  ("BACK UP + LATERAL RIGHT",  C_RED),
    "OVERSHOOT_RIGHT": ("BACK UP + LATERAL LEFT",   C_RED),
    "OVERSHOOT":       ("BACK UP",                  C_RED),
    "PASS_THROUGH":    ("PASS THROUGH",             C_LGREEN),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FPSMeter:
    def __init__(self, window=30):
        self._t = deque(maxlen=window)
    def tick(self, dt):
        if dt > 0: self._t.append(dt)
    @property
    def fps(self):
        return 1.0 / (sum(self._t) / len(self._t)) if self._t else 0.0


class NavigationFSM:
    def __init__(self, hysteresis, search_frames):
        self.k             = hysteresis
        self.search_frames = search_frames
        self.state         = "SEARCHING"
        self._candidate    = "SEARCHING"
        self._count        = 0
        self._no_det       = 0

    def update(self, candidate):
        if candidate == "SEARCHING":
            self._no_det += 1
            if self._no_det >= self.search_frames:
                self.state = "SEARCHING"; self._candidate = "SEARCHING"; self._count = 0
        else:
            self._no_det = 0
            if candidate == self._candidate:
                self._count += 1
                if self._count >= self.k:
                    self.state = candidate
            else:
                self._candidate = candidate; self._count = 1
        return self.state


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------

def build_camera_matrix(iw, ih, fov_deg):
    f = iw / (2 * math.tan(math.radians(fov_deg / 2)))
    return np.array([[f, 0, iw / 2],
                     [0, f, ih / 2],
                     [0, 0,      1]], dtype=np.float64)


def solve_gate_pose(x1, y1, x2, y2, gate_3d, cam_K):
    """
    Map YOLO bbox corners → 3-D gate corners via solvePnP.
    Returns (distance_m, yaw_deg, pitch_deg) or (None, None, None) on failure.
    """
    pts_2d = np.array([[x1, y1], [x2, y1],
                       [x2, y2], [x1, y2]], dtype=np.float64)
    dist_c = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(gate_3d, pts_2d, cam_K, dist_c)
    if not ok:
        return None, None, None
    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
    distance  = math.sqrt(tx**2 + ty**2 + tz**2)
    yaw_deg   = math.degrees(math.atan2(tx,  tz))
    pitch_deg = math.degrees(math.atan2(-ty, tz))
    return distance, yaw_deg, pitch_deg


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(yaw_deg, distance, is_bar_only, args):
    if is_bar_only:
        if distance < args.dist_overshoot:
            if abs(yaw_deg) < args.yaw_center:
                return "OVERSHOOT"
            return "OVERSHOOT_LEFT" if yaw_deg < 0 else "OVERSHOOT_RIGHT"
        return "RECOVER_LEFT" if yaw_deg < 0 else "RECOVER_RIGHT"

    if distance < args.dist_passthru and abs(yaw_deg) < args.yaw_center:
        return "PASS_THROUGH"
    if abs(yaw_deg) < args.yaw_center:
        return "APPROACH"
    return "ALIGNING_LEFT" if yaw_deg < 0 else "ALIGNING_RIGHT"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_banner(frame, state, w):
    label, color = STATE_STYLE.get(state, ("UNKNOWN", C_GRAY))
    cv2.rectangle(frame, (0, 0), (w, 55), C_BLACK, -1)
    cv2.rectangle(frame, (0, 0), (w, 55), color, 2)
    (tw, _), _ = cv2.getTextSize(label, FONT, 1.2, 2)
    cv2.putText(frame, label, ((w - tw) // 2, 38), FONT, 1.2, color, 2, cv2.LINE_AA)


def draw_reference(frame, yaw_deg, iw, ih):
    cv2.line(frame, (iw // 2, 60), (iw // 2, ih - 40), C_WHITE, 1, cv2.LINE_AA)
    col = C_GREEN if abs(yaw_deg) < 5 else C_YELLOW if abs(yaw_deg) < 15 else C_RED
    px  = iw // 2 + int(yaw_deg / 45 * (iw // 2))
    px  = max(0, min(iw - 1, px))
    cv2.line(frame, (px, 60), (px, ih - 40), col, 2, cv2.LINE_AA)
    dz = int(iw * 0.05)
    ov = frame.copy()
    cv2.rectangle(ov, (iw // 2 - dz, 60), (iw // 2 + dz, ih - 40), (50, 50, 50), -1)
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)


def draw_arrows(frame, state, iw, ih):
    mx, my = iw // 2, ih // 2
    arm = 60
    arrows = {
        "ALIGNING_LEFT":   ((mx - arm, my),  (mx - arm * 3, my)),
        "ALIGNING_RIGHT":  ((mx + arm, my),  (mx + arm * 3, my)),
        "APPROACH":        ((mx, my + arm),  (mx, my - arm * 2)),
        "RECOVER_LEFT":    ((mx - arm, my),  (mx - arm * 3, my)),
        "RECOVER_RIGHT":   ((mx + arm, my),  (mx + arm * 3, my)),
        "OVERSHOOT":       ((mx, my - arm),  (mx, my + arm * 2)),
        "OVERSHOOT_LEFT":  ((mx, my - arm),  (mx, my + arm * 2)),
        "OVERSHOOT_RIGHT": ((mx, my - arm),  (mx, my + arm * 2)),
        "PASS_THROUGH":    ((mx, my + arm),  (mx, my - arm * 2)),
    }
    if state not in arrows:
        return
    _, color = STATE_STYLE[state]
    s, e = arrows[state]
    cv2.arrowedLine(frame, s, e, color, 4, cv2.LINE_AA, tipLength=0.3)
    if state == "OVERSHOOT_LEFT":
        cv2.arrowedLine(frame, (mx + arm, my), (mx + arm * 3, my), color, 4, cv2.LINE_AA, tipLength=0.3)
    elif state == "OVERSHOOT_RIGHT":
        cv2.arrowedLine(frame, (mx - arm, my), (mx - arm * 3, my), color, 4, cv2.LINE_AA, tipLength=0.3)


def draw_debug(frame, state, yaw, pitch, dist, conf, tid, fps, ih, iw):
    d = f"{dist:.2f}m"  if dist  is not None else "---"
    y = f"{yaw:+.1f}°"  if yaw   is not None else "---"
    pi = f"{pitch:+.1f}°" if pitch is not None else "---"
    txt = (f"state:{state}  yaw:{y}  pitch:{pi}  dist:{d}"
           f"  conf:{conf:.2f}  id:{tid}  fps:{fps:.1f}")
    cv2.rectangle(frame, (0, ih - 28), (iw, ih), C_BLACK, -1)
    cv2.putText(frame, txt, (6, ih - 8), MONO, 1.0, C_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    from ultralytics import YOLO

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.source}  ({iw}×{ih} @ {fps_src:.1f} fps)")

    gw, gh = args.gate_width, args.gate_height
    gate_3d = np.array([
        [-gw / 2, -gh / 2, 0],
        [ gw / 2, -gh / 2, 0],
        [ gw / 2,  gh / 2, 0],
        [-gw / 2,  gh / 2, 0],
    ], dtype=np.float64)
    cam_K = build_camera_matrix(iw, ih, args.cam_fov)
    print(f"Gate: {gw}m × {gh}m  |  Camera FOV: {args.cam_fov}°")

    ffmpeg_proc = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_proc = subprocess.Popen([
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{iw}x{ih}", "-pix_fmt", "bgr24", "-r", str(fps_src),
            "-i", "pipe:", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", args.save,
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(f"Saving to: {args.save}")

    log_file = log_writer = None
    if args.log:
        log_file   = open(args.log, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp", "state", "yaw_deg", "pitch_deg",
                              "distance_m", "conf", "track_id"])

    fsm        = NavigationFSM(args.hysteresis, args.search_frames)
    fps_meter  = FPSMeter()
    yaw_smooth = 0.0
    snap_idx   = 0

    cv2.namedWindow("AUV Navigation PnP", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("\nRunning — q: quit  s: snapshot\n")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, persist=True, conf=args.conf,
                              imgsz=args.imgsz, device=args.device, verbose=False)
        r       = results[0]
        has_ids = r.boxes.id is not None

        best = None; best_area = 0; best_conf = 0.0; best_tid = -1
        for i, box in enumerate(r.boxes):
            name = r.names[int(box.cls[0])]
            if "gate" not in name.lower():
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best      = (x1, y1, x2, y2)
                best_area = area
                best_conf = float(box.conf[0])
                best_tid  = int(r.boxes.id[i]) if has_ids else -1

        out = frame.copy()
        for i, box in enumerate(r.boxes):
            name = r.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            color = C_GREEN if "gate" in name.lower() else C_ORANGE
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)),
                        FONT, 0.45, color, 1, cv2.LINE_AA)

        distance = yaw_deg = pitch_deg = None
        if best:
            x1, y1, x2, y2 = best
            aspect   = (x2 - x1) / max(y2 - y1, 1)
            is_bar   = aspect <= args.aspect_bar
            distance, yaw_deg, pitch_deg = solve_gate_pose(x1, y1, x2, y2, gate_3d, cam_K)

            if yaw_deg is not None:
                yaw_smooth = args.ema_alpha * yaw_deg + (1 - args.ema_alpha) * yaw_smooth
                candidate  = decide(yaw_smooth, distance, is_bar, args)
                draw_reference(out, yaw_smooth, iw, ih)
            else:
                candidate = "SEARCHING"
            state = fsm.update(candidate)
        else:
            state = fsm.update("SEARCHING")

        cv2.line(out, (iw // 2 - 20, ih // 2), (iw // 2 + 20, ih // 2), C_CYAN, 1, cv2.LINE_AA)
        cv2.line(out, (iw // 2, ih // 2 - 20), (iw // 2, ih // 2 + 20), C_CYAN, 1, cv2.LINE_AA)
        draw_banner(out, state, iw)
        draw_arrows(out, state, iw, ih)
        draw_debug(out, state,
                   yaw_smooth if yaw_deg is not None else None,
                   pitch_deg, distance, best_conf, best_tid, fps_meter.fps, ih, iw)

        fps_meter.tick(time.perf_counter() - t0)

        if log_writer and best:
            log_writer.writerow([
                time.time(), state,
                f"{yaw_smooth:.2f}" if yaw_deg is not None else "",
                f"{pitch_deg:.2f}"  if pitch_deg is not None else "",
                f"{distance:.3f}"   if distance  is not None else "",
                f"{best_conf:.3f}", best_tid,
            ])

        cv2.imshow("AUV Navigation PnP", out)
        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(out.tobytes())

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap = f"pnp_snap_{snap_idx:04d}.png"
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
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

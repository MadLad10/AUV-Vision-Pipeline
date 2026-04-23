#!/usr/bin/env python3
"""
AUV Gate Navigation — ORB Feature Matching

ORB keypoints are detected inside the YOLO bounding box each frame.
Matched keypoints across consecutive frames give a robust lateral centroid
that is more stable than the raw bbox centre (handles partial occlusion,
reflections, edge-case detections).

Fill ratio (bbox area / frame area) is still used for distance estimation
since it requires no calibration.

No known gate dimensions or camera calibration required.

What you see on screen:
  - Magenta dots  : ORB keypoints detected inside the gate bbox
  - Yellow lines  : matched keypoint pairs between prev and current frame
  - Blue line     : frame centre
  - Coloured line : robust lateral gate centre (green/yellow/red by offset)

Usage:
    python navigate_orb.py --weights best.pt --source video.mkv --save out.mp4
    python navigate_orb.py --weights best.pt --source 0
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
    p = argparse.ArgumentParser(description="AUV Gate Navigation — ORB")
    p.add_argument("--weights",        required=True)
    p.add_argument("--source",         required=True)
    p.add_argument("--conf",           type=float, default=0.50)
    p.add_argument("--imgsz",          type=int,   default=640)
    p.add_argument("--device",         default="mps")
    p.add_argument("--save",           default=None)
    p.add_argument("--log",            default=None)
    # ORB
    p.add_argument("--orb-features",   type=int,   default=500)
    p.add_argument("--match-ratio",    type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--min-matches",    type=int,   default=6,    help="Min matches to use ORB centroid")
    # Decision thresholds (same units as navigate.py)
    p.add_argument("--err-center",     type=float, default=0.10, help="Centred band (fraction of half-width)")
    p.add_argument("--aspect-bar",     type=float, default=0.50)
    p.add_argument("--fill-passthru",  type=float, default=0.60)
    p.add_argument("--fill-overshoot", type=float, default=0.25)
    # Smoothing / FSM
    p.add_argument("--ema-alpha",      type=float, default=0.30)
    p.add_argument("--hysteresis",     type=int,   default=4)
    p.add_argument("--search-frames",  type=int,   default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Colors & fonts
# ---------------------------------------------------------------------------

FONT      = cv2.FONT_HERSHEY_SIMPLEX
MONO      = cv2.FONT_HERSHEY_PLAIN

C_GREEN   = (0,  220,  60)
C_LGREEN  = (0,  255, 120)
C_YELLOW  = (0,  210, 255)
C_ORANGE  = (0,  150, 255)
C_RED     = (0,   50, 255)
C_GRAY    = (140, 140, 140)
C_WHITE   = (230, 230, 230)
C_CYAN    = (255, 210,   0)
C_BLACK   = (0,     0,   0)
C_MAGENTA = (255,   0, 200)

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
# ORB matching
# ---------------------------------------------------------------------------

def detect_in_bbox(orb, gray, x1, y1, x2, y2):
    """Detect ORB features inside the bbox ROI; return in full-image coords."""
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return [], None
    kps, des = orb.detectAndCompute(roi, None)
    for kp in kps:
        kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
    return kps, des


def ratio_match(des1, des2, matcher, ratio):
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    pairs = matcher.knnMatch(des1, des2, k=2)
    return [m for m, n in pairs if m.distance < ratio * n.distance]


def robust_lateral(kps, good_matches, prev_kps, iw):
    """Normalised lateral centroid of matched current keypoints [-1, +1]."""
    pts = np.array([kps[m.trainIdx].pt[0] for m in good_matches])
    return float((np.mean(pts) - iw / 2) / (iw / 2))


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide(err_x, fill, is_bar_only, args):
    if is_bar_only:
        if fill >= args.fill_overshoot:
            if abs(err_x) < args.err_center:
                return "OVERSHOOT"
            return "OVERSHOOT_LEFT" if err_x < 0 else "OVERSHOOT_RIGHT"
        return "RECOVER_LEFT" if err_x < 0 else "RECOVER_RIGHT"

    if fill >= args.fill_passthru:
        return "PASS_THROUGH"
    if abs(err_x) < args.err_center:
        return "APPROACH"
    return "ALIGNING_LEFT" if err_x < 0 else "ALIGNING_RIGHT"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_banner(frame, state, w):
    label, color = STATE_STYLE.get(state, ("UNKNOWN", C_GRAY))
    cv2.rectangle(frame, (0, 0), (w, 55), C_BLACK, -1)
    cv2.rectangle(frame, (0, 0), (w, 55), color, 2)
    (tw, _), _ = cv2.getTextSize(label, FONT, 1.2, 2)
    cv2.putText(frame, label, ((w - tw) // 2, 38), FONT, 1.2, color, 2, cv2.LINE_AA)


def draw_reference(frame, err_x, orb_cx, iw, ih):
    cv2.line(frame, (iw // 2, 60), (iw // 2, ih - 40), C_WHITE, 1, cv2.LINE_AA)
    col = C_GREEN if abs(err_x) < 0.10 else C_YELLOW if abs(err_x) < 0.30 else C_RED
    cv2.line(frame, (orb_cx, 60), (orb_cx, ih - 40), col, 2, cv2.LINE_AA)
    dz = int(iw * 0.10)
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


def draw_debug(frame, state, err_x, fill, n_kps, n_matches, conf, fps, ih, iw):
    src_str = "ORB" if n_matches >= 6 else "bbox"
    txt = (f"state:{state}  lateral:{err_x:+.3f}({src_str})"
           f"  fill:{fill:.2f}  kps:{n_kps}  matches:{n_matches}"
           f"  conf:{conf:.2f}  fps:{fps:.1f}")
    cv2.rectangle(frame, (0, ih - 28), (iw, ih), C_BLACK, -1)
    cv2.putText(frame, txt, (6, ih - 8), MONO, 1.0, C_GRAY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    from ultralytics import YOLO

    print(f"Loading model: {args.weights}")
    model   = YOLO(args.weights)
    orb     = cv2.ORB_create(args.orb_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.source}  ({iw}×{ih} @ {fps_src:.1f} fps)")

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
        log_writer.writerow(["timestamp", "state", "lateral", "fill",
                              "n_kps", "n_matches", "conf"])

    fsm        = NavigationFSM(args.hysteresis, args.search_frames)
    fps_meter  = FPSMeter()
    err_smooth = 0.0
    prev_kps   = None
    prev_des   = None
    snap_idx   = 0

    cv2.namedWindow("AUV Navigation ORB", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    print("\nRunning — q: quit  s: snapshot\n")

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        for box in r.boxes:
            name = r.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            color = C_GREEN if "gate" in name.lower() else C_ORANGE
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)),
                        FONT, 0.45, color, 1, cv2.LINE_AA)

        n_kps = n_matches = 0
        orb_cx = iw // 2

        if best:
            x1, y1, x2, y2 = best
            aspect  = (x2 - x1) / max(y2 - y1, 1)
            fill    = (x2 - x1) * (y2 - y1) / (iw * ih)
            is_bar  = aspect <= args.aspect_bar
            bbox_cx = (x1 + x2) // 2

            kps, des = detect_in_bbox(orb, gray, x1, y1, x2, y2)
            n_kps    = len(kps)

            # Draw ORB keypoints
            for kp in kps:
                cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), 3, C_MAGENTA, -1)

            good = ratio_match(prev_des, des, matcher, args.match_ratio)
            n_matches = len(good)

            if prev_kps and n_matches >= args.min_matches:
                # Draw match lines (first 20)
                for m in good[:20]:
                    p1 = tuple(map(int, prev_kps[m.queryIdx].pt))
                    p2 = tuple(map(int, kps[m.trainIdx].pt))
                    cv2.line(out, p1, p2, (50, 200, 200), 1, cv2.LINE_AA)
                err_x  = robust_lateral(kps, good, prev_kps, iw)
                orb_cx = int(np.mean([kps[m.trainIdx].pt[0] for m in good]))
            else:
                err_x  = (bbox_cx - iw / 2) / (iw / 2)
                orb_cx = bbox_cx

            err_smooth = args.ema_alpha * err_x + (1 - args.ema_alpha) * err_smooth
            candidate  = decide(err_smooth, fill, is_bar, args)
            state      = fsm.update(candidate)

            draw_reference(out, err_smooth, orb_cx, iw, ih)
            prev_kps = kps
            prev_des = des
        else:
            fill = 0.0
            state    = fsm.update("SEARCHING")
            prev_kps = None
            prev_des = None

        cv2.line(out, (iw // 2 - 20, ih // 2), (iw // 2 + 20, ih // 2), C_CYAN, 1, cv2.LINE_AA)
        cv2.line(out, (iw // 2, ih // 2 - 20), (iw // 2, ih // 2 + 20), C_CYAN, 1, cv2.LINE_AA)
        draw_banner(out, state, iw)
        draw_arrows(out, state, iw, ih)
        draw_debug(out, state, err_smooth, fill, n_kps, n_matches, best_conf, fps_meter.fps, ih, iw)

        fps_meter.tick(time.perf_counter() - t0)

        if log_writer and best:
            log_writer.writerow([time.time(), state, f"{err_smooth:.3f}",
                                 f"{fill:.3f}", n_kps, n_matches, f"{best_conf:.3f}"])

        cv2.imshow("AUV Navigation ORB", out)
        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(out.tobytes())

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap = f"orb_snap_{snap_idx:04d}.png"
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

#!/usr/bin/env python3
"""
Extract N evenly-spaced frames from a video and save RAW vs INPAINTED
side-by-side PNGs for visual inspection.

Usage:
    python check_inpaint.py --source video.mkv
    python check_inpaint.py --source video.mkv --frames 10 --out inpaint_check
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--frames", type=int, default=8, help="Number of frames to sample")
    p.add_argument("--out",    default="inpaint_check", help="Output folder")
    p.add_argument("--sat-thresh",     type=int, default=50)
    p.add_argument("--dark-thresh",    type=int, default=40)
    p.add_argument("--bright-thresh",  type=int, default=230)
    p.add_argument("--dilate-ksize",   type=int, default=5)
    p.add_argument("--inpaint-radius", type=int, default=3)
    return p.parse_args()


def build_mask(frame, sat_thresh, dark_thresh, bright_thresh, dilate_ksize):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s < sat_thresh) | (v < dark_thresh) | (v > bright_thresh)).astype(np.uint8) * 255
    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask   = cv2.dilate(mask, kernel, iterations=1)
    return mask


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(total // args.frames, 1)
    print(f"Video: {total} frames — sampling every {step} frames ({args.frames} total)")

    saved = 0
    for i in range(args.frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        mask     = build_mask(frame, args.sat_thresh, args.dark_thresh,
                              args.bright_thresh, args.dilate_ksize)
        restored = cv2.inpaint(frame, mask, args.inpaint_radius, cv2.INPAINT_TELEA) \
                   if mask.any() else frame.copy()

        # Draw mask overlay in red on a copy
        mask_vis        = frame.copy()
        mask_vis[mask > 0] = [0, 0, 200]

        # Stack: RAW | MASK | INPAINTED
        combined = cv2.hconcat([frame, mask_vis, restored])

        # Add labels
        for j, lbl in enumerate(["RAW", "MASK", "INPAINTED"]):
            x = j * frame.shape[1] + 10
            cv2.putText(combined, lbl, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)

        out_path = out_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved: {out_path}")
        saved += 1

    cap.release()
    print(f"\nDone — {saved} frames saved to '{args.out}/'")


if __name__ == "__main__":
    main()

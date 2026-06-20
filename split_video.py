#!/usr/bin/env python3
"""Split a video into N equal parts, saving frames to folders and CVAT-ready zip files."""

import argparse
import math
import sys
import zipfile
from pathlib import Path

import cv2


def count_frames(cap) -> int:
    print("Counting actual frames...", flush=True)
    n = 0
    while cap.grab():
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n


def split_video(video_path: str, num_parts: int, desired_frames: int, output_dir: str, zip_only: bool) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video '{video_path}'")

    total_frames = count_frames(cap)
    if total_frames == 0:
        sys.exit("Error: video has 0 frames")

    # Evenly spaced indices (0-based) across the video
    step = total_frames / desired_frames
    sample_indices = sorted(set(int(i * step) for i in range(desired_frames)))
    actual_sampled = len(sample_indices)

    frames_per_part = math.ceil(actual_sampled / num_parts)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Video         : {video_path}")
    print(f"Total frames  : {total_frames}")
    print(f"Extracting    : {actual_sampled}  |  Parts: {num_parts}  |  ~{frames_per_part} frames/part")
    if zip_only:
        print("Mode          : zip only (no frame folders kept)")
    else:
        print("Mode          : frame folders + zips")

    # Walk through video once, only decoding sampled frames
    sample_set = set(sample_indices)
    sampled_frames = []  # list of (global_idx_1based, frame_bytes_or_path)

    global_idx = 1
    video_pos = 0
    while True:
        if video_pos in sample_set:
            ret, frame = cap.read()
            if not ret:
                break
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                sampled_frames.append((global_idx, buf.tobytes()))
                global_idx += 1
        else:
            if not cap.grab():
                break
        video_pos += 1

    cap.release()

    # Split sampled frames into parts and zip
    for part in range(1, num_parts + 1):
        part_slice = sampled_frames[(part - 1) * frames_per_part: part * frames_per_part]
        if not part_slice:
            print(f"  vid_{part}  (empty, skipped)")
            continue

        zip_path = out_root / f"vid_{part}.zip"
        part_dir = out_root / f"vid_{part}"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for idx, data in part_slice:
                fname = f"frame_{idx:06d}.jpg"
                if not zip_only:
                    part_dir.mkdir(parents=True, exist_ok=True)
                    img_path = part_dir / fname
                    img_path.write_bytes(data)
                zf.writestr(fname, data)

        start = part_slice[0][0]
        end = part_slice[-1][0]
        print(f"  vid_{part}  frames {start}–{end}  ({len(part_slice)} frames)  -> {zip_path.name}")

    print(f"\nDone. Output: {out_root.resolve()}")
    print("Upload each vid_N.zip to CVAT via: Create Task → Add Files → (select zip)")


def main():
    parser = argparse.ArgumentParser(
        description="Split a video into N parts with global frame numbering, ready for CVAT upload."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output root directory (prompted if not provided)",
    )
    parser.add_argument(
        "--zip-only",
        action="store_true",
        help="Produce only zip files without keeping the frame folders on disk",
    )
    args = parser.parse_args()

    # Step 1: count frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open video '{args.video}'")
    total_frames = count_frames(cap)
    cap.release()
    if total_frames == 0:
        sys.exit("Error: video has 0 frames")

    print(f"\nTotal frames in video: {total_frames}\n")

    # Step 2: ask how many frames to extract
    while True:
        raw = input(f"How many frames to extract? [default: {total_frames}]: ").strip()
        if not raw:
            desired_frames = total_frames
            break
        try:
            desired_frames = int(raw)
            if 1 <= desired_frames <= total_frames:
                break
            print(f"  Must be between 1 and {total_frames}.")
        except ValueError:
            print("  Enter a valid number.")

    # Step 3: ask number of parts
    while True:
        raw = input("Split into how many parts? [default: 1]: ").strip()
        if not raw:
            num_parts = 1
            break
        try:
            num_parts = int(raw)
            if num_parts >= 1:
                break
            print("  Must be >= 1.")
        except ValueError:
            print("  Enter a valid number.")

    # Step 4: output dir
    output_dir = args.output
    if not output_dir:
        output_dir = input("Output folder name: ").strip()
        if not output_dir:
            sys.exit("Error: output folder name cannot be empty")

    print()
    split_video(args.video, num_parts, desired_frames, output_dir, args.zip_only)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full AUV annotation-to-training pipeline — email automation via n8n.

Steps:
  1. Split video into N zips  (one per annotator)
  2. Trigger n8n workflow → n8n sends emails + watches for replies
  3. Local webhook receives callback from n8n when all zips returned
  4. Merge CVAT exports → split 85/5/10 → train YOLO11n

Setup:
  1. Start n8n locally:
       docker run -it --rm -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n
  2. Import n8n_workflow.json into n8n (Settings → Import Workflow)
  3. Add Gmail OAuth2 credentials in n8n and link to the Gmail nodes
  4. Activate both workflows in n8n
  5. Fill in pipeline_config.yaml (n8n section)

Usage:
  python pipeline.py --video dive.mp4 --classes gate rescue repair
  python pipeline.py --video dive.mp4 --frames 500
"""

import argparse
import base64
import http.server
import json
import math
import random
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import cv2
import requests
import yaml


CONFIG_FILE = "pipeline_config.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

EXAMPLE_CONFIG = """\
n8n:
  url: "http://localhost:5678"
  api_key: ""                        # n8n Settings → API → Create API Key
  send_webhook: "auv-pipeline-send"  # must match the Webhook node path in n8n
  callback_port: 9876                # pipeline listens here for n8n callback
  timeout_hours: 48

annotators:
  - name: Annotator 1
    email: annotator1@example.com
  - name: Annotator 2
    email: annotator2@example.com

training:
  epochs: 200
  batch: 16
  patience: 50
  imgsz: 640
  device: "0"

split:
  train: 0.85
  val: 0.05
  test: 0.10
  seed: 42
"""


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        p.write_text(EXAMPLE_CONFIG)
        sys.exit(
            f"\nConfig not found. Created template at '{path}'.\n"
            "Fill in your n8n settings and annotator emails, then re-run."
        )
    with open(p) as f:
        return yaml.safe_load(f)


# ── Step 1: Split video ───────────────────────────────────────────────────────

def _count_frames(cap) -> int:
    n = 0
    while cap.grab():
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n


def split_video_to_zips(video_path: str, num_parts: int,
                         desired_frames: int, output_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    total   = _count_frames(cap)
    step    = total / desired_frames
    indices = sorted(set(int(i * step) for i in range(desired_frames)))
    fpp     = math.ceil(len(indices) / num_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Total frames : {total}  |  Extracting : {len(indices)}  |  Parts : {num_parts}")

    sample_set = set(indices)
    sampled: list[tuple[int, bytes]] = []
    gidx, pos = 1, 0
    while True:
        if pos in sample_set:
            ret, frame = cap.read()
            if not ret:
                break
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                sampled.append((gidx, buf.tobytes()))
                gidx += 1
        else:
            if not cap.grab():
                break
        pos += 1
    cap.release()

    zip_paths: list[Path] = []
    for part in range(1, num_parts + 1):
        chunk = sampled[(part - 1) * fpp: part * fpp]
        if not chunk:
            continue
        zp = output_dir / f"part_{part}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
            for idx, data in chunk:
                zf.writestr(f"frame_{idx:06d}.jpg", data)
        print(f"  part_{part}.zip  ({len(chunk)} frames)")
        zip_paths.append(zp)
    return zip_paths


# ── Step 2: Trigger n8n send workflow ─────────────────────────────────────────

def trigger_n8n_send(n8n_cfg: dict, annotators: list[dict],
                      zip_paths: list[Path], classes: list[str],
                      callback_url: str) -> None:
    webhook_url = f"{n8n_cfg['url']}/webhook/{n8n_cfg['send_webhook']}"
    payload = {
        "annotators": [
            {
                "name":     ann["name"],
                "email":    ann["email"],
                "zip_path": str(zp.resolve()),
                "part_num": i + 1,
            }
            for i, (ann, zp) in enumerate(zip(annotators, zip_paths))
        ],
        "classes":      classes,
        "callback_url": callback_url,
        "expected":     len(annotators),
    }
    headers = {}
    if n8n_cfg.get("api_key"):
        headers["X-N8N-API-KEY"] = n8n_cfg["api_key"]

    print(f"  Triggering n8n → {webhook_url}")
    resp = requests.post(webhook_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    print(f"  n8n acknowledged: {resp.status_code}")


# ── Step 3: Wait for n8n callback ─────────────────────────────────────────────

def wait_for_n8n_callback(port: int, expected: int,
                           output_dir: Path, timeout_hours: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    received: dict[str, Path] = {}
    done = threading.Event()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            email    = body.get("email", f"annotator_{len(received)}")
            filename = body.get("filename", "annotated.zip")
            zip_b64  = body.get("zip_base64", "")

            save_path = output_dir / f"annotated_{email.split('@')[0]}.zip"
            save_path.write_bytes(base64.b64decode(zip_b64))
            received[email] = save_path
            print(f"  n8n → received {filename} from {email}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

            if len(received) >= expected:
                done.set()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"  Callback server listening on http://localhost:{port}")
    print(f"  Waiting for {expected} annotated zip(s) from n8n ...")

    done.wait(timeout=timeout_hours * 3600)
    server.shutdown()

    if len(received) < expected:
        print(f"\n[WARN] Timed out — received {len(received)}/{expected} replies.")
        if not received:
            sys.exit("No replies received. Exiting.")
        print("Proceeding with what was received...")

    return list(received.values())


# ── Step 4: Merge + split dataset ─────────────────────────────────────────────

def _extract_zip(zp: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(dest)
    print(f"  Extracted: {zp.name} → {dest.name}/")
    return dest


def _find_label(img: Path, root: Path) -> Path | None:
    c = img.with_suffix(".txt")
    if c.exists():
        return c
    parts = list(img.parts)
    for seg in ("images", "obj_train_data", "obj_valid_data", "obj_test_data"):
        if seg in parts:
            p = parts[:]
            p[p.index(seg)] = "labels"
            c = Path(*p).with_suffix(".txt")
            if c.exists():
                return c
    hits = list(root.rglob(f"{img.stem}.txt"))
    return hits[0] if hits else None


def _collect_pairs(d: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(d.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lbl = _find_label(img, d)
        if lbl:
            pairs.append((img, lbl))
    return pairs


def _detect_classes(d: Path) -> list[str]:
    for name in ("obj.names", "classes.txt", "data.names"):
        for p in d.rglob(name):
            lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            if lines:
                return lines
    for p in d.rglob("data.yaml"):
        with open(p) as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if names:
            return ([names[k] for k in sorted(names)] if isinstance(names, dict)
                    else list(names))
    return []


def merge_and_split(zip_paths: list[Path], output_dir: Path,
                    class_names: list[str], split_cfg: dict) -> Path:
    extract_dir = output_dir / "extracted"
    source_dirs = [_extract_zip(zp, extract_dir / zp.stem) for zp in zip_paths]

    all_pairs: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    detected = class_names[:]

    for src in source_dirs:
        pairs = _collect_pairs(src)
        print(f"  {src.name}: {len(pairs)} image-label pairs")
        if not detected:
            detected = _detect_classes(src)
        for img, lbl in pairs:
            fname = img.name
            if fname in seen:
                fname = f"{src.name}__{img.name}"
            seen.add(fname)
            all_pairs.append((img, lbl, fname))

    if not all_pairs:
        sys.exit("No valid image-label pairs found.")

    random.seed(split_cfg.get("seed", 42))
    random.shuffle(all_pairs)

    n       = len(all_pairs)
    n_train = round(n * split_cfg.get("train", 0.85))
    n_val   = round(n * split_cfg.get("val",   0.05))
    splits  = {
        "train": all_pairs[:n_train],
        "val":   all_pairs[n_train: n_train + n_val],
        "test":  all_pairs[n_train + n_val:],
    }
    print(f"  Split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    dataset_dir = output_dir / "dataset"
    for split_name, chunk in splits.items():
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl, fname in chunk:
            shutil.copy2(img, img_dir / fname)
            shutil.copy2(lbl, lbl_dir / Path(fname).with_suffix(".txt").name)

    if not detected:
        detected = ["class0"]
        print("  [WARN] Could not detect classes — edit dataset/data.yaml before training.")

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(dataset_dir.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc":    len(detected),
            "names": detected,
        }, f, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml saved  |  classes: {detected}")
    return yaml_path


# ── Step 5: Train ─────────────────────────────────────────────────────────────

def train_model(data_yaml: Path, train_cfg: dict, run_name: str) -> Path:
    try:
        import albumentations as A
        from ultralytics.data import augment as ult_augment

        transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
            A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        ])
        _orig = ult_augment.Albumentations.__init__

        def _patched(self, p=1.0, **kwargs):
            _orig(self, p)
            self.transform = transforms

        ult_augment.Albumentations.__init__ = _patched
        print("  [augment] Underwater Albumentations pipeline active.")
    except (ImportError, AttributeError):
        pass

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_yaml),
        epochs=train_cfg.get("epochs",   200),
        patience=train_cfg.get("patience", 50),
        imgsz=train_cfg.get("imgsz",     640),
        batch=train_cfg.get("batch",      16),
        device=str(train_cfg.get("device", "0")),
        amp=True, pretrained=True, optimizer="AdamW",
        lr0=1e-3, lrf=1e-2, warmup_epochs=3, close_mosaic=10,
        save=True, val=True, plots=True,
        project="runs/pipeline",
        name=run_name,
        hsv_h=0.02, hsv_s=0.90, hsv_v=0.60,
    )
    return Path("runs/pipeline") / run_name / "weights" / "best.pt"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AUV annotation-to-training pipeline (n8n)")
    p.add_argument("--video",   required=True, help="Input video")
    p.add_argument("--classes", nargs="+", default=[], help="Class names (auto-detected if omitted)")
    p.add_argument("--frames",  type=int, default=None, help="Frames to extract (default: all)")
    p.add_argument("--config",  default=CONFIG_FILE)
    p.add_argument("--output",  default="pipeline_output")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    config    = load_config(args.config)
    output    = Path(args.output)
    n8n_cfg   = config["n8n"]
    annotators = config["annotators"]
    train_cfg  = config.get("training", {})
    split_cfg  = config.get("split", {})

    callback_port = n8n_cfg.get("callback_port", 9876)
    callback_url  = f"http://localhost:{callback_port}/n8n-callback"

    # ── 1. Split video ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"[1/5] Splitting video into {len(annotators)} parts ...")
    print(f"{'─'*55}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    total_frames = _count_frames(cap)
    cap.release()

    zip_paths = split_video_to_zips(
        args.video, len(annotators),
        args.frames or total_frames, output / "zips"
    )
    if len(zip_paths) < len(annotators):
        annotators = annotators[:len(zip_paths)]

    # ── 2. Trigger n8n ───────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("[2/5] Triggering n8n email workflow ...")
    print(f"{'─'*55}")
    trigger_n8n_send(n8n_cfg, annotators, zip_paths, args.classes, callback_url)

    # ── 3. Wait for n8n callback ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("[3/5] Waiting for annotated zips via n8n ...")
    print(f"{'─'*55}")
    received_zips = wait_for_n8n_callback(
        callback_port, len(annotators),
        output / "replies", n8n_cfg.get("timeout_hours", 48)
    )

    # ── 4. Merge + split ──────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("[4/5] Merging annotated datasets ...")
    print(f"{'─'*55}")
    data_yaml = merge_and_split(received_zips, output, args.classes, split_cfg)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("[5/5] Training YOLO11n ...")
    print(f"{'─'*55}")
    train_model(data_yaml, train_cfg, f"pipeline_{Path(args.video).stem}")

    print(f"\n{'═'*55}")
    print("  Pipeline complete!")
    print(f"  Dataset : {data_yaml}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()

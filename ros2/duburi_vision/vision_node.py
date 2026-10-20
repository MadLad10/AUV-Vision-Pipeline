#!/usr/bin/env python3
"""ROS 2 vision node: YOLO detection + ByteTrack tracking on the AUV camera.

Subscribes  vision_topic (custom_interfaces/CameraMsg)
            What the mission wants: which camera (`camera`) and which model (`model`).
Publishes   camera_topic (custom_interfaces/CameraMsg)
            One message per tracked object: class, confidence, smoothed box,
            box centre and its offset from the frame centre (x_diff / y_diff),
            which the controller uses to steer towards the target.

Tracking: Ultralytics `model.track(..., tracker="bytetrack.yaml", persist=True)` keeps a
stable ID per object across frames; `BoxSmoother` then low-pass filters each track's box
so the steering offsets don't jitter.

Parameters (ros2 run ... --ros-args -p name:=value):
    gate_weights  path to the gate model (.pt or TensorRT .engine)
    conf          detection confidence threshold (default 0.80)
    tracker       Ultralytics tracker config (default bytetrack.yaml)
    show          show the annotated window (default true)

Originally `simulator/vision_node.py` in Duburi_ws (entry point `vision_node`).
"""

import os

os.environ["YOLO_CONFIG_DIR"] = "/tmp"

import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import torch  # noqa: E402
from custom_interfaces.msg import CameraMsg  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from ultralytics.models.yolo.model import YOLO  # noqa: E402
from ultralytics.utils.plotting import Annotator  # noqa: E402

DEFAULT_GATE_WEIGHTS = "/home/duburi-jetson/workspaces/duburi_ws/src/duburi_vision/models/gate_flare_medium_100ep.pt"

# camera name (from vision_topic) -> V4L2 device index
CAMERA_DEVICES = {"front": 0}


class BoxSmoother:
    """Exponential moving average of each track's box: smooth = a * new + (1 - a) * smooth."""

    def __init__(self, alpha=0.5):
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


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.declare_parameter("gate_weights", DEFAULT_GATE_WEIGHTS)
        self.declare_parameter("conf", 0.80)
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("show", True)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.vision_subscription_ = self.create_subscription(CameraMsg, "vision_topic", self.listener_callback, qos)
        self.camera_publisher_ = self.create_publisher(CameraMsg, "camera_topic", qos)
        self.last_msg_vision = None  # latest request from the mission

    def listener_callback(self, msg):
        self.last_msg_vision = msg

    def param(self, name):
        return self.get_parameter(name).value


def publish_detection(node, class_name, conf, box, frame_cx, frame_cy):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    msg = CameraMsg()
    msg.class_name = [class_name]
    msg.cx, msg.cy = cx, cy
    msg.frame_cx, msg.frame_cy = frame_cx, frame_cy
    msg.x1, msg.y1, msg.x2, msg.y2 = x1, y1, x2, y2
    msg.width, msg.height = x2 - x1, y2 - y1
    msg.x_diff, msg.y_diff = cx - frame_cx, cy - frame_cy
    msg.conf = conf
    node.camera_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    log = node.get_logger()
    cap = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_gate = YOLO(node.param("gate_weights"), task="track", verbose=False)
        model_gate.to(device)
        log.info(f"Gate model loaded on {device}: {node.param('gate_weights')}")

        current_camera = None
        smoother = BoxSmoother(alpha=0.5)
        show = node.param("show")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1 / 30)
            request = node.last_msg_vision

            selected_camera = "front"
            if request and request.camera in ("front", "down", "grabber"):
                selected_camera = request.camera
            if selected_camera not in CAMERA_DEVICES:
                log.warn(f"No device configured for '{selected_camera}' camera, staying on {current_camera}")
                selected_camera = current_camera or "front"

            if selected_camera != current_camera:
                if cap:
                    cap.release()
                cap = cv2.VideoCapture(CAMERA_DEVICES[selected_camera], cv2.CAP_V4L2)
                if not cap.isOpened():
                    log.error(f"Failed to open {selected_camera} camera. Retrying...")
                    time.sleep(1)
                    continue
                log.info(f"Switched to {selected_camera} camera.")
                current_camera = selected_camera

            success, img = cap.read()
            if not success or img is None:
                log.error("Failed to read frame from camera")
                time.sleep(0.5)
                continue

            results = None
            if request and request.model == "gate":
                results = model_gate.track(
                    source=img, conf=node.param("conf"), persist=True, tracker=node.param("tracker"), verbose=False
                )[0]

            if results is not None and len(results.boxes):
                boxes, names = results.boxes, results.names
                h, w, _ = img.shape
                frame_cx, frame_cy = w // 2, h // 2
                annotator = Annotator(img)
                has_ids = boxes.id is not None
                if has_ids:
                    smoother.prune(set(boxes.id.int().cpu().tolist()))

                for i, box in enumerate(boxes):
                    class_name = names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    raw = box.xyxy[0].tolist()
                    smoothed = smoother.update(int(boxes.id[i]), raw) if has_ids else raw
                    x1, y1, x2, y2 = map(int, smoothed)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    annotator.box_label((x1, y1, x2, y2), f"{class_name} {conf:.2f}")
                    cv2.line(img, (w // 2, h), (w // 2, 0), (0, 255, 0), 2)
                    cv2.line(img, (frame_cx, frame_cy), (cx, cy), (0, 255, 0), 2)
                    cv2.putText(
                        img, f"W:{x2 - x1}  H:{y2 - y1}  X:{cx - frame_cx} Y:{cy - frame_cy} CX: {cx}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1,
                    )
                    publish_detection(node, class_name, conf, (x1, y1, x2, y2), frame_cx, frame_cy)
                img = annotator.result()

            if show:
                cv2.imshow("Vision camera", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

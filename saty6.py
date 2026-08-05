# ============================================================
# NOTE:
# This file uses lower latency steering defaults than
# sep_com_combined_steering_filter.py.
# ============================================================

# ============================================================
#  sep_com_combined.py
#  Single-computer pipeline:
#  cameras -> BEV -> segmentation/navigation -> /cmd_vel
#  เพิ่มเงื่อนไขกรองค่าพวงมาลัย
#  เพิ่มการลดขนาดภาพเข้าโมเดล
# ============================================================

import argparse
import math
import os
import queue
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image as PILImage
from scipy.interpolate import CubicSpline
from torchvision import transforms as T

import network
from bev_processor import BEVProcessor, BEV_AVAILABLE, img_car
from datasets import VOCSegmentation, Cityscapes
from nav_processing import (
    DEFAULT_DRIVABLE_CLASS_IDS,
    MODE_LABEL,
    ROI_H,
    ROI_W,
    ROS2_AVAILABLE,
    SAFETY_CHECK_INTERVAL,
    SAFETY_TIMEOUT,
    WHEELBASE,
    alpha_smooth,
    calculate_steering_angle,
    check_drivable_height_ahead,
    check_drivable_width_ahead,
    clean_drivable_mask,
    compute_roi_linear_x,
    detect_vehicle_and_heading,
    draw_compare_roi,
    draw_vehicle_roi,
    select_device,
    select_target_point,
)

DEFAULT_STEERING_FILTER_ALPHA = 0.9
DEFAULT_STEERING_DEADBAND_DEG = 1.0
DEFAULT_STEERING_MAX_RATE_DEG_S = 10.0
DEFAULT_SPLINE_CURVATURE_THRESHOLD = 0.003
DEFAULT_STRAIGHT_LINEAR_X = 1.0
DEFAULT_CURVED_LINEAR_X = 0.5
DEFAULT_MIN_NAV_PATH_POINTS = 30

prev_spline_points = None

if ROS2_AVAILABLE:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
else:
    Node = object
    Twist = None
    String = None


class CombinedVehicleControllerNode(Node):
    def __init__(
        self,
        cmd_vel_topic="/cmd_vel_1",
        vehicle_config=None,
        cmd_vel_publish_hz=30.0,
    ):
        super().__init__("combined_vehicle_controller")
        self.config = vehicle_config or {}
        self.cmd_vel_topic = cmd_vel_topic
        self.steering_filter_enabled = self.config.get(
            "steering_filter_enabled", True
        )
        self.steering_filter_alpha = min(
            0.99,
            max(
                0.0,
                float(
                    self.config.get(
                        "steering_filter_alpha", DEFAULT_STEERING_FILTER_ALPHA
                    )
                ),
            ),
        )
        self.steering_deadband_deg = float(
            self.config.get(
                "steering_deadband_deg", DEFAULT_STEERING_DEADBAND_DEG
            )
        )
        self.steering_max_rate_deg_s = float(
            self.config.get(
                "steering_max_rate_deg_s", DEFAULT_STEERING_MAX_RATE_DEG_S
            )
        )
        self.last_filtered_steering_deg = None
        self.last_filter_time = None
        self.cmd_vel_publish_hz = max(float(cmd_vel_publish_hz), 1.0)
        self.cmd_lock = threading.Lock()
        self.latest_cmd = Twist()
        self.latest_cmd_time = time.time()

        low_latency_qos = QoSProfile(depth=1)
        self.cmd_vel_publisher = self.create_publisher(
            Twist, self.cmd_vel_topic, low_latency_qos
        )
        self.navigation_status_publisher = self.create_publisher(
            String, "/navigation_status", 10
        )

        self.last_cmd_time = time.time()
        self.safety_timer = self.create_timer(
            SAFETY_CHECK_INTERVAL, self.safety_check
        )
        self.cmd_vel_timer = self.create_timer(
            1.0 / self.cmd_vel_publish_hz, self.publish_latest_cmd
        )
        self.get_logger().info(
            f"Combined vehicle controller publishing Twist to "
            f"{self.cmd_vel_topic} at {self.cmd_vel_publish_hz:.1f} Hz"
        )
        if self.steering_filter_enabled:
            self.get_logger().info(
                "Steering filter enabled: "
                f"alpha={self.steering_filter_alpha:.2f}, "
                f"deadband={self.steering_deadband_deg:.2f} deg, "
                f"max_rate={self.steering_max_rate_deg_s:.1f} deg/s"
            )

    def publish_cmd_vel(self, steering_angle_deg, linear_x=1.0):
        steering_angle_deg = self.filter_steering_angle(
            steering_angle_deg, linear_x
        )
        cmd = Twist()
        linear_x = float(linear_x)
        cmd.linear.x = 0.0 if linear_x < 0.0 else 1.0 if linear_x > 1.0 else linear_x
        cmd.angular.z = math.radians(steering_angle_deg)
        now = time.time()
        with self.cmd_lock:
            self.latest_cmd = cmd
            self.latest_cmd_time = now
        self.cmd_vel_publisher.publish(cmd)
        self.last_cmd_time = now

    def filter_steering_angle(self, steering_angle_deg, linear_x):
        steering_angle_deg = float(steering_angle_deg)
        if not self.steering_filter_enabled:
            return steering_angle_deg

        now = time.time()
        if self.last_filtered_steering_deg is None:
            self.last_filtered_steering_deg = steering_angle_deg
            self.last_filter_time = now
            return steering_angle_deg

        if linear_x <= 0.0 and abs(steering_angle_deg) < 1e-6:
            self.last_filtered_steering_deg = 0.0
            self.last_filter_time = now
            return 0.0

        previous = self.last_filtered_steering_deg
        filtered = (
            self.steering_filter_alpha * previous
            + (1.0 - self.steering_filter_alpha) * steering_angle_deg
        )

        if abs(filtered - previous) < self.steering_deadband_deg:
            filtered = previous

        dt = (
            0.0
            if self.last_filter_time is None
            else max(0.0, now - self.last_filter_time)
        )
        if self.steering_max_rate_deg_s > 0.0 and dt > 0.0:
            max_delta = self.steering_max_rate_deg_s * dt
            lower = previous - max_delta
            upper = previous + max_delta
            if filtered < lower:
                filtered = lower
            elif filtered > upper:
                filtered = upper

        self.last_filtered_steering_deg = filtered
        self.last_filter_time = now
        return filtered

    def publish_latest_cmd(self):
        with self.cmd_lock:
            cmd = self.latest_cmd
            cmd_age = time.time() - self.latest_cmd_time
        if cmd_age > SAFETY_TIMEOUT:
            cmd = Twist()
        self.cmd_vel_publisher.publish(cmd)

    def stop(self):
        stop_cmd = Twist()
        now = time.time()
        with self.cmd_lock:
            self.latest_cmd = stop_cmd
            self.latest_cmd_time = now
        self.last_filtered_steering_deg = 0.0
        self.last_filter_time = now
        self.cmd_vel_publisher.publish(stop_cmd)
        self.get_logger().info("Vehicle stopped")

    def publish_status(self, msg_str):
        msg = String()
        msg.data = msg_str
        self.navigation_status_publisher.publish(msg)

    def safety_check(self):
        elapsed = time.time() - self.last_cmd_time
        if elapsed > SAFETY_TIMEOUT:
            self.stop()
            self.get_logger().warning(
                f"No command for {elapsed:.1f}s - stopping"
            )


def ros2_spin_thread(ros2_node):
    rclpy.spin(ros2_node)


def get_argparser():
    available_models = sorted(
        name
        for name in network.modeling.__dict__
        if name.islower() and callable(network.modeling.__dict__[name])
    )

    parser = argparse.ArgumentParser(
        description="Combined BEV + navigation + ROS2 Twist publisher"
    )

    parser.add_argument("--front_cam", type=int, default=2)
    parser.add_argument("--left_cam", type=int, default=4)
    parser.add_argument("--rear_cam", type=int, default=0)
    parser.add_argument("--right_cam", type=int, default=6)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)

    parser.add_argument("--enable_ros2", action="store_true", default=False)
    parser.add_argument(
        "--cmd_vel_topic",
        type=str,
        default="/cmd_vel_1",
        help="ROS2 Twist topic to publish navigation commands",
    )

    parser.add_argument(
        "--dataset", type=str, default="cityscapes", choices=["voc", "cityscapes", "custom"]
    )
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--save_output", action="store_true", default=False)
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=False,
        help="Save BEV and overlay frames as PNG images under the output folder",
    )
    parser.add_argument(
        "--save_image_interval",
        type=int,
        default=1,
        help="Save one image every N processed frames when --save_images is enabled",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--record_fps",
        type=int,
        default=10,
        help="FPS for saved output video; lower values reduce recording load",
    )
    parser.add_argument(
        "--record_every",
        type=int,
        default=1,
        help="Record one frame every N processed frames",
    )
    parser.add_argument(
        "--record_queue_size",
        type=int,
        default=2,
        help="Async recording queue size; small queues avoid control latency",
    )
    parser.add_argument(
        "--cmd_vel_publish_hz",
        type=float,
        default=30.0,
        help="Rate for re-publishing the latest /cmd_vel_1 command",
    )
    parser.add_argument(
        "--disable_steering_filter",
        action="store_true",
        default=False,
        help="Disable low-pass/deadband/rate-limit filtering before publishing cmd_vel_1",
    )
    parser.add_argument(
        "--steering_filter_alpha",
        type=float,
        default=DEFAULT_STEERING_FILTER_ALPHA,
        help="Low-pass alpha for steering command. Higher is smoother but slower",
    )
    parser.add_argument(
        "--steering_deadband_deg",
        type=float,
        default=DEFAULT_STEERING_DEADBAND_DEG,
        help="Ignore filtered steering changes smaller than this many degrees",
    )
    parser.add_argument(
        "--steering_max_rate_deg_s",
        type=float,
        default=DEFAULT_STEERING_MAX_RATE_DEG_S,
        help="Maximum steering command change rate in degrees per second; <=0 disables",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 inference on CUDA",
    )
    parser.add_argument(
        "--inference_width",
        "--model_input_width",
        dest="inference_width",
        type=int,
        default=0,
        help="Resize BEV to this width before segmentation inference; 0 keeps original size",
    )
    parser.add_argument(
        "--inference_height",
        "--model_input_height",
        dest="inference_height",
        type=int,
        default=0,
        help="Resize BEV to this height before segmentation inference; 0 keeps original size",
    )
    parser.add_argument(
        "--vehicle_width",
        type=int,
        default=ROI_W,
        help="Vehicle ROI width in pixels for drivable-width checks",
    )
    parser.add_argument(
        "--vehicle_height",
        type=int,
        default=ROI_H,
        help="Vehicle ROI height in pixels for drivable-width checks",
    )
    parser.add_argument(
        "--async_capture",
        action="store_true",
        default=False,
        help="Capture BEV frames in a background thread and process the latest frame",
    )
    parser.add_argument("--show_preview", action="store_true", default=False)
    parser.add_argument(
        "--low_latency_mode",
        action="store_true",
        default=False,
        help="Disable expensive visualization work and favor newest data",
    )
    parser.add_argument("--model", type=str, default="deeplabv3plus_mobilenet", choices=available_models)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--conf_thresh", type=float, default=0.7)
    parser.add_argument(
        "--lookahead_distance",
        type=int,
        default=200,
        help="Compatibility option from saty3.py; saty4.py uses path tracking instead",
    )
    parser.add_argument(
        "--target_lookahead_m",
        type=float,
        default=2.5,
        help="Pure pursuit target distance from vehicle to target point in meters",
    )
    parser.add_argument(
        "--min_nav_path_points",
        type=int,
        default=DEFAULT_MIN_NAV_PATH_POINTS,
        help=(
            "Minimum tracked nav_path points required to move. "
            "Fewer points publish linear.x=0.0 as an obstacle/blocked-road case"
        ),
    )
    parser.add_argument(
        "--spline_curvature_threshold",
        type=float,
        default=DEFAULT_SPLINE_CURVATURE_THRESHOLD,
        help=(
            "Mean spline curvature threshold. At or below this value, "
            "linear.x uses --straight_linear_x; above it uses --curved_linear_x"
        ),
    )
    parser.add_argument(
        "--straight_linear_x",
        type=float,
        default=DEFAULT_STRAIGHT_LINEAR_X,
        help="linear.x command when the spline is close to straight",
    )
    parser.add_argument(
        "--curved_linear_x",
        type=float,
        default=DEFAULT_CURVED_LINEAR_X,
        help="linear.x command when the spline is more curved",
    )
    parser.add_argument(
        "--drivable_class_ids",
        type=int,
        nargs="+",
        default=DEFAULT_DRIVABLE_CLASS_IDS,
        help="Class IDs treated as drivable area",
    )
    return parser


def build_model(opts, device):
    if opts.dataset.lower() == "voc":
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == "cityscapes":
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    else:
        opts.num_classes = 9
        decode_fn = Cityscapes.decode_target

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
    ).to(device)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=device, weights_only=False)
        key = (
            "model_state"
            if "model_state" in checkpoint
            else "state_dict"
            if "state_dict" in checkpoint
            else None
        )
        if key:
            model.load_state_dict(checkpoint[key])
        elif isinstance(checkpoint, dict) and "module" in str(list(checkpoint.keys())[0]):
            clean_state = OrderedDict(
                (k[7:] if k.startswith("module.") else k, v)
                for k, v in checkpoint.items()
            )
            model.load_state_dict(clean_state)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {opts.ckpt}")
    elif opts.ckpt:
        print(f"Warning: checkpoint not found: {opts.ckpt}")

    if device.type == "cuda" and opts.fp16:
        model = model.half()

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    model.eval()

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform, decode_fn


def warmup_model(model, device, opts):
    if device.type != "cuda":
        return
    infer_w = opts.inference_width if opts.inference_width > 0 else opts.display_width
    infer_h = opts.inference_height if opts.inference_height > 0 else opts.display_height
    dtype = torch.float16 if opts.fp16 else torch.float32
    dummy = torch.zeros((1, 3, infer_h, infer_w), device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(3):
            model(dummy)
    torch.cuda.synchronize()


class LatestBEVCapture:
    def __init__(self, bev_processor, include_display=False):
        self.bev_processor = bev_processor
        self.include_display = include_display
        self.latest_frame = None
        self.latest_display = None
        self.latest_id = 0
        self.running = False
        self.thread = None
        self.condition = threading.Condition()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            frame, display = self.bev_processor.get_bev_frame(
                include_display=self.include_display
            )
            if frame is None:
                time.sleep(0.002)
                continue
            with self.condition:
                self.latest_frame = frame
                self.latest_display = display
                self.latest_id += 1
                self.condition.notify_all()

    def get_latest(self, last_id, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.latest_id != last_id or not self.running,
                timeout=timeout,
            )
            return self.latest_frame, self.latest_display, self.latest_id

    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size, queue_size=2):
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.writer.write(frame)
            self.queue.task_done()

    def release(self):
        self.running = False
        self.thread.join(timeout=3.0)
        self.writer.release()

class LatestCommandPublisher:
    def __init__(self, ros2_node):
        self.ros2_node = ros2_node
        self.cmd_queue = queue.Queue(maxsize=1)
        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.thread.start()

    # API ใหม่
    def publish(self, steering_deg, linear_x=1.0):
        try:
            # เก็บเฉพาะคำสั่งล่าสุด
            if self.cmd_queue.full():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    pass

            self.cmd_queue.put_nowait(
                (float(steering_deg), float(linear_x))
            )

        except Exception:
            pass

    # API เดิม เพื่อให้โค้ดเก่าใช้งานได้
    def publish_cmd_vel(self, steering_deg, linear_x=1.0):
        self.publish(
            steering_deg,
            linear_x
        )

    # ส่งต่อ status ไปยัง ROS2 node จริง
    def publish_status(self, msg):
        self.ros2_node.publish_status(msg)

    def _worker(self):
        while self.running:

            try:
                steering_deg, linear_x = self.cmd_queue.get(
                    timeout=0.05
                )

                self.ros2_node.publish_cmd_vel(
                    steering_deg,
                    linear_x=linear_x
                )

                self.cmd_queue.task_done()

            except queue.Empty:
                continue

            except Exception as e:
                print(
                    f"[ROS CMD QUEUE ERROR] {e}"
                )

    def stop(self):
        self.running = False

        # ปลุก thread ถ้ากำลังรอ queue
        try:
            self.cmd_queue.put_nowait((0.0, 0.0))
        except Exception:
            pass

        self.thread.join(timeout=1.0)


def clamp(value, low, high):
    return max(low, min(high, value))


def calculate_spline_curvature(spline_points):
    pts = np.asarray(spline_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return {
            "mean": 0.0,
            "max": 0.0,
            "score": 0.0,
            "total_heading_change_deg": 0.0,
        }

    p_prev = pts[:-2]
    p_curr = pts[1:-1]
    p_next = pts[2:]

    a = np.linalg.norm(p_curr - p_prev, axis=1)
    b = np.linalg.norm(p_next - p_curr, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)
    cross = np.abs(
        (p_curr[:, 0] - p_prev[:, 0]) * (p_next[:, 1] - p_prev[:, 1])
        - (p_curr[:, 1] - p_prev[:, 1]) * (p_next[:, 0] - p_prev[:, 0])
    )
    denom = a * b * c
    valid = denom > 1e-6
    curvature = np.zeros_like(denom)
    curvature[valid] = 2.0 * cross[valid] / denom[valid]

    segment_vectors = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_segments = segment_lengths > 1e-6
    heading_change = np.array([], dtype=np.float64)
    if np.count_nonzero(valid_segments) >= 2:
        headings = np.arctan2(
            segment_vectors[valid_segments, 1],
            segment_vectors[valid_segments, 0],
        )
        heading_change = np.abs(np.diff(np.unwrap(headings)))

    mean_curvature = float(np.mean(curvature[valid])) if np.any(valid) else 0.0
    max_curvature = float(np.max(curvature[valid])) if np.any(valid) else 0.0
    total_heading_change_deg = (
        float(np.degrees(np.sum(heading_change))) if len(heading_change) > 0 else 0.0
    )
    return {
        "mean": mean_curvature,
        "max": max_curvature,
        "score": mean_curvature,
        "total_heading_change_deg": total_heading_change_deg,
    }


def adjust_linear_x_by_spline_curvature(
    base_linear_x,
    curvature_score,
    curvature_threshold=DEFAULT_SPLINE_CURVATURE_THRESHOLD,
    straight_linear_x=DEFAULT_STRAIGHT_LINEAR_X,
    curved_linear_x=DEFAULT_CURVED_LINEAR_X,
):
    base_linear_x = float(base_linear_x)
    if base_linear_x <= 0.0:
        return 0.0

    straight_linear_x = clamp(float(straight_linear_x), 0.0, 1.0)
    curved_linear_x = clamp(float(curved_linear_x), 0.0, 1.0)
    curvature_threshold = max(float(curvature_threshold), 1e-9)
    return straight_linear_x if curvature_score <= curvature_threshold else curved_linear_x


def split_contiguous_runs(xs):
    if len(xs) == 0:
        return []

    breaks = np.where(np.diff(xs) > 1)[0] + 1
    return np.split(xs, breaks)


def build_tracked_nav_points(mask, vehicle_center):
    height, width = mask.shape
    start_x = int(np.clip(vehicle_center[0], 0, width - 1))
    start_y = int(np.clip(vehicle_center[1], 0, height - 1))

    nav_points = []
    current_x = start_x

    # Path tracking: keep following the closest lane segment from the last x.
    for y in range(start_y, -1, -1):
        xs = np.where(mask[y, :] > 0)[0]
        runs = split_contiguous_runs(xs)
        if not runs:
            continue

        def run_distance(run):
            left = int(run[0])
            right = int(run[-1])
            if left <= current_x <= right:
                return 0
            return min(abs(current_x - left), abs(current_x - right))

        selected_run = min(runs, key=run_distance)
        current_x = int((int(selected_run[0]) + int(selected_run[-1])) / 2)
        nav_points.append((current_x, y))

    return nav_points


def process_frame(
    frame,
    model,
    transform,
    device,
    decode_fn,
    conf_thresh=0.7,
    lookahead_distance=200,
    target_lookahead_m=2.5,
    ros2_node=None,
    drivable_class_ids=None,
    inference_size=None,
    vehicle_size=None,
    use_fp16=False,
    render_overlay=True,
    spline_curvature_threshold=DEFAULT_SPLINE_CURVATURE_THRESHOLD,
    straight_linear_x=DEFAULT_STRAIGHT_LINEAR_X,
    curved_linear_x=DEFAULT_CURVED_LINEAR_X,
    min_nav_path_points=DEFAULT_MIN_NAV_PATH_POINTS,
):
    global prev_spline_points

    input_frame = frame
    if inference_size:
        infer_w, infer_h = inference_size
        if infer_w > 0 and infer_h > 0:
            input_frame = cv2.resize(
                frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA
            )

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(PILImage.fromarray(frame_rgb)).unsqueeze(0).to(device)
    if use_fp16 and device.type == "cuda":
        input_tensor = input_tensor.half()

    with torch.inference_mode():
        logits = model(input_tensor)
        pred_gpu = torch.argmax(logits, dim=1)[0]
        pred = pred_gpu.cpu().numpy().astype(np.int64)
    if pred.shape[:2] != frame.shape[:2]:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    if render_overlay:
        colorized = cv2.cvtColor(decode_fn(pred).astype("uint8"), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 0.35, colorized, 0.65, 0)
    else:
        overlay = None

    height, width = pred.shape
    final_mask = clean_drivable_mask(pred, drivable_class_ids)

    if render_overlay:
        vehicle_center, heading_vector, roi_rect = draw_vehicle_roi(
            overlay, height, width, vehicle_size
        )
    else:
        vehicle_center, heading_vector, _, roi_rect = detect_vehicle_and_heading(
            None, height, width, vehicle_size
        )
    linear_x, drivable_width_px, cmp_roi = compute_roi_linear_x(final_mask, roi_rect)

    nav_points = build_tracked_nav_points(final_mask, vehicle_center)
    nav_point_count = len(nav_points)
    min_nav_path_points = max(0, int(min_nav_path_points))

    spline_points = []
    interesting_point = None
    steering_deg = 0.0
    steering_rad = 0.0
    angle_diff = 0.0
    target_distance_m = 0.0
    curvature_metrics = calculate_spline_curvature(spline_points)

    if nav_point_count < min_nav_path_points:
        linear_x = 0.0
        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(0.0, linear_x=0.0)
            ros2_node.publish_status(
                "NAV_PATH_TOO_SHORT: "
                f"nav_points={nav_point_count} < {min_nav_path_points}"
            )

    elif len(nav_points) >= 3:
        pts = np.array(nav_points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        t = np.insert(np.cumsum(dist), 0, 0)
        t_new = np.linspace(t[0], t[-1], 200)
        spline_points = np.stack(
            (CubicSpline(t, pts[:, 0])(t_new), CubicSpline(t, pts[:, 1])(t_new)),
            axis=-1,
        ).astype(int)
        if prev_spline_points is not None and len(prev_spline_points) == len(
            spline_points
        ):
            spline_points = (
                alpha_smooth * prev_spline_points
                + (1 - alpha_smooth) * spline_points
            ).astype(int)
        prev_spline_points = spline_points
        curvature_metrics = calculate_spline_curvature(spline_points)

        if render_overlay:
            for i in range(len(spline_points) - 1):
                cv2.line(
                    overlay,
                    tuple(spline_points[i]),
                    tuple(spline_points[i + 1]),
                    (255, 255, 0),
                    2,
                )

        interesting_point, target_distance_m = select_target_point(
            spline_points, vehicle_center, target_lookahead_m
        )
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                "Target",
                (interesting_point[0] + 10, interesting_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx)
            - np.arctan2(-heading_vector[1], heading_vector[0])
        )
        if angle_diff > 180:
            angle_diff -= 360
        if angle_diff < -180:
            angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE
        )

        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect
        )
        height_linear_x, drivable_height_px, height_cmp_roi = check_drivable_height_ahead(
            final_mask, roi_rect
        )
        if height_linear_x <= 0.0:
            linear_x = 0.0
            cmp_roi = height_cmp_roi
        linear_x = adjust_linear_x_by_spline_curvature(
            linear_x,
            curvature_metrics["score"],
            spline_curvature_threshold,
            straight_linear_x,
            curved_linear_x,
        )

        if render_overlay:
            cv2.putText(
                overlay,
                f"Alpha: {angle_diff:.1f} deg",
                (vehicle_center[0] - 80, vehicle_center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f} deg | "
                f"linear_x={linear_x:.1f} | "
                f"spline_curvature={curvature_metrics['score']:.5f} | "
                f"drivable_width_px={drivable_width_px} | "
                f"drivable_height_px={drivable_height_px} | "
                f"nav_points={nav_point_count}"
            )

    elif len(nav_points) > 0:
        interesting_point, target_distance_m = select_target_point(
            nav_points, vehicle_center, target_lookahead_m
        )
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx)
            - np.arctan2(-heading_vector[1], heading_vector[0])
        )
        if angle_diff > 180:
            angle_diff -= 360
        if angle_diff < -180:
            angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE
        )
        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect
        )
        height_linear_x, drivable_height_px, height_cmp_roi = check_drivable_height_ahead(
            final_mask, roi_rect
        )
        if height_linear_x <= 0.0:
            linear_x = 0.0
            cmp_roi = height_cmp_roi

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f} deg | "
                f"linear_x={linear_x:.1f} | "
                f"drivable_height_px={drivable_height_px} | "
                f"nav_points={nav_point_count}"
            )

    else:
        linear_x = 0.0
        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(0.0, linear_x=0.0)
            ros2_node.publish_status("NO_PATH_FOUND")

    if render_overlay:
        draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px)

        h_orig, w_orig = frame.shape[:2]
        info_lines = [
            f"PURE PURSUIT  [{MODE_LABEL}]",
            f"Steering: {steering_deg:+.4f} deg  ({steering_rad:+.4f} rad)",
            f"Target Ld: {target_distance_m:.2f} m / set {target_lookahead_m:.2f} m",
            f"Nav points: {nav_point_count} / min {min_nav_path_points}",
            f"Curvature: {curvature_metrics['score']:.5f}",
            f"linear.x: {'GO' if linear_x > 0.0 else 'STOP'} ({linear_x:.1f})",
        ]
        box_h = 25 * len(info_lines) + 10
        box_w = 420
        bx, by = w_orig - box_w - 10, 10
        sub = overlay[by : by + box_h, bx : bx + box_w]
        white = np.ones(sub.shape, dtype=np.uint8) * 255
        overlay[by : by + box_h, bx : bx + box_w] = cv2.addWeighted(
            sub, 0.4, white, 0.6, 1.0
        )
        for i, line in enumerate(info_lines):
            color = (0, 0, 0)
            if i == 5:
                color = (0, 150, 0) if linear_x > 0.0 else (0, 0, 180)
            cv2.putText(
                overlay,
                line,
                (bx + 8, by + 20 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return (
        overlay if render_overlay else frame,
        pred,
        spline_points.tolist()
        if isinstance(spline_points, np.ndarray)
        else spline_points,
        vehicle_center,
        interesting_point,
    )


def main():
    opts = get_argparser().parse_args()

    # ==========================================================
    # LOW LATENCY MODE
    # ==========================================================
    if opts.low_latency_mode:
        opts.show_preview = False

        if opts.record_queue_size > 1:
            opts.record_queue_size = 1

        if opts.record_fps > 10:
            opts.record_fps = 10

        if opts.record_every < 3:
            opts.record_every = 3

        if opts.inference_width == 0 and opts.inference_height == 0:
            opts.inference_width = 320
            opts.inference_height = 180

        print(
            "[LOW LATENCY MODE] "
            f"inference={opts.inference_width}x{opts.inference_height} "
            f"record_every={opts.record_every} "
            f"record_fps={opts.record_fps} "
            "preview=OFF"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    if torch.cuda.is_available():
        cudnn.benchmark = True
    device = select_device()
    inference_size = None
    if opts.inference_width > 0 and opts.inference_height > 0:
        inference_size = (opts.inference_width, opts.inference_height)
    elif opts.inference_width > 0 or opts.inference_height > 0:
        raise ValueError(
            "Set both --model_input_width and --model_input_height "
            "or leave both as 0."
        )
    vehicle_size = (
        max(1, int(opts.vehicle_width)),
        max(1, int(opts.vehicle_height)),
    )
    model, transform, decode_fn = build_model(opts, device)
    warmup_model(model, device, opts)

    ros2_node_real = None
    ros_cmd_pub = None
    ros2_thread = None
    if opts.enable_ros2:
        if not ROS2_AVAILABLE:
            print(f"ROS2 is not available. Cannot publish {opts.cmd_vel_topic}.")
            return
        rclpy.init()
        ros2_node_real = CombinedVehicleControllerNode(
            cmd_vel_topic=opts.cmd_vel_topic,
            vehicle_config={
                "steering_filter_enabled": not opts.disable_steering_filter,
                "steering_filter_alpha": opts.steering_filter_alpha,
                "steering_deadband_deg": opts.steering_deadband_deg,
                "steering_max_rate_deg_s": opts.steering_max_rate_deg_s,
            },
            cmd_vel_publish_hz=opts.cmd_vel_publish_hz,
        )
        ros2_thread = threading.Thread(
            target=ros2_spin_thread,
            args=(ros2_node_real,),
            daemon=True,
        )
        ros2_thread.start()

    if ros2_node_real is not None:
        ros_cmd_pub = LatestCommandPublisher(
            ros2_node_real
        )

    video_paths = {
        "front": opts.front_cam,
        "left": opts.left_cam,
        "rear": opts.rear_cam,
        "right": opts.right_cam,
    }

    bev_processor = BEVProcessor(
        video_paths,
        img_car=img_car if BEV_AVAILABLE else None,
        display_width=opts.display_width,
        display_height=opts.display_height,
    )
    capture_worker = None
    if opts.async_capture:
        capture_worker = LatestBEVCapture(
            bev_processor, include_display=opts.show_preview
        )
        capture_worker.start()

    overlay_writer = None
    bev_writer = None
    if opts.save_output or opts.save_images:
        os.makedirs(opts.output, exist_ok=True)
    if opts.save_images:
        os.makedirs(os.path.join(opts.output, "bev_frames"), exist_ok=True)
        os.makedirs(os.path.join(opts.output, "overlay_frames"), exist_ok=True)

    print("=" * 60)
    print("COMBINED BEV NAVIGATION")
    print(f"  Cameras       : {video_paths}")
    print(f"  Device        : {device} | Mode: {MODE_LABEL}")
    print(f"  Model         : {opts.model}")
    if inference_size:
        print(f"  Inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print("  Inference size: original BEV")
    print(f"  Vehicle ROI   : {vehicle_size[0]}x{vehicle_size[1]} px")
    print(f"  Min nav points: {max(0, int(opts.min_nav_path_points))}")
    print(f"  FP16          : {'enabled' if opts.fp16 and device.type == 'cuda' else 'disabled'}")
    print(f"  Async capture : {'enabled' if capture_worker else 'disabled'}")
    print(f"  ROS2          : {'enabled' if ros2_node_real else 'disabled'}")
    print(f"  cmd_vel_1 topic : {opts.cmd_vel_topic}")
    if ros2_node_real:
        print(f"  cmd_vel_1 rate  : {opts.cmd_vel_publish_hz:.1f} Hz")
    print("=" * 60)

    frame_count = 0
    last_capture_id = 0
    interval = 1.0 / max(opts.fps, 1)
    # render_overlay = opts.show_preview or opts.save_output or opts.save_images
    render_overlay = (
        opts.show_preview
        or opts.save_images
    )
    record_every = max(opts.record_every, 1)
    save_image_interval = max(opts.save_image_interval, 1)

    try:
        while True:
            t_start = time.time()
            if capture_worker is not None:
                bev_frame, cam_display, last_capture_id = capture_worker.get_latest(
                    last_capture_id
                )
            else:
                bev_frame, cam_display = bev_processor.get_bev_frame(
                    include_display=opts.show_preview
                )
            if bev_frame is None:
                time.sleep(0.01)
                continue

            result, _, _, _, _ = process_frame(
                bev_frame,
                model,
                transform,
                device,
                decode_fn,
                conf_thresh=opts.conf_thresh,
                lookahead_distance=opts.lookahead_distance,
                target_lookahead_m=opts.target_lookahead_m,
                ros2_node=ros_cmd_pub,
                drivable_class_ids=opts.drivable_class_ids,
                inference_size=inference_size,
                vehicle_size=vehicle_size,
                use_fp16=opts.fp16,
                render_overlay=render_overlay,
                spline_curvature_threshold=opts.spline_curvature_threshold,
                straight_linear_x=opts.straight_linear_x,
                curved_linear_x=opts.curved_linear_x,
                min_nav_path_points=opts.min_nav_path_points,
            )

            if opts.save_output and frame_count % record_every == 0:
                if overlay_writer is None:
                    h, w = result.shape[:2]
                    overlay_path = os.path.join(opts.output, "combined_nav_output.mp4")
                    overlay_writer = AsyncVideoWriter(
                        overlay_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        opts.record_fps,
                        (w, h),
                        queue_size=opts.record_queue_size,
                    )
                    print(f"Recording overlay to: {overlay_path}")
                overlay_writer.write(result)

                if bev_writer is None:
                    h_b, w_b = bev_frame.shape[:2]
                    bev_path = os.path.join(opts.output, "combined_bev_output.mp4")
                    bev_writer = AsyncVideoWriter(
                        bev_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        opts.record_fps,
                        (w_b, h_b),
                        queue_size=opts.record_queue_size,
                    )
                    print(f"Recording BEV to: {bev_path}")
                bev_writer.write(bev_frame)

            if opts.save_images and frame_count % save_image_interval == 0:
                frame_name = f"{frame_count:06d}.png"
                cv2.imwrite(
                    os.path.join(opts.output, "bev_frames", frame_name),
                    bev_frame,
                )
                cv2.imwrite(
                    os.path.join(opts.output, "overlay_frames", frame_name),
                    result,
                )

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

            if opts.show_preview:
                h_r, w_r = result.shape[:2]
                cv2.imshow(
                    "Combined Navigation",
                    cv2.resize(result, (w_r // 2, h_r // 2)),
                )
                if cam_display is not None:
                    h_c, w_c = cam_display.shape[:2]
                    cv2.imshow(
                        "Cameras",
                        cv2.resize(cam_display, (w_c // 2, h_c // 2)),
                    )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.time() - t_start
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if ros_cmd_pub is not None:
            ros_cmd_pub.stop()
        if ros2_node_real is not None:
            ros2_node_real.stop()
        if capture_worker is not None:
            capture_worker.stop()
        if overlay_writer is not None:
            overlay_writer.release()
        if bev_writer is not None:
            bev_writer.release()
        bev_processor.release()
        cv2.destroyAllWindows()
        if ros2_node_real is not None:
            ros2_node_real.destroy_node()
            rclpy.shutdown()
        print(f"Stopped. Total frames: {frame_count}")


if __name__ == "__main__":
    main()

# ============================================================
#  sep_com_combined_timing.py
#  Single-computer pipeline with per-frame timing log:
#  cameras -> BEV -> segmentation/navigation -> /cmd_vel
# ============================================================

import argparse
import csv
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime

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
    ROS2_AVAILABLE,
    SAFETY_CHECK_INTERVAL,
    SAFETY_TIMEOUT,
    WHEELBASE,
    alpha_smooth,
    calculate_steering_angle,
    check_drivable_width_ahead,
    clean_drivable_mask,
    compute_roi_linear_x,
    draw_compare_roi,
    draw_vehicle_roi,
    select_device,
    select_target_point,
)

if ROS2_AVAILABLE:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
else:
    Node = object
    Twist = None
    Odometry = None
    String = None


TIMING_FIELDS = [
    "frame_index",
    "wall_time",
    "loop_start_ts",
    "loop_end_ts",
    "loop_duration_ms",
    "capture_start_ts",
    "capture_end_ts",
    "capture_duration_ms",
    "preprocess_start_ts",
    "preprocess_end_ts",
    "preprocess_duration_ms",
    "inference_start_ts",
    "inference_end_ts",
    "inference_duration_ms",
    "postprocess_start_ts",
    "postprocess_end_ts",
    "postprocess_duration_ms",
    "navigation_start_ts",
    "navigation_end_ts",
    "navigation_duration_ms",
    "ros_publish_start_ts",
    "ros_publish_end_ts",
    "ros_publish_duration_ms",
    "save_output_duration_ms",
    "preview_duration_ms",
    "sleep_duration_ms",
    "actual_fps",
    "bev_width",
    "bev_height",
    "nav_points_count",
    "spline_points_count",
    "target_x",
    "target_y",
    "target_distance_m",
    "angle_diff_deg",
    "steering_deg",
    "steering_rad",
    "steering_delta_deg",
    "steering_change_dt_s",
    "steering_rate_deg_s",
    "linear_x",
    "angular_z",
    "drivable_width_px",
    "command_published",
    "path_found",
]


prev_spline_points = None
STEERING_CHANGE_EPSILON_DEG = 1e-3


def now_ts():
    return time.time()


def ms(start_ts, end_ts):
    if start_ts is None or end_ts is None:
        return ""
    return (end_ts - start_ts) * 1000.0


def add_timing_mark(timing_marks, name):
    if timing_marks is not None:
        timing_marks.append((name, time.perf_counter_ns()))


def add_timing_mark_at(timing_marks, name, timestamp_ns):
    if timing_marks is not None and timestamp_ns is not None:
        timing_marks.append((name, int(timestamp_ns)))


def update_steering_change_timing(timing_marks, steering_state, steering_deg):
    if steering_state is None:
        return None

    current_ns = time.perf_counter_ns()
    previous_deg = steering_state.get("last_angle_deg")
    previous_change_ns = steering_state.get("last_change_ns")
    changed = (
        previous_deg is None
        or abs(float(steering_deg) - float(previous_deg)) > STEERING_CHANGE_EPSILON_DEG
    )

    metrics = {
        "steering_deg": float(steering_deg),
        "previous_steering_deg": previous_deg if previous_deg is not None else "",
        "steering_delta_deg": (
            "" if previous_deg is None else float(steering_deg) - float(previous_deg)
        ),
        "steering_changed": bool(changed),
        "steering_change_interval_ms": "",
    }

    if changed:
        if previous_change_ns is not None:
            add_timing_mark_at(timing_marks, "previous_steering_change", previous_change_ns)
            metrics["steering_change_interval_ms"] = (
                current_ns - previous_change_ns
            ) / 1_000_000.0
        add_timing_mark_at(timing_marks, "steering_change_done", current_ns)
        steering_state["last_change_ns"] = current_ns

    steering_state["last_angle_deg"] = float(steering_deg)
    return metrics


class TimingCsvLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=TIMING_FIELDS)
        self._writer.writeheader()

    def write(self, row):
        clean_row = {field: row.get(field, "") for field in TIMING_FIELDS}
        self._writer.writerow(clean_row)
        self._file.flush()

    def close(self):
        self._file.close()


class TimingMarkLogger:
    def __init__(self, output_dir, log_path, flush_every=100):
        self.enabled = bool(log_path)
        self.path = None
        self.file = None
        self.buffer = []
        self.flush_every = max(1, int(flush_every))

        if not self.enabled:
            return

        self.path = (
            log_path
            if os.path.isabs(log_path)
            else os.path.join(output_dir or ".", log_path)
        )
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8", buffering=1024 * 1024)
        print(f"Timing marks: {self.path}")

    def record(self, frame_id, timing_marks, extra_metrics=None):
        if not self.enabled or self.file is None or timing_marks is None:
            return

        wall_time_ns = time.time_ns()
        perf_time_ns = time.perf_counter_ns()
        row = {
            "frame": int(frame_id),
            "wall_time_ns": wall_time_ns,
            "perf_to_wall_offset_ns": wall_time_ns - perf_time_ns,
            "perf_marks_ns": {name: int(ts) for name, ts in timing_marks},
        }
        if extra_metrics:
            row["metrics"] = extra_metrics
        self.buffer.append(json.dumps(row, separators=(",", ":")) + "\n")
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.file is not None and self.buffer:
            self.file.writelines(self.buffer)
            self.buffer = []
            self.file.flush()

    def close(self):
        self.flush()
        if self.file is not None:
            self.file.close()
            self.file = None


class TimedVehicleControllerNode(Node):
    def __init__(
        self,
        cmd_vel_topic="/cmd_vel",
        vehicle_config=None,
        cmd_vel_publish_hz=30.0,
    ):
        super().__init__("timed_vehicle_controller")
        self.config = vehicle_config or {}
        self.cmd_vel_topic = cmd_vel_topic
        self.wheelbase = self.config.get("wheelbase", WHEELBASE)
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
        self.odom_subscriber = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.current_position = {"x": 0.0, "y": 0.0, "theta": 0.0}
        self.last_cmd_time = time.time()
        self.last_steering_deg = None
        self.last_steering_time = None
        self.safety_timer = self.create_timer(
            SAFETY_CHECK_INTERVAL, self.safety_check
        )
        self.cmd_vel_timer = self.create_timer(
            1.0 / self.cmd_vel_publish_hz, self.publish_latest_cmd
        )
        self.get_logger().info(
            f"Timed vehicle controller publishing Twist to "
            f"{self.cmd_vel_topic} at {self.cmd_vel_publish_hz:.1f} Hz"
        )

    def odom_callback(self, msg):
        self.current_position["x"] = msg.pose.pose.position.x
        self.current_position["y"] = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        self.current_position["theta"] = np.arctan2(
            2 * (o.w * o.z + o.x * o.y),
            1 - 2 * (o.y**2 + o.z**2),
        )

    def publish_cmd_vel(self, steering_angle_deg, linear_x=1.0):
        publish_start = now_ts()
        steering_angle_deg = float(steering_angle_deg)
        steering_angle_rad = float(np.radians(steering_angle_deg))
        linear_x = float(np.clip(linear_x, 0.0, 1.0))

        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = steering_angle_rad
        with self.cmd_lock:
            self.latest_cmd = cmd
            self.latest_cmd_time = publish_start
        self.cmd_vel_publisher.publish(cmd)

        publish_end = now_ts()
        steering_delta_deg = ""
        steering_change_dt_s = ""
        steering_rate_deg_s = ""
        if self.last_steering_deg is not None and self.last_steering_time is not None:
            steering_delta_deg = steering_angle_deg - self.last_steering_deg
            steering_change_dt_s = publish_end - self.last_steering_time
            if steering_change_dt_s > 0:
                steering_rate_deg_s = steering_delta_deg / steering_change_dt_s

        self.last_steering_deg = steering_angle_deg
        self.last_steering_time = publish_end
        self.last_cmd_time = publish_end

        return {
            "ros_publish_start_ts": publish_start,
            "ros_publish_end_ts": publish_end,
            "ros_publish_duration_ms": ms(publish_start, publish_end),
            "steering_deg": steering_angle_deg,
            "steering_rad": steering_angle_rad,
            "steering_delta_deg": steering_delta_deg,
            "steering_change_dt_s": steering_change_dt_s,
            "steering_rate_deg_s": steering_rate_deg_s,
            "linear_x": linear_x,
            "angular_z": steering_angle_rad,
            "command_published": 1,
        }

    def publish_latest_cmd(self):
        with self.cmd_lock:
            cmd = self.latest_cmd
            cmd_age = time.time() - self.latest_cmd_time
        if cmd_age > SAFETY_TIMEOUT:
            cmd = Twist()
        self.cmd_vel_publisher.publish(cmd)

    def stop(self):
        stop_cmd = Twist()
        with self.cmd_lock:
            self.latest_cmd = stop_cmd
            self.latest_cmd_time = time.time()
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
        description="Combined BEV + navigation + ROS2 Twist publisher with timing CSV"
    )
    parser.add_argument("--front_cam", type=int, default=2)
    parser.add_argument("--left_cam", type=int, default=4)
    parser.add_argument("--rear_cam", type=int, default=0)
    parser.add_argument("--right_cam", type=int, default=6)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)

    parser.add_argument("--enable_ros2", action="store_true", default=False)
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")

    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["voc", "cityscapes", "custom"])
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
        help="Rate for re-publishing the latest /cmd_vel command",
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
        "--async_capture",
        action="store_true",
        default=False,
        help="Capture BEV frames in a background thread and process the latest frame",
    )
    parser.add_argument("--show_preview", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="deeplabv3plus_mobilenet", choices=available_models)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--conf_thresh", type=float, default=0.7)
    parser.add_argument("--lookahead_distance", type=int, default=200)
    parser.add_argument("--target_lookahead_m", type=float, default=2.5)
    parser.add_argument("--drivable_class_ids", type=int, nargs="+", default=DEFAULT_DRIVABLE_CLASS_IDS)
    parser.add_argument(
        "--timing_csv",
        type=str,
        default="timing_log.csv",
        help="CSV path for per-frame timestamp and duration log. Relative paths are saved under --output.",
    )
    parser.add_argument(
        "--timing_log",
        type=str,
        default="combined_timing_marks.jsonl",
        help="Raw timing mark JSONL path. Relative paths are saved under --output. Use empty string to disable.",
    )
    parser.add_argument(
        "--timing_flush_every",
        type=int,
        default=100,
        help="Number of frames to buffer before writing timing marks to disk",
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


def process_frame_timed(
    frame,
    model,
    transform,
    device,
    decode_fn,
    opts,
    ros2_node=None,
    timing_marks=None,
    steering_timing_state=None,
):
    global prev_spline_points

    timing = {
        "bev_width": frame.shape[1],
        "bev_height": frame.shape[0],
        "command_published": 0,
        "path_found": 0,
    }

    add_timing_mark(timing_marks, "process_start")
    timing["preprocess_start_ts"] = now_ts()
    input_frame = frame
    if opts.inference_width > 0 and opts.inference_height > 0:
        input_frame = cv2.resize(
            frame,
            (opts.inference_width, opts.inference_height),
            interpolation=cv2.INTER_AREA,
        )

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(PILImage.fromarray(frame_rgb)).unsqueeze(0).to(device)
    if opts.fp16 and device.type == "cuda":
        input_tensor = input_tensor.half()
    timing["preprocess_end_ts"] = now_ts()
    add_timing_mark(timing_marks, "preprocess_done")
    timing["preprocess_duration_ms"] = ms(
        timing["preprocess_start_ts"], timing["preprocess_end_ts"]
    )

    timing["inference_start_ts"] = now_ts()
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
    timing["inference_end_ts"] = now_ts()
    add_timing_mark(timing_marks, "inference_done")
    timing["inference_duration_ms"] = ms(
        timing["inference_start_ts"], timing["inference_end_ts"]
    )

    timing["postprocess_start_ts"] = now_ts()
    colorized = cv2.cvtColor(decode_fn(pred).astype("uint8"), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame, 0.35, colorized, 0.65, 0)
    add_timing_mark(timing_marks, "colorize_done")
    final_mask = clean_drivable_mask(pred, opts.drivable_class_ids)
    add_timing_mark(timing_marks, "mask_done")
    timing["postprocess_end_ts"] = now_ts()
    timing["postprocess_duration_ms"] = ms(
        timing["postprocess_start_ts"], timing["postprocess_end_ts"]
    )

    timing["navigation_start_ts"] = now_ts()
    height, width = pred.shape
    vehicle_center, heading_vector, roi_rect = draw_vehicle_roi(overlay, height, width)
    linear_x, drivable_width_px, cmp_roi = compute_roi_linear_x(final_mask, roi_rect)

    roi_height = height // 3
    roi_mask = final_mask[:roi_height, :]
    nav_points = []
    for y in range(0, roi_height, max(1, roi_height // 5)):
        xs = np.where(roi_mask[y, :] > 0)[0]
        if len(xs) > 0:
            nav_points.append((min(xs[0] + opts.lookahead_distance, xs[-1]), y))
    add_timing_mark(timing_marks, "nav_points_done")

    spline_points = []
    interesting_point = None
    steering_deg = 0.0
    steering_rad = 0.0
    angle_diff = 0.0
    target_distance_m = 0.0

    if len(nav_points) >= 3:
        pts = np.array(nav_points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        t = np.insert(np.cumsum(dist), 0, 0)
        t_new = np.linspace(t[0], t[-1], 200)
        spline_points = np.stack(
            (
                CubicSpline(t, pts[:, 0])(t_new),
                CubicSpline(t, pts[:, 1])(t_new),
            ),
            axis=-1,
        ).astype(int)

        if prev_spline_points is not None and len(prev_spline_points) == len(spline_points):
            spline_points = (
                alpha_smooth * prev_spline_points
                + (1 - alpha_smooth) * spline_points
            ).astype(int)
        prev_spline_points = spline_points

        for i in range(len(spline_points) - 1):
            cv2.line(overlay, tuple(spline_points[i]), tuple(spline_points[i + 1]), (255, 255, 0), 2)

        interesting_point, target_distance_m = select_target_point(
            spline_points, vehicle_center, opts.target_lookahead_m
        )
        timing["path_found"] = 1

    elif len(nav_points) > 0:
        interesting_point, target_distance_m = select_target_point(
            nav_points, vehicle_center, opts.target_lookahead_m
        )
        timing["path_found"] = 1

    if interesting_point is not None:
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
    else:
        linear_x = 0.0

    add_timing_mark(timing_marks, "steering_calc_done")
    steering_metrics = update_steering_change_timing(
        timing_marks, steering_timing_state, steering_deg
    )

    timing["navigation_end_ts"] = now_ts()
    timing["navigation_duration_ms"] = ms(
        timing["navigation_start_ts"], timing["navigation_end_ts"]
    )

    if ros2_node is not None and ROS2_AVAILABLE:
        add_timing_mark(timing_marks, "cmd_vel_publish_start")
        command_metrics = ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
        add_timing_mark(timing_marks, "cmd_vel_publish_done")
        timing.update(command_metrics)
        if interesting_point is not None:
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f} deg | linear_x={linear_x:.1f}"
            )
        else:
            ros2_node.publish_status("NO_PATH_FOUND")
    else:
        timing.update(
            {
                "steering_deg": steering_deg,
                "steering_rad": steering_rad,
                "linear_x": linear_x,
                "angular_z": steering_rad,
            }
        )
    add_timing_mark(timing_marks, "navigation_done")

    draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px)

    h_orig, w_orig = frame.shape[:2]
    info_lines = [
        f"PURE PURSUIT  [{MODE_LABEL}]",
        f"Steering: {steering_deg:+.4f} deg  ({steering_rad:+.4f} rad)",
        f"Target Ld: {target_distance_m:.2f} m / set {opts.target_lookahead_m:.2f} m",
        f"linear.x: {'GO' if linear_x >= 1.0 else 'STOP'} ({linear_x:.1f})",
    ]
    box_h = 25 * len(info_lines) + 10
    box_w = 380
    bx, by = w_orig - box_w - 10, 10
    sub = overlay[by:by + box_h, bx:bx + box_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 255
    overlay[by:by + box_h, bx:bx + box_w] = cv2.addWeighted(sub, 0.4, white, 0.6, 1.0)
    for i, line in enumerate(info_lines):
        color = (0, 0, 0)
        if i == 3:
            color = (0, 150, 0) if linear_x >= 1.0 else (0, 0, 180)
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
    add_timing_mark(timing_marks, "visualize_done")

    timing.update(
        {
            "nav_points_count": len(nav_points),
            "spline_points_count": len(spline_points),
            "target_x": interesting_point[0] if interesting_point is not None else "",
            "target_y": interesting_point[1] if interesting_point is not None else "",
            "target_distance_m": target_distance_m,
            "angle_diff_deg": angle_diff,
            "drivable_width_px": drivable_width_px,
        }
    )
    return overlay, timing, steering_metrics


def main():
    opts = get_argparser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    if torch.cuda.is_available():
        cudnn.benchmark = True
    device = select_device()
    if opts.inference_width > 0 and opts.inference_height > 0:
        inference_size = (opts.inference_width, opts.inference_height)
    elif opts.inference_width > 0 or opts.inference_height > 0:
        raise ValueError(
            "Set both --model_input_width and --model_input_height "
            "or leave both as 0."
        )
    else:
        inference_size = None
    model, transform, decode_fn = build_model(opts, device)
    warmup_model(model, device, opts)

    ros2_node = None
    if opts.enable_ros2:
        if not ROS2_AVAILABLE:
            print(f"ROS2 is not available. Cannot publish {opts.cmd_vel_topic}.")
            return
        rclpy.init()
        ros2_node = TimedVehicleControllerNode(
            cmd_vel_topic=opts.cmd_vel_topic,
            vehicle_config={"wheelbase": WHEELBASE},
            cmd_vel_publish_hz=opts.cmd_vel_publish_hz,
        )
        threading.Thread(target=ros2_spin_thread, args=(ros2_node,), daemon=True).start()

    timing_csv_path = (
        opts.timing_csv
        if os.path.isabs(opts.timing_csv)
        else os.path.join(opts.output, opts.timing_csv)
    )
    timing_logger = TimingCsvLogger(timing_csv_path)
    timing_mark_logger = TimingMarkLogger(
        opts.output, opts.timing_log, opts.timing_flush_every
    )
    steering_timing_state = {
        "last_angle_deg": None,
        "last_change_ns": None,
    }

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
    print("COMBINED BEV NAVIGATION WITH TIMING")
    print(f"  Cameras       : {video_paths}")
    print(f"  Device        : {device} | Mode: {MODE_LABEL}")
    print(f"  Model         : {opts.model}")
    if inference_size:
        print(f"  Inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print("  Inference size: original BEV")
    print(f"  FP16          : {'enabled' if opts.fp16 and device.type == 'cuda' else 'disabled'}")
    print(f"  Async capture : {'enabled' if capture_worker else 'disabled'}")
    print(f"  ROS2          : {'enabled' if ros2_node else 'disabled'}")
    print(f"  cmd_vel topic : {opts.cmd_vel_topic}")
    if ros2_node:
        print(f"  cmd_vel rate  : {opts.cmd_vel_publish_hz:.1f} Hz")
    print(f"  Timing CSV    : {timing_csv_path}")
    print(f"  Timing marks  : {timing_mark_logger.path or 'disabled'}")
    print("=" * 60)

    frame_count = 0
    last_capture_id = 0
    interval = 1.0 / max(opts.fps, 1)

    try:
        while True:
            timing_marks = [] if timing_mark_logger.enabled else None
            add_timing_mark(timing_marks, "loop_start")
            row = {
                "frame_index": frame_count + 1,
                "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            }
            loop_start = now_ts()
            row["loop_start_ts"] = loop_start

            add_timing_mark(timing_marks, "capture_start")
            row["capture_start_ts"] = now_ts()
            if capture_worker is not None:
                bev_frame, cam_display, last_capture_id = capture_worker.get_latest(
                    last_capture_id
                )
            else:
                bev_frame, cam_display = bev_processor.get_bev_frame(
                    include_display=opts.show_preview
                )
            row["capture_end_ts"] = now_ts()
            add_timing_mark(timing_marks, "capture_done")
            row["capture_duration_ms"] = ms(row["capture_start_ts"], row["capture_end_ts"])

            if bev_frame is None:
                row["loop_end_ts"] = now_ts()
                row["loop_duration_ms"] = ms(loop_start, row["loop_end_ts"])
                timing_logger.write(row)
                add_timing_mark(timing_marks, "loop_done")
                timing_mark_logger.record(frame_count + 1, timing_marks)
                time.sleep(0.01)
                continue

            result, process_metrics, steering_metrics = process_frame_timed(
                bev_frame,
                model,
                transform,
                device,
                decode_fn,
                opts,
                ros2_node=ros2_node,
                timing_marks=timing_marks,
                steering_timing_state=steering_timing_state,
            )
            row.update(process_metrics)

            save_start = now_ts()
            if opts.save_output and frame_count % max(opts.record_every, 1) == 0:
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

            if opts.save_images and frame_count % max(opts.save_image_interval, 1) == 0:
                frame_name = f"{frame_count:06d}.png"
                cv2.imwrite(
                    os.path.join(opts.output, "bev_frames", frame_name),
                    bev_frame,
                )
                cv2.imwrite(
                    os.path.join(opts.output, "overlay_frames", frame_name),
                    result,
                )
            save_end = now_ts()
            add_timing_mark(timing_marks, "video_write_done")
            row["save_output_duration_ms"] = ms(save_start, save_end)

            preview_start = now_ts()
            if opts.show_preview:
                h_r, w_r = result.shape[:2]
                cv2.imshow("Combined Navigation Timing", cv2.resize(result, (w_r // 2, h_r // 2)))
                if cam_display is not None:
                    h_c, w_c = cam_display.shape[:2]
                    cv2.imshow("Cameras", cv2.resize(cam_display, (w_c // 2, h_c // 2)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            preview_end = now_ts()
            add_timing_mark(timing_marks, "preview_done")
            row["preview_duration_ms"] = ms(preview_start, preview_end)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

            before_sleep = now_ts()
            elapsed = before_sleep - loop_start
            sleep_s = max(0.0, interval - elapsed)
            time.sleep(sleep_s)
            row["sleep_duration_ms"] = sleep_s * 1000.0

            row["loop_end_ts"] = now_ts()
            add_timing_mark(timing_marks, "loop_done")
            row["loop_duration_ms"] = ms(loop_start, row["loop_end_ts"])
            if row["loop_duration_ms"]:
                row["actual_fps"] = 1000.0 / row["loop_duration_ms"]
            timing_logger.write(row)
            timing_mark_logger.record(
                frame_count, timing_marks, extra_metrics=steering_metrics
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if ros2_node is not None:
            ros2_node.stop()
        if overlay_writer is not None:
            overlay_writer.release()
        if bev_writer is not None:
            bev_writer.release()
        if capture_worker is not None:
            capture_worker.stop()
        bev_processor.release()
        cv2.destroyAllWindows()
        timing_logger.close()
        timing_mark_logger.close()
        if ros2_node is not None:
            ros2_node.destroy_node()
            rclpy.shutdown()
        print(f"Stopped. Total frames: {frame_count}")
        print(f"Timing CSV: {timing_csv_path}")


if __name__ == "__main__":
    main()

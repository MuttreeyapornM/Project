# # # ============================================================
# # #  computer2_nav_client.py  —  Navigation & ROS2 Control
# # #  คอมพิวเตอร์ที่ 2: รับ BEV จาก WebSocket → โมเดล Seg → Navigation → ROS2 cmd_vel
# # # ============================================================

# โค๊ดนี้มีการปรับค่า target point ที่ใช้ในการคำนวณ PP controller ให้สามาถปรับค่าได้

import os
import cv2
import time
import asyncio
import threading
import numpy as np
import argparse
import base64
import csv
import json
import websockets
from glob import glob
import sys
import subprocess
import signal
from scipy.interpolate import CubicSpline
from PIL import Image as PILImage
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms as T

import network
import utils
from datasets import VOCSegmentation, Cityscapes, CustomSegmentation


def patch_numpy_core_alias():
    try:
        import numpy.core as numpy_core
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=[name])
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


patch_numpy_core_alias()

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, CompressedImage
    ROS2_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: ROS2 not found. Running without ROS2 support.")
    ROS2_AVAILABLE = False
    # ── fallback classes เพื่อไม่ให้ NameError ──────────────
    class Node:
        pass
    # sensor_msgs stubs สำหรับ BagRecorder
    class Image: pass
    class CompressedImage: pass
    class Twist:
        def __init__(self):
            self.linear  = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
            self.angular = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
    class String:
        def __init__(self):
            self.data = ""
    class Odometry:
        pass

# ── Constants ────────────────────────────────────────────────
prev_spline_points = None
alpha_smooth       = 0.6

WHEELBASE          = 1.67
MAX_STEERING_ANGLE = 50.0
ROI_W              = 200
ROI_H              = 310

SAFETY_CHECK_INTERVAL = 1.0
SAFETY_TIMEOUT        = 3.0
MODE_LABEL            = "FULL"
METERS_PER_PIXEL      = 0.01
DEFAULT_DRIVABLE_CLASS_IDS = [0, 5, 6]

METRIC_FIELDNAMES = [
    "frame_idx",
    "time_sec",
    "angular_z",
    "linear_x",
]


def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    print("CUDA GPU is not available. Falling back to CPU.")
    return torch.device('cpu')


# ═══════════════════════════════════════════════════════════
#  BagRecorder (ROS 2) — publish result frame → ros2 bag record
# ═══════════════════════════════════════════════════════════

class BagRecorder(Node):
    """
    บันทึกภาพผลลัพธ์จาก Computer 2 (navigation overlay) ลง rosbag2
    โดย publish topic แล้วให้ subprocess "ros2 bag record" จัดการบันทึก

    Topics ที่ publish / บันทึก:
      /nav/result/image_raw      sensor_msgs/Image  — ภาพ overlay หลัง inference
      /nav/bev/image_raw         sensor_msgs/Image  — BEV ต้นฉบับที่รับมาจาก Computer 1
    """

    RESULT_TOPIC = "/nav/result/image_raw"
    BEV_TOPIC    = "/nav/bev/image_raw"

    def __init__(self, bag_path: str,
                 record_result: bool = True,
                 record_bev:    bool = False,
                 record_cmd_vel: bool = True,
                 record_status: bool = False):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 (rclpy) is not available.")

        super().__init__("nav_bag_recorder")

        self.bag_path      = bag_path
        self.record_result = record_result
        self.record_bev    = record_bev
        self.record_cmd_vel = record_cmd_vel
        self.record_status  = record_status
        self._lock         = threading.Lock()
        self._frame_count  = 0
        self._bag_proc     = None

        # สร้าง publisher
        self._pubs = {}
        if record_result:
            self._pubs["result"] = self.create_publisher(Image, self.RESULT_TOPIC, 10)
        if record_bev:
            self._pubs["bev"]    = self.create_publisher(Image, self.BEV_TOPIC,    10)

        # spin node ใน background thread
        self._executor   = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _cv2_to_imgmsg(self, image: np.ndarray, frame_id: str) -> Image:
        """แปลง numpy BGR → sensor_msgs/Image โดยไม่ใช้ cv_bridge"""
        msg                 = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height          = image.shape[0]
        msg.width           = image.shape[1]
        msg.encoding        = "bgr8"
        msg.is_bigendian    = 0
        msg.step            = image.shape[1] * 3
        msg.data            = image.tobytes()
        return msg

    def _topics(self):
        topics = []
        if self.record_result: topics.append(self.RESULT_TOPIC)
        if self.record_bev:    topics.append(self.BEV_TOPIC)
        if self.record_cmd_vel: topics.append("/cmd_vel")
        if self.record_status:  topics.append("/navigation_status")
        return topics

    def open(self):
        """เปิด subprocess ros2 bag record"""
        os.makedirs(os.path.dirname(self.bag_path) or ".", exist_ok=True)
        topics = self._topics()
        cmd = ["ros2", "bag", "record", "-o", self.bag_path] + topics
        self._bag_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"🎬 ros2 bag record started → {self.bag_path}")
        print(f"   Topics: {topics}")
        time.sleep(1.0)   # รอให้ recorder พร้อมก่อน publish

    def write_result(self, result_frame: np.ndarray):
        """publish ภาพ overlay หลัง inference"""
        if not self.record_result or result_frame is None:
            return
        pub = self._pubs.get("result")
        if pub:
            pub.publish(self._cv2_to_imgmsg(result_frame, "nav_result"))
        with self._lock:
            self._frame_count += 1

    def write_bev(self, bev_frame: np.ndarray):
        """publish BEV ต้นฉบับ (optional)"""
        if not self.record_bev or bev_frame is None:
            return
        pub = self._pubs.get("bev")
        if pub:
            pub.publish(self._cv2_to_imgmsg(bev_frame, "nav_bev"))

    def close(self):
        """หยุด ros2 bag record และ shutdown node"""
        if self._bag_proc is not None:
            self._bag_proc.send_signal(signal.SIGINT)
            try:
                self._bag_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._bag_proc.kill()
            self._bag_proc = None
        self._executor.shutdown(wait=False)
        print(f"✅ rosbag closed — {self._frame_count} frames → {self.bag_path}")


# ═══════════════════════════════════════════════════════════
#  ROS2 Node (เหมือนเดิม)
# ═══════════════════════════════════════════════════════════

class VehicleControllerNode(Node):
    def __init__(self, vehicle_config=None):
        super().__init__('vehicle_controller')
        self.config    = vehicle_config or {}
        self.wheelbase = self.config.get('wheelbase', WHEELBASE)
        self.cmd_vel_publisher           = self.create_publisher(Twist,  '/cmd_vel',           10)
        self.navigation_status_publisher = self.create_publisher(String, '/navigation_status', 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.current_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.last_cmd_time    = time.time()
        self.safety_timer     = self.create_timer(SAFETY_CHECK_INTERVAL, self.safety_check)
        self.get_logger().info('✅ Vehicle Controller Node initialized')

    def odom_callback(self, msg):
        self.current_position['x'] = msg.pose.pose.position.x
        self.current_position['y'] = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        self.current_position['theta'] = np.arctan2(
            2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y**2 + o.z**2))

    def publish_cmd_vel(self, steering_angle_deg, linear_x=1.0):
        steering_angle_rad = float(np.radians(steering_angle_deg))
        linear_x           = float(np.clip(linear_x, 0.0, 1.0))
        cmd = Twist()
        cmd.linear.x  = linear_x
        cmd.angular.z = steering_angle_rad
        self.cmd_vel_publisher.publish(cmd)
        self.last_cmd_time = time.time()

    def stop(self):
        self.cmd_vel_publisher.publish(Twist())
        self.get_logger().info('🛑 Vehicle stopped')

    def publish_status(self, msg_str):
        msg = String(); msg.data = msg_str
        self.navigation_status_publisher.publish(msg)

    def safety_check(self):
        elapsed = time.time() - self.last_cmd_time
        if elapsed > SAFETY_TIMEOUT:
            self.stop()
            self.get_logger().warning(f'⚠️  No command for {elapsed:.1f}s - stopping')


# ═══════════════════════════════════════════════════════════
#  Helper functions (เหมือนเดิม)
# ═══════════════════════════════════════════════════════════

def calculate_steering_angle(alpha_deg, target_distance_m, wheelbase=WHEELBASE):
    alpha_rad    = np.radians(alpha_deg)
    ld           = float(target_distance_m)
    steering_rad = np.arctan2(2 * wheelbase * np.sin(alpha_rad), ld) if ld > 0 else 0.0
    steering_deg = np.clip(np.degrees(steering_rad), -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    return steering_deg, np.radians(steering_deg)


def get_fixed_roi(height, width):
    cx, cy = width // 2, height // 2
    return (max(0, cx - ROI_W // 2), max(0, cy - ROI_H // 2),
            min(width, cx + ROI_W // 2), min(height, cy + ROI_H // 2))


def measure_continuous_drivable_width_x(mask, offset_x=0, offset_y=0):
    col_has_drivable = np.any(mask > 0, axis=0)
    if not np.any(col_has_drivable):
        return 0, (offset_x, offset_y, offset_x + 10, offset_y + mask.shape[0])

    start_idx = int(np.argmax(col_has_drivable))
    end_idx = start_idx
    while end_idx + 1 < len(col_has_drivable) and col_has_drivable[end_idx + 1]:
        end_idx += 1

    run_mask = mask[:, start_idx:end_idx + 1]
    ys, _ = np.where(run_mask > 0)
    y1 = int(ys.min()) + offset_y if len(ys) > 0 else offset_y
    y2 = int(ys.max()) + offset_y if len(ys) > 0 else offset_y + mask.shape[0]
    x1 = start_idx + offset_x
    x2 = end_idx + offset_x
    drivable_width_px = x2 - x1 + 1
    return drivable_width_px, (x1, y1, x2, y2)


def check_drivable_width_ahead(final_mask, roi_rect):
    height, width = final_mask.shape
    zone_y1, zone_y2 = 0, height // 3
    vehicle_x1, _, vehicle_x2, _ = roi_rect
    vehicle_width = max(1, vehicle_x2 - vehicle_x1)
    zone_x1 = max(0, min(vehicle_x1, width - 1))
    zone_mask = final_mask[zone_y1:zone_y2, zone_x1:width]
    drivable_width_px, roi = measure_continuous_drivable_width_x(zone_mask, zone_x1, zone_y1)
    linear_x = 1.0 if drivable_width_px >= vehicle_width else 0.0
    return linear_x, drivable_width_px, roi


def clean_drivable_mask(mask, valid_ids=None):
    if valid_ids is None:
        valid_ids = DEFAULT_DRIVABLE_CLASS_IDS
    combined = np.isin(mask, valid_ids).astype(np.uint8) * 255
    kernel   = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(combined)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    else:
        out = combined
    return (out > 0).astype(np.uint8)


def detect_vehicle_and_heading(pred, height, width):
    return (
        (width // 2, height // 2),
        (0, -250),
        np.zeros((height, width), dtype=np.uint8),
        get_fixed_roi(height, width),
    )


def draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px):
    cx1, cy1, cx2, cy2 = cmp_roi
    cmp_color  = (0, 255, 0)   if linear_x >= 1.0 else (0, 0, 255)
    fill_color = (0, 200, 0)   if linear_x >= 1.0 else (0, 0, 200)
    mask_crop  = final_mask[cy1:cy2 + 1, cx1:cx2 + 1]
    layer      = np.zeros_like(overlay)
    layer[cy1:cy2 + 1, cx1:cx2 + 1][mask_crop > 0] = fill_color
    cv2.addWeighted(layer, 0.35, overlay, 1.0, 0, overlay)
    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), cmp_color, 2)


def draw_vehicle_roi(overlay, height, width):
    vehicle_center, heading_vector, _, roi_rect = detect_vehicle_and_heading(
        None, height, width)
    vx1, vy1, vx2, vy2 = roi_rect
    arrow_end = (vehicle_center[0] + heading_vector[0],
                 vehicle_center[1] + heading_vector[1])
    cv2.arrowedLine(overlay, vehicle_center, arrow_end, (255, 0, 255), 3, tipLength=0.3)
    cv2.circle(overlay, vehicle_center, 8, (0, 255, 0), -1)
    cv2.putText(overlay, "Vehicle",
                (vehicle_center[0] + 10, vehicle_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(overlay, (vx1, vy1), (vx2, vy2), (255, 255, 255), 2)
    return vehicle_center, heading_vector, roi_rect


def compute_roi_linear_x(final_mask, roi_rect):
    vx1, vy1, vx2, vy2 = roi_rect
    roi_region      = final_mask[vy1:vy2, vx1:vx2]
    vehicle_roi_width = max(1, vx2 - vx1)
    drivable_width_px, _ = measure_continuous_drivable_width_x(roi_region, vx1, vy1)
    linear_x        = 1.0 if drivable_width_px >= vehicle_roi_width else 0.0
    cmp_roi         = (vx1, vy1, vx2, vy2)
    return linear_x, drivable_width_px, cmp_roi


def select_target_point(path_points, vehicle_center, target_lookahead_m):
    pts = np.asarray(path_points)
    if len(pts) == 0:
        return None, 0.0

    target_px = max(float(target_lookahead_m) / METERS_PER_PIXEL, 1.0)
    vehicle = np.asarray(vehicle_center)
    dists_px = np.linalg.norm(pts - vehicle, axis=1)
    idx = int(np.argmin(np.abs(dists_px - target_px)))
    target_point = tuple(pts[idx].astype(int))
    target_distance_m = float(dists_px[idx] * METERS_PER_PIXEL)
    return target_point, target_distance_m


# ═══════════════════════════════════════════════════════════
#  process_frame  (เหมือนเดิม ไม่มีการเปลี่ยนแปลง logic)
# ═══════════════════════════════════════════════════════════

def process_frame(frame, model, transform, device, decode_fn,
                  conf_thresh=0.7, lookahead_distance=200,
                  target_lookahead_m=2.5, ros2_node=None,
                  drivable_class_ids=None, inference_size=None,
                  use_fp16=False, render_overlay=True,
                  return_metrics=False):
    global prev_spline_points

    input_frame = frame
    if inference_size:
        infer_w, infer_h = inference_size
        if infer_w > 0 and infer_h > 0:
            input_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

    frame_rgb    = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(PILImage.fromarray(frame_rgb)).unsqueeze(0).to(device)
    if use_fp16 and device.type == "cuda":
        input_tensor = input_tensor.half()

    t0 = time.time()
    with torch.inference_mode():
        logits             = model(input_tensor)
        pred_gpu           = torch.argmax(logits, dim=1)[0]
        pred      = pred_gpu.cpu().numpy().astype(np.int64)
    if pred.shape[:2] != frame.shape[:2]:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)
    inference_ms = (time.time() - t0) * 1000.0

    if render_overlay:
        colorized = cv2.cvtColor(decode_fn(pred).astype('uint8'), cv2.COLOR_RGB2BGR)
        overlay   = cv2.addWeighted(frame, 0.35, colorized, 0.65, 0)
    else:
        overlay = None

    height, width = pred.shape
    final_mask = clean_drivable_mask(pred, drivable_class_ids)

    if render_overlay:
        vehicle_center, heading_vector, roi_rect = draw_vehicle_roi(overlay, height, width)
    else:
        vehicle_center, heading_vector, _, roi_rect = detect_vehicle_and_heading(
            None, height, width)
    linear_x, drivable_width_px, cmp_roi = compute_roi_linear_x(final_mask, roi_rect)

    roi_height = height // 3
    roi_mask   = final_mask[:roi_height, :]
    nav_points = []
    for y in range(0, roi_height, max(1, roi_height // 5)):
        xs = np.where(roi_mask[y, :] > 0)[0]
        if len(xs) > 0:
            nav_points.append((min(xs[0] + lookahead_distance, xs[-1]), y))

    spline_points     = []
    interesting_point = None
    steering_deg      = 0.0
    steering_rad      = 0.0
    angle_diff        = 0.0
    target_distance_m = 0.0

    if len(nav_points) >= 3:
        pts  = np.array(nav_points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        t    = np.insert(np.cumsum(dist), 0, 0)
        t_new = np.linspace(t[0], t[-1], 200)
        spline_points = np.stack(
            (CubicSpline(t, pts[:, 0])(t_new),
             CubicSpline(t, pts[:, 1])(t_new)), axis=-1
        ).astype(int)
        if prev_spline_points is not None and len(prev_spline_points) == len(spline_points):
            spline_points = (alpha_smooth * prev_spline_points +
                             (1 - alpha_smooth) * spline_points).astype(int)
        prev_spline_points = spline_points

        if render_overlay:
            for i in range(len(spline_points) - 1):
                cv2.line(overlay, tuple(spline_points[i]), tuple(spline_points[i + 1]), (255, 255, 0), 2)

        interesting_point, target_distance_m = select_target_point(
            spline_points, vehicle_center, target_lookahead_m)
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.putText(overlay, "Target", (interesting_point[0] + 10, interesting_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx) - np.arctan2(-heading_vector[1], heading_vector[0]))
        if angle_diff >  180: angle_diff -= 360
        if angle_diff < -180: angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE)

        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect)

        if render_overlay:
            cv2.putText(overlay, f"Alpha: {angle_diff:.1f} deg",
                        (vehicle_center[0] - 80, vehicle_center[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f}° | linear_x={linear_x:.1f} | "
                f"drivable_width_px={drivable_width_px}")

    elif len(nav_points) > 0:
        interesting_point, target_distance_m = select_target_point(
            nav_points, vehicle_center, target_lookahead_m)
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx) - np.arctan2(-heading_vector[1], heading_vector[0]))
        if angle_diff >  180: angle_diff -= 360
        if angle_diff < -180: angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE)
        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect)

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f}° | linear_x={linear_x:.1f}")

    else:
        linear_x = 0.0
        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(0.0, linear_x=0.0)
            ros2_node.publish_status("NO_PATH_FOUND")

    if render_overlay:
        draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px)

    # ── Info Box ────────────────────────────────────────────
    if render_overlay:
        h_orig, w_orig = frame.shape[:2]
        info_lines = [
            f"PURE PURSUIT  [{MODE_LABEL}]",
            f"Steering: {steering_deg:+.4f} deg  ({steering_rad:+.4f} rad)",
            f"Target Ld: {target_distance_m:.2f} m / set {target_lookahead_m:.2f} m",
            f"linear.x: {'GO' if linear_x >= 1.0 else 'STOP'} ({linear_x:.1f})",
        ]
        box_h = 25 * len(info_lines) + 10
        box_w = 380
        bx, by = w_orig - box_w - 10, 10
        sub   = overlay[by:by + box_h, bx:bx + box_w]
        white = np.ones(sub.shape, dtype=np.uint8) * 255
        overlay[by:by + box_h, bx:bx + box_w] = cv2.addWeighted(sub, 0.4, white, 0.6, 1.0)
        for i, line in enumerate(info_lines):
            color = (0, 0, 0)
            if i == 3: color = (0, 150, 0) if linear_x >= 1.0 else (0, 0, 180)
            cv2.putText(overlay, line, (bx + 8, by + 20 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    spline_points_out = (
        spline_points.tolist() if isinstance(spline_points, np.ndarray) else spline_points
    )
    result = (
        overlay if render_overlay else frame,
        pred,
        spline_points_out,
        vehicle_center,
        interesting_point,
    )
    if not return_metrics:
        return result

    target_x = interesting_point[0] if interesting_point is not None else ""
    target_y = interesting_point[1] if interesting_point is not None else ""
    metrics = {
        "inference_ms": float(inference_ms),
        "vehicle_center_x": int(vehicle_center[0]),
        "vehicle_center_y": int(vehicle_center[1]),
        "target_x": target_x,
        "target_y": target_y,
        "target_offset_px": (
            int(interesting_point[0] - vehicle_center[0])
            if interesting_point is not None else ""
        ),
        "target_distance_m": float(target_distance_m),
        "heading_error_deg": float(angle_diff),
        "steering_deg": float(steering_deg),
        "steering_rad": float(steering_rad),
        "angular_z": float(steering_rad),
        "linear_x": float(linear_x),
        "drivable_width_px": int(drivable_width_px),
        "nav_points_count": int(len(nav_points)),
        "spline_points_count": int(len(spline_points_out)),
        "spline_lateral_std_px": (
            float(np.std(np.asarray(spline_points_out)[:, 0]))
            if len(spline_points_out) > 0 else ""
        ),
        "path_found": 1 if interesting_point is not None else 0,
    }
    return result + (metrics,)


# ═══════════════════════════════════════════════════════════
#  Metrics logger / plots for video-only experiments
# ═══════════════════════════════════════════════════════════

class SimulationMetricsLogger:
    def __init__(self, csv_path, plot_path=None, summary_path=None, enabled=True):
        self.enabled = enabled
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.summary_path = summary_path
        self.rows = []
        self._file = None
        self._writer = None
        self._prev_time = None

        if not self.enabled:
            return

        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDNAMES)
        self._writer.writeheader()
        print(f"📊 Metrics CSV: {self.csv_path}")

    @staticmethod
    def _as_float(value, default=np.nan):
        if value == "" or value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def write(self, frame_idx, time_sec, metrics):
        if not self.enabled:
            return

        row = {name: "" for name in METRIC_FIELDNAMES}
        row["frame_idx"] = int(frame_idx)
        row["time_sec"] = float(time_sec)
        row["angular_z"] = float(self._as_float(metrics.get("angular_z"), 0.0))
        row["linear_x"] = float(self._as_float(metrics.get("linear_x"), 0.0))

        self._writer.writerow(row)
        self.rows.append(row.copy())
        self._prev_time = float(time_sec)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

        if not self.enabled or not self.rows:
            return

        summary = self._build_summary()
        if self.summary_path:
            os.makedirs(os.path.dirname(self.summary_path) or ".", exist_ok=True)
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"📄 Metrics summary: {self.summary_path}")

        if self.plot_path:
            self._save_plots(summary)

    def _series(self, name):
        return np.asarray([self._as_float(row.get(name)) for row in self.rows], dtype=float)

    def _build_summary(self):
        linear_x = self._series("linear_x")

        go_stop_switch_count = 0
        if len(linear_x) > 1:
            states = np.where(np.nan_to_num(linear_x, nan=0.0) >= 0.5, 1, 0)
            go_stop_switch_count = int(np.sum(np.abs(np.diff(states)) > 0))

        return {
            "frame_count": int(len(self.rows)),
            "duration_sec": (
                float(self._series("time_sec")[-1] - self._series("time_sec")[0])
                if len(self.rows) > 1 else 0.0
            ),
            "stop_count": int(np.sum(np.nan_to_num(linear_x, nan=0.0) < 0.5)),
            "go_stop_switch_count": go_stop_switch_count,
        }

    def _save_plots(self, summary):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not installed; skip metrics plots")
            return

        os.makedirs(os.path.dirname(self.plot_path) or ".", exist_ok=True)

        t = self._series("time_sec")
        angular_z = self._series("angular_z")
        linear_x = self._series("linear_x")

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

        axes[0].plot(t, angular_z, color="tab:orange", label="angular.z")
        axes[0].set_title("cmd_vel angular.z")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("rad")

        axes[1].step(t, linear_x, where="post", color="tab:green", label="linear.x")
        axes[1].set_title("cmd_vel linear.x")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("m/s or command scale")
        axes[1].set_ylim(-0.05, max(1.05, float(np.nanmax(linear_x)) + 0.05))

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            "Video-only cmd_vel metrics "
            f"(switches={summary['go_stop_switch_count']})",
            fontsize=14,
        )
        fig.savefig(self.plot_path, dpi=160)
        plt.close(fig)
        print(f"📈 Metrics plots: {self.plot_path}")


# ═══════════════════════════════════════════════════════════
#  WebSocket Client  —  รับ BEV frame จาก Computer 1
# ═══════════════════════════════════════════════════════════

class BEVNavClient:
    def __init__(self, server_uri, model, transform, device, decode_fn, opts,
                 ros2_node=None, bag_recorder=None):
        self.server_uri   = server_uri
        self.model        = model
        self.transform    = transform
        self.device       = device
        self.decode_fn    = decode_fn
        self.opts         = opts
        self.ros2_node    = ros2_node
        self.bag_recorder = bag_recorder   # ← BagRecorder หรือ None
        self.running      = True
        self.frame_count  = 0

        # สำหรับ save output video (optional)
        self.out_writer   = None
        if opts.output:
            os.makedirs(opts.output, exist_ok=True)

    def decode_frame(self, b64_data):
        """แปลง base64 JPEG กลับเป็น numpy BGR"""
        raw   = base64.b64decode(b64_data)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    def _init_writer(self, frame):
        if self.opts.output and self.out_writer is None:
            h, w = frame.shape[:2]
            path = os.path.join(self.opts.output, "nav_output_full.mp4")
            self.out_writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'mp4v'),
                self.opts.fps, (w, h))
            print(f"📹 Recording to: {path}")

    async def run(self):
        global prev_spline_points
        prev_spline_points = None

        print(f"🔌 Connecting to BEV server: {self.server_uri}")
        reconnect_delay = 2.0

        while self.running:
            try:
                async with websockets.connect(
                    self.server_uri,
                    ping_interval=10,
                    ping_timeout=20,
                    max_size=50 * 1024 * 1024      # รองรับ frame ขนาดใหญ่ถึง 50MB
                ) as ws:
                    print("✅ Connected to BEV server")
                    reconnect_delay = 2.0            # reset หลัง connect สำเร็จ

                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue

                        msg_type = data.get("type")

                        if msg_type == "metadata":
                            print(f"   BEV size: {data.get('bev_width')}x{data.get('bev_height')} "
                                  f"@ {data.get('fps')} FPS")

                        elif msg_type == "bev_frame":
                            t_recv = time.time()
                            frame  = self.decode_frame(data["frame"])
                            if frame is None:
                                continue

                            # ─── inference + navigation ──────────────────────
                            result, _, _, _, _ = process_frame(
                                frame,
                                self.model, self.transform, self.device, self.decode_fn,
                                conf_thresh=self.opts.conf_thresh,
                                lookahead_distance=self.opts.lookahead_distance,
                                target_lookahead_m=self.opts.target_lookahead_m,
                                ros2_node=self.ros2_node,
                                drivable_class_ids=self.opts.drivable_class_ids,
                                inference_size=(
                                    (self.opts.inference_width, self.opts.inference_height)
                                    if self.opts.inference_width > 0 and self.opts.inference_height > 0
                                    else None
                                ),
                                use_fp16=self.opts.fp16)

                            self._init_writer(result)
                            if self.out_writer:
                                self.out_writer.write(result)

                            # ── บันทึกลง rosbag ─────────────────────
                            if self.bag_recorder is not None:
                                try:
                                    self.bag_recorder.write_result(result)
                                    self.bag_recorder.write_bev(frame)
                                except Exception as e:
                                    print(f"⚠️  BagRecorder error: {e}")
                            # ─────────────────────────────────────────

                            self.frame_count += 1
                            if self.frame_count % 30 == 0:
                                latency_ms = (time.time() - data.get("timestamp", t_recv)) * 1000
                                bag_info = (f" | 🎬 bag: {self.bag_recorder._frame_count}"
                                            if self.bag_recorder else "")
                                print(f"  🧠 Processed {self.frame_count} frames | "
                                      f"latency: {latency_ms:.0f}ms{bag_info}")

                            if self.opts.show_preview:
                                h_r, w_r = result.shape[:2]
                                cv2.imshow(f"Navigation [{MODE_LABEL}] - Computer 2",
                                           cv2.resize(result, (w_r // 2, h_r // 2)))
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self.running = False
                                    break

            except (websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError, OSError) as e:
                if self.running:
                    print(f"⚠️  Connection lost: {e}. Reconnecting in {reconnect_delay:.0f}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 30.0)

        # cleanup
        if self.ros2_node and ROS2_AVAILABLE:
            self.ros2_node.stop()
        if self.out_writer:
            self.out_writer.release()
        if self.bag_recorder is not None:
            self.bag_recorder.close()
        cv2.destroyAllWindows()
        print(f"✅ Navigation client stopped. Total frames: {self.frame_count}")


class VideoSimulationRunner:
    def __init__(self, video_path, model, transform, device, decode_fn, opts,
                 ros2_node=None, bag_recorder=None):
        self.video_path   = video_path
        self.model        = model
        self.transform    = transform
        self.device       = device
        self.decode_fn    = decode_fn
        self.opts         = opts
        self.ros2_node    = ros2_node
        self.bag_recorder = bag_recorder
        self.running      = True
        self.frame_count  = 0
        self.out_writer   = None
        self.metrics_logger = None
        if opts.output:
            os.makedirs(opts.output, exist_ok=True)

    def _init_writer(self, frame):
        if self.opts.output and self.out_writer is None:
            h, w = frame.shape[:2]
            path = os.path.join(self.opts.output, "nav_sim_output.mp4")
            fps = self.opts.output_fps if self.opts.output_fps > 0 else self.opts.fps
            self.out_writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"📹 Recording simulation output to: {path}")

    def run(self):
        global prev_spline_points
        prev_spline_points = None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video input: {self.video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = self.opts.fps
        frame_period = 1.0 / max(float(source_fps), 1.0)

        if self.opts.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.opts.start_frame)

        print("=" * 60)
        print("🎞️  VIDEO SIMULATION — sep_com2 pipeline")
        print(f"   Video      : {self.video_path}")
        print(f"   Source FPS : {source_fps:.2f}")
        print(f"   ROS2       : {'✅' if self.ros2_node else '❌'}")
        print("=" * 60)

        metrics_csv = self.opts.metrics_csv
        metrics_plot = self.opts.metrics_plot
        metrics_summary = self.opts.metrics_summary
        if not metrics_csv:
            metrics_csv = os.path.join(self.opts.output, "sim_metrics.csv")
        if not metrics_plot:
            metrics_plot = os.path.join(self.opts.output, "sim_metrics_plots.png")
        if not metrics_summary:
            metrics_summary = os.path.join(self.opts.output, "sim_metrics_summary.json")

        self.metrics_logger = SimulationMetricsLogger(
            csv_path=metrics_csv,
            plot_path=None if self.opts.no_metrics_plot else metrics_plot,
            summary_path=metrics_summary,
            enabled=not self.opts.no_metrics,
        )

        try:
            while self.running:
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok:
                    if self.opts.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, self.opts.start_frame)
                        prev_spline_points = None
                        continue
                    break

                result, _, _, _, _, metrics = process_frame(
                    frame,
                    self.model, self.transform, self.device, self.decode_fn,
                    conf_thresh=self.opts.conf_thresh,
                    lookahead_distance=self.opts.lookahead_distance,
                    target_lookahead_m=self.opts.target_lookahead_m,
                    ros2_node=self.ros2_node,
                    drivable_class_ids=self.opts.drivable_class_ids,
                    inference_size=(
                        (self.opts.inference_width, self.opts.inference_height)
                        if self.opts.inference_width > 0 and self.opts.inference_height > 0
                        else None
                    ),
                    use_fp16=self.opts.fp16,
                    return_metrics=True)

                source_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                time_sec = source_frame_idx / max(float(source_fps), 1.0)
                self.metrics_logger.write(source_frame_idx, time_sec, metrics)

                self._init_writer(result)
                if self.out_writer:
                    self.out_writer.write(result)

                if self.bag_recorder is not None:
                    try:
                        self.bag_recorder.write_result(result)
                        self.bag_recorder.write_bev(frame)
                    except Exception as e:
                        print(f"⚠️  BagRecorder error: {e}")

                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    bag_info = (f" | 🎬 bag: {self.bag_recorder._frame_count}"
                                if self.bag_recorder else "")
                    print(f"  🧠 Simulated {self.frame_count} frames{bag_info}")

                if self.opts.show_preview:
                    h_r, w_r = result.shape[:2]
                    cv2.imshow(f"Video Simulation [{MODE_LABEL}]",
                               cv2.resize(result, (w_r // 2, h_r // 2)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                if self.opts.max_frames > 0 and self.frame_count >= self.opts.max_frames:
                    break

                if self.opts.realtime:
                    elapsed = time.time() - loop_start
                    if elapsed < frame_period:
                        time.sleep(frame_period - elapsed)
        finally:
            if self.ros2_node and ROS2_AVAILABLE:
                self.ros2_node.stop()
            if self.out_writer:
                self.out_writer.release()
            if self.metrics_logger is not None:
                self.metrics_logger.close()
            cap.release()
            cv2.destroyAllWindows()
            print(f"✅ Video simulation stopped. Total frames: {self.frame_count}")


# ═══════════════════════════════════════════════════════════
#  ROS2 spin thread (non-blocking)
# ═══════════════════════════════════════════════════════════

def ros2_spin_thread(ros2_node):
    rclpy.spin(ros2_node)


# ═══════════════════════════════════════════════════════════
#  Argparser
# ═══════════════════════════════════════════════════════════

def get_argparser():
    parser = argparse.ArgumentParser(description="Computer 2: Navigation Client & ROS2 Controller")
    parser.add_argument("--video_input", type=str, default=None,
                        help="Run simulation from a local video instead of WebSocket BEV input")
    parser.add_argument("--loop_video", action='store_true', default=False,
                        help="Loop --video_input when the end is reached")
    parser.add_argument("--realtime", action='store_true', default=True,
                        help="Pace video simulation at the source video FPS")
    parser.add_argument("--no_realtime", action='store_false',
                        dest="realtime",
                        help="Process video as fast as possible")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start simulation from this frame index")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop simulation after N frames; 0 processes until video end")
    parser.add_argument("--output_fps", type=float, default=0.0,
                        help="FPS for saved simulation output video; 0 uses --fps")
    parser.add_argument("--no_metrics", action='store_true', default=False,
                        help="Disable video simulation metrics CSV/summary/plots")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="CSV path for video simulation metrics; default: <output>/sim_metrics.csv")
    parser.add_argument("--metrics_plot", type=str, default=None,
                        help="PNG path for metrics plots; default: <output>/sim_metrics_plots.png")
    parser.add_argument("--metrics_summary", type=str, default=None,
                        help="JSON path for metrics summary; default: <output>/sim_metrics_summary.json")
    parser.add_argument("--no_metrics_plot", action='store_true', default=False,
                        help="Write CSV/summary but skip metrics PNG plot generation")
    parser.add_argument("--server_host",    type=str,   default="192.168.1.100",
                        help="IP ของ Computer 1 (BEV server)")
    parser.add_argument("--server_port",    type=int,   default=8765)
    parser.add_argument("--enable_ros2",    action='store_true', default=False)
    parser.add_argument("--dataset",        type=str,   default='cityscapes',
                        choices=['voc', 'cityscapes', 'custom'])
    parser.add_argument("--output",         type=str,   default="./output")
    parser.add_argument("--fps",            type=int,   default=15)
    parser.add_argument("--show_preview",   action='store_true', default=False)
    available_models = sorted(
        n for n in network.modeling.__dict__
        if n.islower() and callable(network.modeling.__dict__[n]))
    parser.add_argument("--model",          type=str,   default='deeplabv3plus_mobilenet',
                        choices=available_models)
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride",  type=int,   default=16, choices=[8, 16])
    parser.add_argument("--ckpt",           type=str,   default=None)
    parser.add_argument("--gpu_id",         type=str,   default='0')
    parser.add_argument("--conf_thresh",    type=float, default=0.7)
    parser.add_argument("--lookahead_distance", type=int, default=200)
    parser.add_argument("--target_lookahead_m", type=float, default=2.5,
                        help="Pure pursuit target distance from vehicle to target point in meters")
    parser.add_argument("--drivable_class_ids", type=int, nargs="+",
                        default=DEFAULT_DRIVABLE_CLASS_IDS,
                        help="Class IDs treated as drivable area for GO/STOP and spline mask")
    parser.add_argument("--fp16", action='store_true', default=False,
                        help="Use FP16 inference on CUDA")
    parser.add_argument("--inference_width", type=int, default=0,
                        help="Resize BEV to this width before inference; 0 keeps original size")
    parser.add_argument("--inference_height", type=int, default=0,
                        help="Resize BEV to this height before inference; 0 keeps original size")
    # ── rosbag options ───────────────────────────────────────
    parser.add_argument("--record_bag",       action='store_true', default=False,
                        help="บันทึกภาพผลลัพธ์ navigation ลง rosbag2")
    parser.add_argument("--bag_path",         type=str,
                        default=f"bags/nav_{time.strftime('%Y%m%d_%H%M%S')}",
                        help="path ของโฟลเดอร์ bag (default: bags/nav_<timestamp>)")
    parser.add_argument("--bag_record_bev",   action='store_true', default=False,
                        help="บันทึก BEV ต้นฉบับไปด้วย (นอกจาก result overlay)")
    parser.add_argument("--no_bag_record_cmd_vel", action='store_false',
                        dest="bag_record_cmd_vel", default=True,
                        help="Do not include /cmd_vel in rosbag topics")
    parser.add_argument("--bag_record_status", action='store_true', default=False,
                        help="Include /navigation_status in rosbag topics")
    return parser


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    opts = get_argparser().parse_args()

    # ── ROS2 ────────────────────────────────────────────────
    ros2_node = None
    ros2_thread = None
    if opts.enable_ros2 and ROS2_AVAILABLE:
        rclpy.init()
        ros2_node   = VehicleControllerNode({'wheelbase': WHEELBASE})
        ros2_thread = threading.Thread(target=ros2_spin_thread, args=(ros2_node,), daemon=True)
        ros2_thread.start()
        print("✅ ROS2 enabled")
    elif opts.enable_ros2 and not ROS2_AVAILABLE:
        print("❌ ROS2 not available!"); return

    # ── Model ───────────────────────────────────────────────
    decode_fn = (VOCSegmentation.decode_target if opts.dataset.lower() == 'voc'
                 else Cityscapes.decode_target)
    opts.num_classes = (21 if opts.dataset.lower() == 'voc' else
                        19 if opts.dataset.lower() == 'cityscapes' else 9)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    if torch.cuda.is_available():
        cudnn.benchmark = True
    device = select_device()
    print(f"Device: {device} | Mode: {MODE_LABEL}")

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride).to(device)
    if opts.ckpt and os.path.isfile(opts.ckpt):
        ckpt = torch.load(opts.ckpt, map_location=device, weights_only=False)
        key  = ("model_state" if "model_state" in ckpt else
                "state_dict"  if "state_dict"  in ckpt else None)
        if key:
            model.load_state_dict(ckpt[key])
        elif isinstance(ckpt, dict) and "module" in str(list(ckpt.keys())[0]):
            from collections import OrderedDict
            nd = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                             for k, v in ckpt.items())
            model.load_state_dict(nd)
        else:
            model.load_state_dict(ckpt)
        print(f"✅ Loaded checkpoint: {opts.ckpt}")

    if device.type == 'cuda' and opts.fp16:
        model = model.half()

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    model.eval()
    if device.type == 'cuda':
        infer_w = opts.inference_width if opts.inference_width > 0 else 800
        infer_h = opts.inference_height if opts.inference_height > 0 else 600
        dtype = torch.float16 if opts.fp16 else torch.float32
        dummy = torch.zeros((1, 3, infer_h, infer_w), device=device, dtype=dtype)
        with torch.inference_mode():
            for _ in range(3):
                model(dummy)
        torch.cuda.synchronize()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    os.makedirs(opts.output, exist_ok=True)

    server_uri = f"ws://{opts.server_host}:{opts.server_port}"
    print("=" * 60)
    print("🖥️  COMPUTER 2 — Navigation Client")
    print(f"   BEV Server : {server_uri}")
    print(f"   Model      : {opts.model}")
    print(f"   ROS2       : {'✅' if ros2_node else '❌'}")
    print("=" * 60)

    # ── ตั้งค่า BagRecorder ─────────────────────────────────
    bag_recorder = None
    if opts.record_bag:
        if not ROS2_AVAILABLE:
            print("❌ --record_bag ถูกเปิดแต่ ROS 2 ไม่พร้อม ข้ามการบันทึก bag")
        else:
            if opts.bag_record_cmd_vel and ros2_node is None:
                print("⚠️  /cmd_vel จะไม่มีข้อมูลถ้าไม่เปิด --enable_ros2")
            bag_recorder = BagRecorder(
                bag_path      = opts.bag_path,
                record_result = True,
                record_bev    = opts.bag_record_bev,
                record_cmd_vel = opts.bag_record_cmd_vel,
                record_status  = opts.bag_record_status,
            )
            bag_recorder.open()
            print(f"   🎬 Bag path  : {opts.bag_path}")
            print(f"   Rec BEV     : {opts.bag_record_bev}")
            print(f"   Rec cmd_vel : {opts.bag_record_cmd_vel}")
            print(f"   Rec status  : {opts.bag_record_status}")
    # ────────────────────────────────────────────────────────

    try:
        if opts.video_input:
            runner = VideoSimulationRunner(
                opts.video_input, model, transform, device, decode_fn, opts,
                ros2_node=ros2_node,
                bag_recorder=bag_recorder,
            )
            runner.run()
        else:
            client = BEVNavClient(
                server_uri, model, transform, device, decode_fn, opts,
                ros2_node    = ros2_node,
                bag_recorder = bag_recorder,
            )
            asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        if bag_recorder is not None:
            bag_recorder.close()
        if ros2_node:
            ros2_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

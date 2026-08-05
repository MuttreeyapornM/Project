# ============================================================
#  sep_com_combined.py
#  Single-computer pipeline:
#  cameras -> BEV -> segmentation/navigation -> /cmd_vel
# ============================================================

import argparse
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
    process_frame,
    select_device,
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


class CombinedVehicleControllerNode(Node):
    def __init__(
        self,
        cmd_vel_topic="/cmd_vel",
        vehicle_config=None,
        cmd_vel_publish_hz=30.0,
    ):
        super().__init__("combined_vehicle_controller")
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

    def odom_callback(self, msg):
        self.current_position["x"] = msg.pose.pose.position.x
        self.current_position["y"] = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        self.current_position["theta"] = np.arctan2(
            2 * (o.w * o.z + o.x * o.y),
            1 - 2 * (o.y**2 + o.z**2),
        )

    def publish_cmd_vel(self, steering_angle_deg, linear_x=1.0):
        cmd = Twist()
        cmd.linear.x = float(np.clip(linear_x, 0.0, 1.0))
        cmd.angular.z = float(np.radians(steering_angle_deg))
        now = time.time()
        with self.cmd_lock:
            self.latest_cmd = cmd
            self.latest_cmd_time = now
        self.cmd_vel_publisher.publish(cmd)
        self.last_cmd_time = now

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
        default="/cmd_vel",
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
        type=int,
        default=0,
        help="Resize BEV to this width before segmentation inference; 0 keeps original size",
    )
    parser.add_argument(
        "--inference_height",
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
    parser.add_argument(
        "--target_lookahead_m",
        type=float,
        default=2.5,
        help="Pure pursuit target distance from vehicle to target point in meters",
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


def main():
    opts = get_argparser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    if torch.cuda.is_available():
        cudnn.benchmark = True
    device = select_device()
    model, transform, decode_fn = build_model(opts, device)
    warmup_model(model, device, opts)
    inference_size = None
    if opts.inference_width > 0 and opts.inference_height > 0:
        inference_size = (opts.inference_width, opts.inference_height)

    ros2_node = None
    ros2_thread = None
    if opts.enable_ros2:
        if not ROS2_AVAILABLE:
            print(f"ROS2 is not available. Cannot publish {opts.cmd_vel_topic}.")
            return
        rclpy.init()
        ros2_node = CombinedVehicleControllerNode(
            cmd_vel_topic=opts.cmd_vel_topic,
            vehicle_config={"wheelbase": WHEELBASE},
            cmd_vel_publish_hz=opts.cmd_vel_publish_hz,
        )
        ros2_thread = threading.Thread(
            target=ros2_spin_thread,
            args=(ros2_node,),
            daemon=True,
        )
        ros2_thread.start()

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
    print(f"  FP16          : {'enabled' if opts.fp16 and device.type == 'cuda' else 'disabled'}")
    print(f"  Async capture : {'enabled' if capture_worker else 'disabled'}")
    print(f"  ROS2          : {'enabled' if ros2_node else 'disabled'}")
    print(f"  cmd_vel topic : {opts.cmd_vel_topic}")
    if ros2_node:
        print(f"  cmd_vel rate  : {opts.cmd_vel_publish_hz:.1f} Hz")
    print("=" * 60)

    frame_count = 0
    last_capture_id = 0
    interval = 1.0 / max(opts.fps, 1)
    render_overlay = opts.show_preview or opts.save_output or opts.save_images

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
                ros2_node=ros2_node,
                drivable_class_ids=opts.drivable_class_ids,
                inference_size=inference_size,
                use_fp16=opts.fp16,
                render_overlay=render_overlay,
            )

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
        if ros2_node is not None:
            ros2_node.stop()
        if capture_worker is not None:
            capture_worker.stop()
        if overlay_writer is not None:
            overlay_writer.release()
        if bev_writer is not None:
            bev_writer.release()
        bev_processor.release()
        cv2.destroyAllWindows()
        if ros2_node is not None:
            ros2_node.destroy_node()
            rclpy.shutdown()
        print(f"Stopped. Total frames: {frame_count}")


if __name__ == "__main__":
    main()

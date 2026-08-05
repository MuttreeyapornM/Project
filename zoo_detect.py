import torch
import torch.nn as nn
import argparse
import numpy as np
import cv2
import os
from collections import OrderedDict
from PIL import Image
from torchvision import transforms as T
import rclpy
from geometry_msgs.msg import Twist
from ros2_publish import PathPublisher

from image_processing import ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h
from nav_processing import (
    ROI_H,
    ROI_W,
    clean_drivable_mask,
    detect_vehicle_and_heading,
    draw_compare_roi,
    draw_vehicle_roi,
    measure_continuous_drivable_width_x,
)


class ZooCmdVelPublisher(PathPublisher):
    def __init__(self, cmd_vel_topic="/cmd_vel"):
        super().__init__()
        self.cmd_vel_topic = cmd_vel_topic
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

    def publish_cmd_vel(self, linear_x):
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_camera", action='store_true', default=False)
    parser.add_argument("--front_cam", type=int, default=2)
    parser.add_argument("--left_cam", type=int, default=4)
    parser.add_argument("--rear_cam", type=int, default=0)
    parser.add_argument("--right_cam", type=int, default=6)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--show_preview", action='store_true', default=False)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--dataset", type=str, default='custom', choices=['voc', 'cityscapes', 'custom'])
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet')
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--crop_val", action='store_true', default=False)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--overlay", action='store_true', default=False)
    parser.add_argument("--overlay_alpha", type=float, default=0.5)
    parser.add_argument("--show_mask_ids", action='store_true', default=False)
    parser.add_argument("--ckpt", default="./checkpoints/clean_state_dict.pth", type=str)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    parser.add_argument("--vehicle_width", type=int, default=ROI_W)
    parser.add_argument("--vehicle_height", type=int, default=ROI_H)
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
        help="Resize frame to this width before segmentation inference; 0 keeps original size",
    )
    parser.add_argument(
        "--inference_height",
        "--model_input_height",
        dest="inference_height",
        type=int,
        default=0,
        help="Resize frame to this height before segmentation inference; 0 keeps original size",
    )
    return parser


def build_model(opts, device):
    if opts.dataset.lower() == "voc":
        opts.num_classes = 21
        from datasets import VOCSegmentation
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == "cityscapes":
        opts.num_classes = 19
        from datasets import Cityscapes
        decode_fn = Cityscapes.decode_target
    else:
        opts.num_classes = 9
        from datasets import Cityscapes
        decode_fn = Cityscapes.decode_target

    from network import modeling
    model = modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride,
    ).to(device)
    if opts.separable_conv and "plus" in opts.model:
        from network import convert_to_separable_conv
        convert_to_separable_conv(model.classifier)
    from utils import set_bn_momentum
    set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        try:
            checkpoint = torch.load(opts.ckpt, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(opts.ckpt, map_location=device)

        key = (
            "model_state"
            if isinstance(checkpoint, dict) and "model_state" in checkpoint
            else "state_dict"
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else None
        )
        if key:
            model.load_state_dict(checkpoint[key])
        elif isinstance(checkpoint, dict) and checkpoint:
            first_key = next(iter(checkpoint.keys()))
            if isinstance(first_key, str) and first_key.startswith("module."):
                clean_state = OrderedDict(
                    (k[7:] if k.startswith("module.") else k, v)
                    for k, v in checkpoint.items()
                )
                model.load_state_dict(clean_state)
            else:
                model.load_state_dict(checkpoint)
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

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform, decode_fn

def check_drivable_width_ahead(final_mask, roi_rect):
    height, width = final_mask.shape
    vehicle_x1, _, vehicle_x2, _ = roi_rect
    vehicle_width = max(1, int(vehicle_x2 - vehicle_x1))
    vehicle_center_x = (vehicle_x1 + vehicle_x2) // 2
    zone_width = min(width, vehicle_width)
    zone_x1 = max(0, vehicle_center_x - zone_width // 2)
    zone_x2 = min(width, zone_x1 + zone_width)
    zone_x1 = max(0, zone_x2 - zone_width)
    zone_y1 = 0
    zone_y2 = height // 3

    zone_mask = final_mask[zone_y1:zone_y2, zone_x1:zone_x2]
    drivable_width_px, _ = measure_continuous_drivable_width_x(
        zone_mask, zone_x1, zone_y1
    )
    linear_x = 1.0 if drivable_width_px >= vehicle_width else 0.0
    return linear_x, drivable_width_px, (zone_x1, zone_y1, zone_x2 - 1, zone_y2 - 1)


def compute_zoo_linear_x(pred, vehicle_size=None):
    height, width = pred.shape
    final_mask = clean_drivable_mask(pred, valid_ids=[0])
    _, _, _, roi_rect = detect_vehicle_and_heading(
        None, height, width, vehicle_size
    )
    linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
        final_mask, roi_rect
    )

    x1, y1, x2, y2 = cmp_roi
    roi_pred = pred[y1 : y2 + 1, x1 : x2 + 1]
    detected_ids = np.unique(roi_pred)
    non_drivable_ids = detected_ids[detected_ids != 0]
    obstacle_detected = len(non_drivable_ids) > 0

    if obstacle_detected:
        linear_x = 0.0

    return (
        linear_x,
        drivable_width_px,
        cmp_roi,
        final_mask,
        detected_ids,
        non_drivable_ids,
        obstacle_detected,
    )


def process_bev_frame(
    bev_frame,
    model,
    transform,
    device,
    decode_fn,
    show_mask_ids=False,
    ros_node=None,
    vehicle_size=None,
    inference_size=None,
    use_fp16=False,
    render_overlay=True,
):
    input_frame = bev_frame
    if inference_size:
        infer_w, infer_h = inference_size
        if infer_w > 0 and infer_h > 0:
            input_frame = cv2.resize(
                bev_frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA
            )

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    if use_fp16 and device.type == "cuda":
        input_tensor = input_tensor.half()

    with torch.inference_mode():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    if pred.shape[:2] != bev_frame.shape[:2]:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (bev_frame.shape[1], bev_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    (
        linear_x,
        drivable_width_px,
        cmp_roi,
        final_mask,
        detected_ids,
        non_drivable_ids,
        obstacle_detected,
    ) = compute_zoo_linear_x(pred, vehicle_size)
    height, width = pred.shape
    roi_height = height // 3
    roi_pred = pred[:roi_height, :]

    if render_overlay:
        colorized_pred = decode_fn(pred).astype("uint8")
        colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)
        result_frame = bev_frame.copy()
        result_frame[:roi_height, :] = cv2.addWeighted(
            bev_frame[:roi_height, :],
            0.35,
            colorized_pred_bgr[:roi_height, :],
            0.65,
            0,
        )
        draw_vehicle_roi(result_frame, height, width, vehicle_size)
        draw_compare_roi(
            result_frame, final_mask, cmp_roi, linear_x, drivable_width_px
        )
    else:
        result_frame = None

    print("🔁 Processing new frame...")
    print(
        "cmd_vel linear.x="
        f"{linear_x:.1f} | drivable_width_px={drivable_width_px} | "
        f"non_drivable_ids={non_drivable_ids.tolist()} | "
        f"{'STOP' if obstacle_detected or linear_x <= 0.0 else 'GO'}"
    )
    if ros_node:
        ros_node.publish_cmd_vel(linear_x)

    if show_mask_ids:
        unique_ids = np.unique(roi_pred)
        print(f"\n📌 ROI (Top 1/3): Mask IDs detected → {unique_ids}")
        if ros_node:
            ros_node.publish_roi_ids(unique_ids.tolist())

    return result_frame, pred, bev_frame.copy(), None


def resize_preview_like_saty4(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))


class BEVProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: self.initialize_video_capture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
        }

    def initialize_video_capture(self, path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def load_calibration_data(self, camera_id):
        yaml_filename = os.path.join("yaml", f"calibration_data_{camera_id}.yaml")
        if not os.path.exists(yaml_filename):
            raise FileNotFoundError(f"Calibration file not found: {yaml_filename}")

        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Could not open calibration file: {yaml_filename}")

        data = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
            "homography": fs.getNode("homography").mat(),
        }
        fs.release()

        missing = [key for key, value in data.items() if value is None]
        if missing:
            raise RuntimeError(
                f"Calibration file {yaml_filename} is missing: {', '.join(missing)}"
            )
        return data

    def process_image(self, image, camera_id):
        calib = self.calibration_data[camera_id]
        undistorted = cv2.undistort(
            image, calib["camera_matrix"], calib["dist_coeffs"]
        )
        if camera_id in ["rear", "right"]:
            undistorted = cv2.rotate(undistorted, cv2.ROTATE_180)

        warped = cv2.warpPerspective(
            undistorted,
            calib["homography"],
            (self.map_width, self.map_height),
        )
        return undistorted, warped

    def get_bev_frame(self):
        images = []
        warped_rgba_ = []

        for cam_id, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Failed to read frame from {cam_id}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    return None, None
            undistorted, warped = self.process_image(frame, cam_id)
            images.append(undistorted)
            warped_rgba_.append(warped)

        final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
        merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)
        return merged_car_image, None

    def release(self):
        for cap in self.caps.values():
            cap.release()

def main():
    opts = get_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inference_size = None
    if opts.inference_width > 0 and opts.inference_height > 0:
        inference_size = (opts.inference_width, opts.inference_height)
    elif opts.inference_width > 0 or opts.inference_height > 0:
        raise ValueError(
            "Set both --model_input_width and --model_input_height "
            "or leave both as 0."
        )

    model, transform, decode_fn = build_model(opts, device)

    rclpy.init()
    publisher_node = ZooCmdVelPublisher(opts.cmd_vel_topic)
    vehicle_size = (
        max(1, int(opts.vehicle_width)),
        max(1, int(opts.vehicle_height)),
    )
    render_overlay = opts.show_preview or bool(opts.output) or opts.overlay

    print("=" * 60)
    print("ZOO DETECT")
    print(f"  Device        : {device}")
    print(f"  Model         : {opts.model}")
    print(f"  Dataset       : {opts.dataset}")
    if inference_size:
        print(f"  Inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print("  Inference size: original frame")
    print(f"  Vehicle ROI   : {vehicle_size[0]}x{vehicle_size[1]} px")
    print(f"  FP16          : {'enabled' if opts.fp16 and device.type == 'cuda' else 'disabled'}")
    print(f"  cmd_vel topic : {opts.cmd_vel_topic}")
    print("=" * 60)

    if opts.use_camera:
        video_paths = {
            "front": opts.front_cam,
            "left": opts.left_cam,
            "rear": opts.rear_cam,
            "right": opts.right_cam,
        }
        bev_processor = BEVProcessor(video_paths, img_car, opts.display_width, opts.display_height)

        try:
            while True:
                bev_frame, _ = bev_processor.get_bev_frame()
                if bev_frame is None:
                    break

                segmented_frame, raw_pred, _, _ = process_bev_frame(
                    bev_frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    ros_node=publisher_node,
                    vehicle_size=vehicle_size,
                    inference_size=inference_size,
                    use_fp16=opts.fp16,
                    render_overlay=render_overlay,
                )

                if opts.show_preview:
                    cv2.imshow(
                        "Segmentation",
                        resize_preview_like_saty4(segmented_frame),
                    )
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            bev_processor.release()
            cv2.destroyAllWindows()
            publisher_node.destroy_node()
            rclpy.shutdown()
    elif opts.input:
        cap = cv2.VideoCapture(opts.input)
        if not cap.isOpened():
            print(f"Failed to open video file: {opts.input}")
            return

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from video")
                    break

                segmented_frame, raw_pred, _, _ = process_bev_frame(
                    frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    ros_node=publisher_node,
                    vehicle_size=vehicle_size,
                    inference_size=inference_size,
                    use_fp16=opts.fp16,
                    render_overlay=render_overlay,
                )

                if opts.show_preview:
                    cv2.imshow(
                        "Segmentation",
                        resize_preview_like_saty4(segmented_frame),
                    )
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            publisher_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

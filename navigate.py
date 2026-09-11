import torch
import torch.nn as nn
from torch.utils.data import dataset
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2
import time
import threading

from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob

from geometry_msgs.msg import Point
import rclpy
from ros_point_publisher import PathPublisher


# Import needed modules from image_processing 
# Note: You'll need to make sure these modules are properly imported
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from capture_settings import configure_capture, is_camera_source, resolve_camera, list_cameras
import camera_grabber
from camera_grabber import CameraGrabber
from param_settings import img_car, Car_dst_points, total_w, total_h

def get_argparser():
    parser = argparse.ArgumentParser()

    # Camera Options
    parser.add_argument("--use_camera", action='store_true', default=False,
                      help="use live camera feed instead of video files")
    # Accepts an index (legacy, e.g. 2) or a stable /dev/v4l/by-path name.
    # Indices move between boots; with four identical cameras a reorder silently
    # applies each homography to the wrong view. Use --list_cameras for the
    # stable names. See capture_settings.resolve_camera().
    parser.add_argument("--front_cam", type=str, default="2",
                      help="front camera: index or /dev/v4l/by-path name")
    parser.add_argument("--left_cam", type=str, default="4",
                      help="left camera: index or /dev/v4l/by-path name")
    parser.add_argument("--rear_cam", type=str, default="0",
                      help="rear camera: index or /dev/v4l/by-path name")
    parser.add_argument("--right_cam", type=str, default="6",
                      help="right camera: index or /dev/v4l/by-path name")
    parser.add_argument("--list_cameras", action="store_true",
                      help="print the stable by-path name of each camera and exit")
    
    # Video Options (for non-camera mode)
    parser.add_argument("--input", type=str, default=None,
                      help="path to input video file or directory")
    parser.add_argument("--output", type=str, default=None,
                      help="path to save the segmented video output")
    parser.add_argument("--fps", type=int, default=5,
                      help="frames per second for output video")
    parser.add_argument("--skip_frames", type=int, default=1,
                      help="process every n-th frame")
    parser.add_argument("--show_preview", action='store_true', default=True,
                      help="show video preview during processing (on by default)")
    parser.add_argument("--no_preview", dest="show_preview", action='store_false',
                      help="disable the preview window. --show_preview defaults to "
                           "True and action='store_true' can never clear it, so this "
                           "is the only way to run headless. Use on the vehicle: the "
                           "preview renders and blits a full-size BEV frame every "
                           "iteration and needs an X display.")
    
    # Display Options
    parser.add_argument("--display_width", type=int, default=800,
                      help="Width of display window")
    parser.add_argument("--display_height", type=int, default=600,
                      help="Height of display window")

    # Dataset Options
    parser.add_argument("--dataset", type=str, default='custom',
                      choices=['voc', 'cityscapes', 'custom'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                            not (name.startswith("__") or name.startswith('_')) and callable(
                            network.modeling.__dict__[name])
                            )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                      choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                      help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Navigation Options
    parser.add_argument("--show_navigation", action='store_true', default=True,
                      help='show navigation path on video')
    parser.add_argument("--nav_line_color", type=str, default='green',
                      help='color of navigation line (red, green, blue, yellow)')
    parser.add_argument("--nav_line_thickness", type=int, default=5,
                      help='thickness of navigation line')
    parser.add_argument("--road_class_id", type=int, default=None,
                      help='class ID for road in the segmentation model')
    parser.add_argument("--lane_class_id", type=int, default=None,
                      help='class ID for lane markings in the segmentation model')

    # Processing Options
    parser.add_argument("--crop_val", action='store_true', default=False,
                      help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                      help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--overlay", action='store_true', default=False,
                      help='overlay segmentation on original video')
    parser.add_argument("--overlay_alpha", type=float, default=0.5,
                      help='opacity of segmentation overlay (0-1)')
    parser.add_argument("--show_mask_ids", action='store_true', default=False,
                      help='print unique mask IDs for each frame')
    
    parser.add_argument("--ckpt", default="./checkpoints/clean_state_dict.pth", type=str,
                      help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                      help="GPU ID")
    return parser

class BEVProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        self.car = img_car
        # Initialize video capture for each camera
        self.caps = {key: self.initialize_video_capture(path) for key, path in video_paths.items()}
        # Live cameras get a grabber thread so the driver queue cannot back up
        # and leave the display running seconds behind reality. Video files are
        # left alone: a grabber would race through the file.
        self.grabbers = {}
        if not camera_grabber.DISABLED:
            for key, path in video_paths.items():
                if is_camera_source(path):
                    self.grabbers[key] = CameraGrabber(self.caps[key], name=key).start()
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
        }

        # The cameras are bolted to the vehicle and the homographies are
        # fixed, so undistort + rotate + homography is a FIXED pixel mapping.
        # Build it once as a single remap lookup table instead of rebuilding
        # ~7 MB of correction maps per camera per frame. Measured on the
        # Jetson AGX Xavier: 4 cameras 252 ms -> 48 ms.
        # See docs/09-hardware/BEV_Benchmark_Xavier_2026-09-05.md
        self.maps = {cam: self._build_maps(cam) for cam in video_paths}

    def _build_maps(self, cameraID):
        """Precompute two LUTs per camera.

        "und": undistorted view (rotated 180 for rear/right, matching the old
               cv2.undistort + cv2.rotate output).
        "bev": undistort + rotate + homography composed into ONE map, so the
               BEV image comes from a single cv2.remap with a single
               resampling instead of undistort followed by warpPerspective.
        """
        CAP_W, CAP_H = 1280, 720
        calib = self.calibration_data[cameraID]
        K = np.asarray(calib["camera_matrix"], dtype=np.float64)
        dist = np.asarray(calib["dist_coeffs"], dtype=np.float64)
        H = np.asarray(calib["homography"], dtype=np.float64)

        # Same K as new camera matrix == cv2.undistort's default behaviour.
        m1, m2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (CAP_W, CAP_H), cv2.CV_32FC1
        )

        rotated = cameraID in ("rear", "right")

        # --- map for the undistorted output ---
        if rotated:
            # rotate180(remap(img, m)) == remap(img, m flipped in both axes)
            u1 = np.ascontiguousarray(m1[::-1, ::-1])
            u2 = np.ascontiguousarray(m2[::-1, ::-1])
        else:
            u1, u2 = m1, m2
        und_map = cv2.convertMaps(u1, u2, cv2.CV_16SC2)

        # --- composed BEV map ---
        Hinv = np.linalg.inv(H)
        us, vs = np.meshgrid(
            np.arange(self.map_width, dtype=np.float64),
            np.arange(self.map_height, dtype=np.float64),
        )
        denom = Hinv[2, 0] * us + Hinv[2, 1] * vs + Hinv[2, 2]
        x = (Hinv[0, 0] * us + Hinv[0, 1] * vs + Hinv[0, 2]) / denom
        y = (Hinv[1, 0] * us + Hinv[1, 1] * vs + Hinv[1, 2]) / denom
        if rotated:
            # H was calibrated against the ROTATED undistorted image; undo the
            # rotation before looking up the undistort map
            x = (CAP_W - 1) - x
            y = (CAP_H - 1) - y
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # compose: sample the undistort maps at the homography's source coords;
        # points outside the camera image get -1 -> remap paints them black
        bx = cv2.remap(m1, x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        by = cv2.remap(m2, x, y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        bev_map = cv2.convertMaps(bx, by, cv2.CV_16SC2)

        return {"und": und_map, "bev": bev_map}

    def initialize_video_capture(self, path):
        cap = cv2.VideoCapture(path)
        # Requests MJPG on live cameras so capture is not held to 10 fps by the
        # default YUYV negotiation. Video files are left untouched.
        return configure_capture(cap, path, 1280, 720)

    def load_calibration_data(self, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        data = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
            "homography": fs.getNode("homography").mat()
        }
        fs.release()
        return data

    def process_image(self, image, cameraID, want_undistorted=True):
        """Return (undistorted, bev, proc_time).

        One cv2.remap per output instead of cv2.undistort + cv2.warpPerspective.
        The old cv2.cuda.undistort branch has been removed: that function does
        not exist in OpenCV's Python bindings and would raise AttributeError on
        a CUDA-enabled build. A CUDA build can run these same LUTs through
        cv2.cuda.remap, which does exist.
        """
        start_time = time.time()
        maps = self.maps[cameraID]
        warped = cv2.remap(image, maps["bev"][0], maps["bev"][1], cv2.INTER_LINEAR)
        img_src_undistorted = None
        if want_undistorted:
            img_src_undistorted = cv2.remap(
                image, maps["und"][0], maps["und"][1], cv2.INTER_LINEAR
            )
        proc_time = time.time() - start_time
        return img_src_undistorted, warped, proc_time

    def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        grabber = self.grabbers.get(cam_id)
        if grabber is not None:
            # Newest frame, never a stale queue entry.
            ret, frame = grabber.read()
            if not ret:
                return False
        else:
            ret, frame = cap.read()
            if not ret:
                # Video-file source: loop back to the start.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    return False
        undistorted, warped, proc_time = self.process_image(frame, cam_id)
        images[index] = undistorted
        warped_rgba_[index] = warped
        processing_times[index] = proc_time
        return True

    def get_bev_frame(self):
        """Get a single BEV frame from all cameras"""
        cam_names = ["Front", "Left", "Rear", "Right"]
        images = [None] * len(self.caps)
        warped_rgba_ = [None] * len(self.caps)
        processing_times = [0.0] * len(self.caps)
        threads = []

        for i, (cam_id, cap) in enumerate(self.caps.items()):
            thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if all(img is not None for img in images):
            # Create the BEV image
            final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
            merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)
            
            # Create debug display
            resized_width = self.display_width // 2
            resized_height = self.display_height // 2
            resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

            for i, img in enumerate(resized_images):
                text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
                cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            top_row = np.hstack((resized_images[0], resized_images[1]))
            bottom_row = np.hstack((resized_images[2], resized_images[3]))
            merged_display_image = np.vstack((top_row, bottom_row))
            
            return merged_car_image, merged_display_image
        
        return None, None

    def release(self):
        for g in self.grabbers.values():
            g.stop()
        self.grabbers = {}
        for cap in self.caps.values():
            cap.release()

def extract_road_mask(pred, road_class_id=None, lane_class_id=None):
    """
    Extract road and/or lane areas from the prediction mask
    
    Args:
        pred: Segmentation prediction mask
        road_class_id: Class ID for road surface
        lane_class_id: Class ID for lane markings
    
    Returns:
        Binary mask with road/lane areas
    """
    # Create empty mask
    mask = np.zeros_like(pred, dtype=np.uint8)
    
    # If we know the road class ID, add it to the mask
    if road_class_id is not None:
        mask = np.logical_or(mask, pred == road_class_id)
    
    # If we know the lane class ID, add it to the mask
    if lane_class_id is not None:
        mask = np.logical_or(mask, pred == lane_class_id)
    
    # If no specific classes were provided, try to use common road class IDs
    # For custom dataset: Drivable=0, Parking=5, Crosswalk=6
    if road_class_id is None and lane_class_id is None:
        # Default road classes for custom dataset
        road_candidates = [0, 5, 6]
        for class_id in road_candidates:
            mask = np.logical_or(mask, pred == class_id)
    
    return mask.astype(np.uint8) * 255

def extract_golf_cart_mask(pred, golf_cart_class_id=8):
    # Create mask for golf cart class
    mask2 = (pred == golf_cart_class_id)
    
    return mask2.astype(np.uint8) * 255

def compute_row_centroids_offset(mask, num_points=3, offset_inside_area=250):
    height, width = mask.shape
    centroids = []

    top_half_height = height // 3
    step = max(1, top_half_height // num_points)

    for y in range(0, top_half_height, step):
        row = mask[y, :]
        x_indices = np.where(row > 0)[0]  # Get all pixels in the mask at this row
        if len(x_indices) > 0:
            x_left = x_indices[0]
            x_right = x_indices[-1]

            # Offset pixels into the area, but not beyond its right edge
            x_offset = min(x_left + offset_inside_area, x_right)
            centroids.append((x_offset, y))

    return centroids

def compute_row_centroids_center(mask2):
    y_indices, x_indices = np.where(mask2 > 0)
    
    if x_indices.size > 0 and y_indices.size > 0:
        x_center = int(np.mean(x_indices))
        y_center = int(np.mean(y_indices))
        return (x_center, y_center)
    else:
        return None

def draw_navigation_line(frame, nav_points, color=(0, 255, 0), radius=10):
    for point in nav_points:
        cv2.circle(frame, point, radius, color, -1)
    
    return frame



# ฟังก์ชันหลัก
def process_bev_frame(bev_frame, model, transform, device, decode_fn, show_mask_ids=False, 
                      show_navigation=False, nav_line_color='green', nav_line_thickness=5, 
                      road_class_id=None, lane_class_id=None, ros_node=None):
    

    # แปลงภาพเป็น tensor
    frame_rgb = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # รันโมเดล segmentation
    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]

    # แปลงเป็นภาพสี
    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)

    # สร้าง mask พื้นที่ที่ขับได้ / ที่จอดรถ / ทางม้าลาย
    drivable_mask = (pred == 0).astype(np.uint8) * 255
    parking_mask = (pred == 5).astype(np.uint8) * 255
    crosswalk_mask = (pred == 6).astype(np.uint8) * 255
    golf_cart_mask = (pred == 8).astype(np.uint8) * 255

    # รวม mask สำหรับสร้างเส้นนำทาง
    combined_mask = ((pred == 0) | (pred == 5) | (pred == 6)).astype(np.uint8) * 255

    # คำนวณจุดสำหรับเส้นนำทาง
    line1_points = compute_row_centroids_offset(combined_mask, offset_inside_area=250)
    center_point = compute_row_centroids_center(golf_cart_mask)

    #ตรวจสอบและจัดให้อยู่ในรูป list
    if center_point is not None:
        line2_points = [center_point]
    else:
        line2_points = []

    print("\n📌 Line 1: Destination Point")
    for i, (x, y) in enumerate(reversed(line1_points)):
        print(f"Point {len(line1_points) - i}: x={x}, y={y}")


    print("\n📌 Line 2: Current Point")
    for i, (x, y) in enumerate(line2_points):
        print(f"Point {i+1}: x={x}, y={y}")


    # วาดเส้นบนภาพ segmentation และภาพ BEV จริง
    segmented_with_lines = colorized_pred_bgr.copy()
    bev_with_lines = bev_frame.copy()

    draw_navigation_line(segmented_with_lines, line1_points, (0, 255, 0), nav_line_thickness)  # Green
    draw_navigation_line(segmented_with_lines, line2_points, (0, 0, 255), nav_line_thickness)  # Red

    draw_navigation_line(bev_with_lines, line1_points, (0, 255, 0), nav_line_thickness)
    draw_navigation_line(bev_with_lines, line2_points, (0, 0, 255), nav_line_thickness)

    # ส่งข้อมูลผ่าน ROS2 ถ้ามี node
    if ros_node and len(line1_points) > 0 and len(line2_points) > 0:
        from geometry_msgs.msg import Point

        destination_point = Point(x=float(line1_points[0][0]), y=float(line1_points[0][1]), z=0.0)
        current_point = Point(x=float(line2_points[0][0]), y=float(line2_points[0][1]), z=0.0)
        
        ros_node.publish_points(destination_point, current_point)

    # สร้างภาพ mask visualization
    road_mask_vis = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
    road_mask_vis[:,:,1] = combined_mask  # ช่องสีเขียว

    return segmented_with_lines, pred, bev_with_lines, road_mask_vis

    
def main():
    opts = get_argparser().parse_args()
    if getattr(opts, "list_cameras", False):
        cams = list_cameras()
        if not cams:
            print("No cameras found under /dev/v4l/by-path.")
            print("Check they are connected: ls /dev/video*")
        else:
            print("Stable camera names (survive reboot and replug):\n")
            for link, target in cams:
                print(f"  {target}   {link}")
            print("\nUse the full path, or just the basename, e.g.:")
            print(f"  --front_cam {cams[0][0]}")
        return

    # Set number of classes and decode function based on dataset
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        from datasets import VOCSegmentation
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        from datasets import Cityscapes
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'custom':
        opts.num_classes = 9
        from datasets import Cityscapes
        decode_fn = Cityscapes.decode_target

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        state_dict = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        # Accept either a bare state_dict or a full training checkpoint
        # ({cur_itrs, model_state, optimizer_state, scheduler_state, best_score}),
        # which is what checkpoints/iter_*.pth actually contains. Previously only
        # the bare form worked, hence the separate clean_weight.py step.
        if isinstance(state_dict, dict) and "model_state" in state_dict:
            print("  training checkpoint detected; using [model_state]",
                  "iter", state_dict.get("cur_itrs"),
                  "best_score", state_dict.get("best_score"))
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model)
        model.to(device)
        print("Loaded model from:", opts.ckpt)
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ✅ เริ่มต้น ROS2 และสร้าง publisher node
    rclpy.init()
    publisher_node = PathPublisher()

    if opts.use_camera:
        video_paths = {
            "front": resolve_camera(opts.front_cam),
            "left": resolve_camera(opts.left_cam),
            "rear": resolve_camera(opts.rear_cam),
            "right": resolve_camera(opts.right_cam),
        }
        bev_processor = BEVProcessor(
            video_paths,
            img_car,
            display_width=opts.display_width,
            display_height=opts.display_height
        )

        if opts.output:
            os.makedirs(opts.output, exist_ok=True)
            output_filename = os.path.join(opts.output, "bev_navigation.mp4")
            test_bev, _ = bev_processor.get_bev_frame()
            if test_bev is not None:
                h, w = test_bev.shape[:2]
                output_w = w // 2
                output_h = h // 2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(output_filename, fourcc, opts.fps, (output_w, output_h))
                print(f"Output video dimensions: {output_w}x{output_h} (50% of original {w}x{h})")
            else:
                print("Could not get test frame to determine output dimensions")
                return

        print("Processing BEV camera feed...")
        try:
            while True:
                bev_frame, camera_display = bev_processor.get_bev_frame()
                if bev_frame is None:
                    print("Failed to get BEV frame")
                    break

                segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
                    bev_frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    show_navigation=opts.show_navigation,
                    nav_line_color=opts.nav_line_color,
                    nav_line_thickness=opts.nav_line_thickness,
                    road_class_id=opts.road_class_id,
                    lane_class_id=opts.lane_class_id,
                    ros_node=publisher_node  # ✅ ส่ง ROS node เข้าไป
                )

                if opts.output:
                    h, w = bev_with_lines.shape[:2]
                    resized_frame = cv2.resize(bev_with_lines, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)

                if opts.show_preview:
                    # Resize frames for display
                    h_seg, w_seg = segmented_frame.shape[:2]
                    h_nav, w_nav = bev_with_lines.shape[:2]

                    segmented_display = cv2.resize(segmented_frame, (w_seg // 2, h_seg // 2))
                    navigation_display = cv2.resize(bev_with_lines, (w_nav // 2, h_nav // 2))

                    # Create combined display - side by side
                    combined_height = max(segmented_display.shape[0], navigation_display.shape[0])
                    combined_width = segmented_display.shape[1] + navigation_display.shape[1]
                    
                    # Create blank canvas
                    combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                    
                    # Place segmentation on the left
                    h1, w1 = segmented_display.shape[:2]
                    combined_frame[:h1, :w1] = segmented_display
                    
                    # Place navigation on the right
                    h2, w2 = navigation_display.shape[:2]
                    combined_frame[:h2, w1:w1+w2] = navigation_display
                    
                    # Add labels
                    cv2.putText(combined_frame, "Segmentation", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined_frame, "BEV with Navigation", (w1 + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Display combined frame
                    cv2.imshow("Combined View", combined_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            bev_processor.release()
            if opts.output:
                out_writer.release()
            cv2.destroyAllWindows()
            publisher_node.destroy_node()
            rclpy.shutdown()

    elif opts.input:
        cap = cv2.VideoCapture(opts.input)
        if not cap.isOpened():
            print(f"Failed to open video file: {opts.input}")
            return

        print("Processing video file ...")

        if opts.output:
            os.makedirs(opts.output, exist_ok=True)
            output_filename = os.path.join(opts.output, "bev_navigation.mp4")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_filename, fourcc, opts.fps, (w, h))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Optionally resize frame here if needed
                bev_frame = frame

                segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
                    bev_frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    show_navigation=opts.show_navigation,
                    nav_line_color=opts.nav_line_color,
                    nav_line_thickness=opts.nav_line_thickness,
                    road_class_id=opts.road_class_id,
                    lane_class_id=opts.lane_class_id,
                    ros_node=publisher_node  # ✅ ROS2 publishing happens here
                )

                if opts.output:
                    resized_frame = cv2.resize(bev_with_lines, (w, h), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)

                if opts.show_preview:
                    segmented_display = cv2.resize(segmented_frame, (w, h))
                    navigation_display = cv2.resize(bev_with_lines, (w, h))

                    combined_frame = np.hstack((segmented_display, navigation_display))
                    cv2.putText(combined_frame, "Video Mode", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow("Combined View", combined_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            cap.release()
            if opts.output:
                out_writer.release()
            cv2.destroyAllWindows()
            publisher_node.destroy_node()
            rclpy.shutdown()

    else:
        print("Error: Either --use_camera or --input must be specified")

if __name__ == '__main__':
    main()
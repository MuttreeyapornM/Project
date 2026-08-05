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

# Import needed modules from image_processing 
# Note: You'll need to make sure these modules are properly imported
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h

def get_argparser():
    parser = argparse.ArgumentParser()

    # Camera Options
    parser.add_argument("--use_camera", action='store_true', default=False,
                      help="use live camera feed instead of video files")
    parser.add_argument("--front_cam", type=int, default=2,
                      help="camera ID for front camera")
    parser.add_argument("--left_cam", type=int, default=4,
                      help="camera ID for left camera")
    parser.add_argument("--rear_cam", type=int, default=0,
                      help="camera ID for rear camera")
    parser.add_argument("--right_cam", type=int, default=6,
                      help="camera ID for right camera")
    
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
                      help="show video preview during processing")
    
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
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in video_paths.keys()
        }

        # Check if CUDA is available
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def initialize_video_capture(self, path):
        cap = cv2.VideoCapture(path)
        # Set the frame size to 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

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

    def process_image(self, image, cameraID):
        start_time = time.time()
        calib = self.calibration_data[cameraID]
        camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

        # Use CUDA for undistortion if available
        if self.cuda_available:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            gpu_camera_matrix = cv2.cuda_GpuMat()
            gpu_camera_matrix.upload(camera_matrix)
            gpu_dist_coeffs = cv2.cuda_GpuMat()
            gpu_dist_coeffs.upload(dist_coeffs)
            # Perform GPU-based undistortion
            gpu_undistorted = cv2.cuda.undistort(gpu_image, gpu_camera_matrix, gpu_dist_coeffs)
            img_src_undistorted = gpu_undistorted.download()
        else:
            # Use CPU if CUDA is not available
            img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        if cameraID in ["rear", "right"]:
            img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

        # Use CUDA for perspective warping if available
        if self.cuda_available:
            gpu_undistorted = cv2.cuda_GpuMat()
            gpu_undistorted.upload(img_src_undistorted)
            gpu_homography = cv2.cuda_GpuMat()
            gpu_homography.upload(H)
            # Perform GPU-based perspective warping
            gpu_warped = cv2.cuda.warpPerspective(gpu_undistorted, gpu_homography, (self.map_width, self.map_height))
            warped = gpu_warped.download()
        else:
            # Use CPU if CUDA is not available
            warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))

        proc_time = time.time() - start_time
        return img_src_undistorted, warped, proc_time

    def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        ret, frame = cap.read()
        if not ret:
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

def compute_row_centroids_offset(mask, num_points=7, offset_inside_area=250):
    """
    For each row, return a point that is 300 pixels *into* the area
    from the leftmost pixel of the mask.
    """
    height, width = mask.shape
    centroids = []
    step = max(1, height // num_points)

    for y in range(0, height, step):
        row = mask[y, :]
        x_indices = np.where(row > 0)[0]  # Get all pixels in the mask at this row
        if len(x_indices) > 0:
            x_left = x_indices[0]
            x_right = x_indices[-1]

            # Offset pixels into the area, but not beyond its right edge
            x_offset = min(x_left + offset_inside_area, x_right)
            centroids.append((x_offset, y))

            # print(f"x={x_offset}, y={y}")

    return centroids

def compute_row_centroids_center(mask, num_points=5):
    """
    For each row, compute the horizontal center (mean x) of the mask.
    Suitable for tracking object center like the golf cart.
    """
    height, width = mask.shape
    centroids = []
    step = max(1, height // num_points)

    for y in range(0, height, step):
        row = mask[y, :]
        x_indices = np.where(row > 0)[0]
        if len(x_indices) > 0:
            x_center = int(np.mean(x_indices))
            centroids.append((x_center, y))

    return centroids

def draw_navigation_line(frame, nav_points, color, thickness=5):
    """
    Draw navigation line on the frame
    
    Args:
        frame: Video frame
        nav_points: List of points forming the navigation line
        color: Color of the line in BGR format
        thickness: Thickness of the line
    
    Returns:
        Frame with navigation line drawn
    """
    # Draw line segments connecting navigation points
    if len(nav_points) >= 2:
        for i in range(len(nav_points)-1):
            cv2.line(frame, nav_points[i], nav_points[i+1], color, thickness)
    
    # Draw small circles at each navigation point for better visibility
    for point in nav_points:
        cv2.circle(frame, point, thickness, color, -1)
    
    return frame

def process_bev_frame(bev_frame, model, transform, device, decode_fn, show_mask_ids=False, 
                      show_navigation=False, nav_line_color='green', nav_line_thickness=5, 
                      road_class_id=None, lane_class_id=None):
    """Process a BEV frame, return segmented frame with Line1 and Line2 drawn"""
    
    # Convert input to tensor
    frame_rgb = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]

    # Decode and convert to OpenCV BGR
    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)

    # Create masks for specific classes
    drivable_mask = (pred == 0).astype(np.uint8) * 255
    parking_mask = (pred == 5).astype(np.uint8) * 255
    crosswalk_mask = (pred == 6).astype(np.uint8) * 255
    golf_cart_mask = (pred == 8).astype(np.uint8) * 255

    # Create combined mask for target line
    combined_mask = ((pred == 0) | (pred == 5) | (pred == 6)).astype(np.uint8) * 255

    # Calculate centroids across full height
    line1_points = compute_row_centroids_offset(combined_mask, offset_inside_area=250)
    line2_points = compute_row_centroids_center(golf_cart_mask)

    #ไม่จำเป็นต้องแสดงค่าใน terminal ---> ควรให้ ros เป็นคนดึงค่าเพื่อไปใช้ต่อดีกว่า
    print("\n📌 Line 1: จุด offset จากขอบซ้ายเข้ามาในเลน")
    for i, (x, y) in enumerate(line1_points):
        print(f"Point {i+1}: x={x}, y={y}")

    print("\n📌 Line 2: จุดกึ่งกลางของเลน")
    for i, (x, y) in enumerate(line2_points):
        print(f"Point {i+1}: x={x}, y={y}")

    # Draw on both the segmented image and original BEV
    segmented_with_lines = colorized_pred_bgr.copy()
    bev_with_lines = bev_frame.copy()
    
    # Draw navigation lines on segmented image
    draw_navigation_line(segmented_with_lines, line1_points, (0, 255, 0), 2)  # Green for target line
    draw_navigation_line(segmented_with_lines, line2_points, (0, 0, 255), 2)  # Red for vehicle heading
    
    # Draw navigation lines on original BEV
    draw_navigation_line(bev_with_lines, line1_points, (0, 255, 0), 2)  # Green for target line
    draw_navigation_line(bev_with_lines, line2_points, (0, 0, 255), 2)  # Red for vehicle heading

    # Create visualization of the road mask
    road_mask_vis = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 3), dtype=np.uint8)
    road_mask_vis[:,:,1] = combined_mask  # Green channel for the combined mask
    
    return segmented_with_lines, pred, bev_with_lines, road_mask_vis

# def main():
#     opts = get_argparser().parse_args()

#     # Set number of classes and decode function based on dataset
#     if opts.dataset.lower() == 'voc':
#         opts.num_classes = 21
#         from datasets import VOCSegmentation
#         decode_fn = VOCSegmentation.decode_target
#     elif opts.dataset.lower() == 'cityscapes':
#         opts.num_classes = 19
#         from datasets import Cityscapes
#         decode_fn = Cityscapes.decode_target
#     elif opts.dataset.lower() == 'custom':
#         opts.num_classes = 9
#         from datasets import Cityscapes
#         decode_fn = Cityscapes.decode_target

#     # Setup device
#     os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device:", device)

#     # Load model
#     model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
#     if opts.separable_conv and 'plus' in opts.model:
#         network.convert_to_separable_conv(model.classifier)
#     utils.set_bn_momentum(model.backbone, momentum=0.01)

#     if opts.ckpt and os.path.isfile(opts.ckpt):
#         state_dict = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
#         model.load_state_dict(state_dict)
#         # model.load_state_dict(checkpoint["model_state"])
#         model = nn.DataParallel(model)
#         model.to(device)
#         print("Loaded model from:", opts.ckpt)
#     else:
#         print("[!] Retrain")
#         model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()

#     # Setup transforms
#     transform = T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Set up BEV processor with camera inputs
#     if opts.use_camera:
#         video_paths = {
#             "front": opts.front_cam,
#             "left": opts.left_cam,
#             "rear": opts.rear_cam,
#             "right": opts.right_cam,
#         }
#         bev_processor = BEVProcessor(
#             video_paths, 
#             img_car,
#             display_width=opts.display_width,
#             display_height=opts.display_height
#         )
        
#         # Setup video output if needed
#         if opts.output:
#             os.makedirs(opts.output, exist_ok=True)
#             output_filename = os.path.join(opts.output, "bev_navigation.mp4")
            
#             # The output size will be determined by the BEV image dimension, so we'll get one
#             # frame first to determine the size
#             test_bev, _ = bev_processor.get_bev_frame()
#             if test_bev is not None:
#                 h, w = test_bev.shape[:2]
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out_writer = cv2.VideoWriter(output_filename, fourcc, opts.fps, (w, h))
#             else:
#                 print("Could not get test frame to determine output dimensions")
#                 return
        
#         print("Processing BEV camera feed...")
#         try:
#             while True:
#                 bev_frame, camera_display = bev_processor.get_bev_frame()
                
#                 if bev_frame is None:
#                     print("Failed to get BEV frame")
#                     break
                
#                 # Process the BEV frame and get results
#                 segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
#                     bev_frame, model, transform, device, decode_fn,
#                     show_mask_ids=opts.show_mask_ids,
#                     show_navigation=opts.show_navigation,
#                     nav_line_color=opts.nav_line_color,
#                     nav_line_thickness=opts.nav_line_thickness,
#                     road_class_id=opts.road_class_id,
#                     lane_class_id=opts.lane_class_id
#                 )
                
#                 # Write to video if output is specified
#                 if opts.output:
#                     out_writer.write(bev_with_lines)
                
#                 # Display results
#                 if opts.show_preview:
#                     #cv2.imshow("Camera Feeds", camera_display)
#                     cv2.imshow("BEV with Navigation", bev_with_lines)
#                     cv2.imshow("Segmentation", segmented_frame)
#                     #cv2.imshow("Road Mask", road_mask_vis)
                    
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
                        
#         except KeyboardInterrupt:
#             print("Interrupted by user")
#         finally:
#             bev_processor.release()
#             if opts.output:
#                 out_writer.release()
#             cv2.destroyAllWindows()
    
#     # Process video file instead of camera
#     elif opts.input:
#         # This branch processes regular video files as in the original script
#         video_files = []
#         if os.path.isdir(opts.input):
#             for ext in ['mp4', 'avi', 'mov', 'mkv']:
#                 files = glob(os.path.join(opts.input, f"*.{ext}"))
#                 video_files.extend(files)
#         elif os.path.isfile(opts.input):
#             video_files.append(opts.input)

#         if not video_files:
#             print("No video files found!")
#             return

#         for video_path in video_files:
#             video_name = os.path.basename(video_path).split('.')[0]
#             cap = cv2.VideoCapture(video_path)

#             if not cap.isOpened():
#                 print(f"Failed to open video: {video_path}")
#                 continue

#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = opts.fps if opts.fps else cap.get(cv2.CAP_PROP_FPS)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             print(f"Processing: {video_name}, {width}x{height}, {fps} FPS, {total_frames} frames")

#             # Setup video writer
#             if opts.output:
#                 os.makedirs(opts.output, exist_ok=True)
#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 out_path = os.path.join(opts.output, f"{video_name}_segmented.mp4")
#                 out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#             frame_count = 0
#             progress_bar = tqdm(total=total_frames)

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 frame_count += 1
#                 progress_bar.update(1)

#                 if (frame_count - 1) % opts.skip_frames != 0:
#                     continue

#                 # Process the frame and get result
#                 segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
#                     frame, model, transform, device, decode_fn,
#                     show_mask_ids=opts.show_mask_ids,
#                     show_navigation=opts.show_navigation,
#                     nav_line_color=opts.nav_line_color,
#                     nav_line_thickness=opts.nav_line_thickness,
#                     road_class_id=opts.road_class_id,
#                     lane_class_id=opts.lane_class_id
#                 )

#                 if opts.show_preview:
#                     cv2.imshow("Segmented Output", segmented_frame)
#                     cv2.imshow("Navigation Output", bev_with_lines)
                    
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break

#                 if opts.output:
#                     out_writer.write(bev_with_lines)

#             progress_bar.close()
#             cap.release()
            
#             if opts.output:
#                 out_writer.release()
#                 print(f"Output video saved to: {out_path}")

#             if opts.show_preview:
#                 cv2.destroyAllWindows()
    
#     else:
#         print("Error: Either --use_camera or --input must be specified")

# if __name__ == '__main__':
#     main()

#-------------------------------new main resize output-----------------------------------
def main():
    opts = get_argparser().parse_args()

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
        model.load_state_dict(state_dict)
        # model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Loaded model from:", opts.ckpt)
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    # Setup transforms
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set up BEV processor with camera inputs
    if opts.use_camera:
        video_paths = {
            "front": opts.front_cam,
            "left": opts.left_cam,
            "rear": opts.rear_cam,
            "right": opts.right_cam,
        }
        bev_processor = BEVProcessor(
            video_paths, 
            img_car,
            display_width=opts.display_width,
            display_height=opts.display_height
        )
        
        # Setup video output if needed
        if opts.output:
            os.makedirs(opts.output, exist_ok=True)
            output_filename = os.path.join(opts.output, "bev_navigation.mp4")
            
            # The output size will be determined by the BEV image dimension, so we'll get one
            # frame first to determine the size and reduce it by 50%
            test_bev, _ = bev_processor.get_bev_frame()
            if test_bev is not None:
                h, w = test_bev.shape[:2]
                # Reduce dimensions by 50%
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
                
                # Process the BEV frame and get results
                segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
                    bev_frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    show_navigation=opts.show_navigation,
                    nav_line_color=opts.nav_line_color,
                    nav_line_thickness=opts.nav_line_thickness,
                    road_class_id=opts.road_class_id,
                    lane_class_id=opts.lane_class_id
                )
                
                # Write to video if output is specified - resize to 50%
                if opts.output:
                    h, w = bev_with_lines.shape[:2]
                    resized_frame = cv2.resize(bev_with_lines, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)
                
                # Display results - also resize display windows to 50%
                if opts.show_preview:
                    # Resize frames for display
                    h_seg, w_seg = segmented_frame.shape[:2]
                    h_nav, w_nav = bev_with_lines.shape[:2]
                    
                    segmented_display = cv2.resize(segmented_frame, (w_seg // 2, h_seg // 2), interpolation=cv2.INTER_AREA)
                    navigation_display = cv2.resize(bev_with_lines, (w_nav // 2, h_nav // 2), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow("BEV with Navigation", navigation_display)
                    cv2.imshow("Segmentation", segmented_display)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            bev_processor.release()
            if opts.output:
                out_writer.release()
            cv2.destroyAllWindows()
    
    # Process video file instead of camera
    elif opts.input:
        # This branch processes regular video files as in the original script
        video_files = []
        if os.path.isdir(opts.input):
            for ext in ['mp4', 'avi', 'mov', 'mkv']:
                files = glob(os.path.join(opts.input, f"*.{ext}"))
                video_files.extend(files)
        elif os.path.isfile(opts.input):
            video_files.append(opts.input)

        if not video_files:
            print("No video files found!")
            return

        for video_path in video_files:
            video_name = os.path.basename(video_path).split('.')[0]
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = opts.fps if opts.fps else cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate 50% reduced dimensions
            output_width = width // 2
            output_height = height // 2
            
            print(f"Processing: {video_name}, {width}x{height} -> {output_width}x{output_height}, {fps} FPS, {total_frames} frames")

            # Setup video writer with reduced dimensions
            if opts.output:
                os.makedirs(opts.output, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join(opts.output, f"{video_name}_segmented.mp4")
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (output_width, output_height))

            frame_count = 0
            progress_bar = tqdm(total=total_frames)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress_bar.update(1)

                if (frame_count - 1) % opts.skip_frames != 0:
                    continue

                # Process the frame and get result
                segmented_frame, raw_pred, bev_with_lines, road_mask_vis = process_bev_frame(
                    frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    show_navigation=opts.show_navigation,
                    nav_line_color=opts.nav_line_color,
                    nav_line_thickness=opts.nav_line_thickness,
                    road_class_id=opts.road_class_id,
                    lane_class_id=opts.lane_class_id
                )

                # Resize frames for display and output
                if opts.show_preview:
                    h_seg, w_seg = segmented_frame.shape[:2]
                    h_nav, w_nav = bev_with_lines.shape[:2]
                    
                    segmented_display = cv2.resize(segmented_frame, (w_seg // 2, h_seg // 2), interpolation=cv2.INTER_AREA)
                    navigation_display = cv2.resize(bev_with_lines, (w_nav // 2, h_nav // 2), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow("Segmented Output (50%)", segmented_display)
                    cv2.imshow("Navigation Output (50%)", navigation_display)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if opts.output:
                    # Resize frame before writing to output
                    h, w = bev_with_lines.shape[:2]
                    resized_frame = cv2.resize(bev_with_lines, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)

            progress_bar.close()
            cap.release()
            
            if opts.output:
                out_writer.release()
                print(f"Output video saved to: {out_path} (50% size reduction)")

            if opts.show_preview:
                cv2.destroyAllWindows()
    
    else:
        print("Error: Either --use_camera or --input must be specified")

if __name__ == '__main__':
    main()
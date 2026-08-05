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
    
    # ROI Options
    parser.add_argument("--use_roi", action='store_true', default=False,
                      help="process only ROI (top 1/3 of frame)")
    parser.add_argument("--roi_height_fraction", type=float, default=0.33,
                      help="fraction of frame height to use as ROI (default: 0.33 for top 1/3)")
    parser.add_argument("--stop_on_ids", type=str, default="",
                      help="comma-separated list of class IDs that should trigger stop (e.g., '1,2,3')")
    
    parser.add_argument("--ckpt", default="./checkpoints/best2700.pth", type=str,
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

def extract_roi(frame, roi_height_fraction=0.33):
    """
    Extract Region of Interest (ROI) from the top portion of the frame
    
    Args:
        frame: input image/frame
        roi_height_fraction: fraction of frame height to use (0.33 = top 1/3)
    
    Returns:
        roi_frame: cropped frame containing only ROI
        roi_coords: tuple of (x1, y1, x2, y2) coordinates of ROI
    """
    h, w = frame.shape[:2]
    roi_height = int(h * roi_height_fraction)
    
    # Extract top portion of the frame
    roi_frame = frame[0:roi_height, 0:w]
    roi_coords = (0, 0, w, roi_height)
    
    return roi_frame, roi_coords

def create_full_frame_result(roi_result, original_frame_shape, roi_coords):
    """
    Create a full-frame result by placing ROI result in correct position
    and filling the rest with black
    
    Args:
        roi_result: segmentation result from ROI processing
        original_frame_shape: shape of original frame (h, w, c)
        roi_coords: coordinates of ROI (x1, y1, x2, y2)
    
    Returns:
        full_frame_result: full-sized frame with ROI result placed correctly
    """
    h, w = original_frame_shape[:2]
    channels = roi_result.shape[2] if len(roi_result.shape) == 3 else 1
    
    if channels == 3:
        full_frame_result = np.zeros((h, w, 3), dtype=roi_result.dtype)
    else:
        full_frame_result = np.zeros((h, w), dtype=roi_result.dtype)
    
    x1, y1, x2, y2 = roi_coords
    
    if channels == 3:
        full_frame_result[y1:y2, x1:x2] = roi_result
    else:
        full_frame_result[y1:y2, x1:x2] = roi_result
    
    return full_frame_result

def check_stop_condition(pred_mask, stop_ids, roi_coords=None):
    """
    Check if any of the specified IDs are present in the prediction mask
    
    Args:
        pred_mask: prediction mask with class IDs
        stop_ids: list of class IDs that should trigger stop
        roi_coords: ROI coordinates for focused checking (optional)
    
    Returns:
        should_stop: boolean indicating if vehicle should stop
        detected_ids: list of detected stop IDs
        detection_info: dictionary with detection details
    """
    if not stop_ids:
        return False, [], {}
    
    # Convert stop_ids to list of integers if it's a string
    if isinstance(stop_ids, str):
        stop_ids = [int(id.strip()) for id in stop_ids.split(',') if id.strip()]
    
    unique_ids = np.unique(pred_mask)
    detected_stop_ids = [id for id in unique_ids if id in stop_ids]
    
    should_stop = len(detected_stop_ids) > 0
    
    detection_info = {
        'all_detected_ids': unique_ids.tolist(),
        'stop_ids_detected': detected_stop_ids,
        'total_pixels': pred_mask.size,
        'roi_coords': roi_coords
    }
    
    # Count pixels for each detected stop ID
    for stop_id in detected_stop_ids:
        pixel_count = np.sum(pred_mask == stop_id)
        detection_info[f'pixels_id_{stop_id}'] = pixel_count
        detection_info[f'percentage_id_{stop_id}'] = (pixel_count / pred_mask.size) * 100
    
    return should_stop, detected_stop_ids, detection_info

def draw_roi_indicator(frame, roi_coords, color=(0, 255, 255), thickness=2):
    """
    Draw ROI boundary on the frame
    
    Args:
        frame: input frame to draw on
        roi_coords: ROI coordinates (x1, y1, x2, y2)
        color: color of the boundary line (B, G, R)
        thickness: thickness of the boundary line
    
    Returns:
        frame with ROI boundary drawn
    """
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, "ROI", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def process_bev_frame(bev_frame, model, transform, device, decode_fn, 
                     show_mask_ids=False, use_roi=False, roi_height_fraction=0.33,
                     stop_ids=None):
    """
    Process a BEV frame, return segmented frame with optional ROI processing
    
    Args:
        bev_frame: input BEV frame
        model: segmentation model
        transform: image transform
        device: computation device
        decode_fn: function to decode predictions
        show_mask_ids: whether to print mask IDs
        use_roi: whether to use ROI processing
        roi_height_fraction: fraction of frame height for ROI
        stop_ids: list of IDs that should trigger stop
    
    Returns:
        colorized_pred_bgr: colorized segmentation result
        pred: raw prediction mask
        detection_info: information about detected objects
        should_stop: whether vehicle should stop
    """
    
    roi_coords = None
    detection_info = {}
    should_stop = False
    
    if use_roi:
        # Extract ROI from the frame
        roi_frame, roi_coords = extract_roi(bev_frame, roi_height_fraction)
        processing_frame = roi_frame
    else:
        processing_frame = bev_frame
    
    # Convert input to tensor
    frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]

    # Decode and convert to OpenCV BGR
    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)
    
    # If using ROI, create full-frame result
    if use_roi:
        colorized_pred_bgr = create_full_frame_result(colorized_pred_bgr, bev_frame.shape, roi_coords)
        pred = create_full_frame_result(pred, bev_frame.shape[:2], roi_coords)
    
    # Check stop condition
    if stop_ids:
        should_stop, detected_stop_ids, detection_info = check_stop_condition(pred, stop_ids, roi_coords)
        
        if should_stop:
            print(f"⚠️  STOP CONDITION TRIGGERED! Detected IDs: {detected_stop_ids}")
            for stop_id in detected_stop_ids:
                pixels = detection_info.get(f'pixels_id_{stop_id}', 0)
                percentage = detection_info.get(f'percentage_id_{stop_id}', 0)
                print(f"   ID {stop_id}: {pixels} pixels ({percentage:.2f}%)")

    # Show unique mask IDs if requested
    if show_mask_ids:
        unique_ids = np.unique(pred)
        print(f"Unique mask IDs in frame: {unique_ids}")
        if use_roi:
            print(f"ROI coordinates: {roi_coords}")
    
    return colorized_pred_bgr, pred, detection_info, should_stop

def main():
    opts = get_argparser().parse_args()

    # Parse stop IDs
    stop_ids = []
    if opts.stop_on_ids:
        stop_ids = [int(id.strip()) for id in opts.stop_on_ids.split(',') if id.strip()]
        print(f"Stop condition IDs: {stop_ids}")

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
            output_filename = os.path.join(opts.output, "bev_segmentation.mp4")
            
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
        if opts.use_roi:
            print(f"Using ROI processing: top {opts.roi_height_fraction*100:.1f}% of frame")
        
        stop_triggered = False
        
        try:
            while True:
                bev_frame, camera_display = bev_processor.get_bev_frame()
                
                if bev_frame is None:
                    print("Failed to get BEV frame")
                    break
                
                # Process the BEV frame and get results
                segmented_frame, raw_pred, detection_info, should_stop = process_bev_frame(
                    bev_frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    use_roi=opts.use_roi,
                    roi_height_fraction=opts.roi_height_fraction,
                    stop_ids=stop_ids
                )
                
                # Handle stop condition
                if should_stop and not stop_triggered:
                    stop_triggered = True
                    print("🛑 VEHICLE SHOULD STOP!")
                    # Here you can add code to actually stop the vehicle
                    # For now, we'll just continue processing but mark it
                
                # Write to video if output is specified - resize to 50%
                if opts.output:
                    h, w = segmented_frame.shape[:2]
                    resized_frame = cv2.resize(segmented_frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)
                
                # Display results - combine original and segmentation in one window
                if opts.show_preview:
                    h_seg, w_seg = segmented_frame.shape[:2]
                    h_bev, w_bev = bev_frame.shape[:2]
                    
                    # Resize both frames to same size for side-by-side display
                    display_width = min(w_seg, w_bev) // 2
                    display_height = min(h_seg, h_bev) // 2
                    
                    bev_display = cv2.resize(bev_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    segmented_display = cv2.resize(segmented_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    
                    # Draw ROI indicator if using ROI
                    if opts.use_roi:
                        roi_h = int(display_height * opts.roi_height_fraction)
                        roi_coords_display = (0, 0, display_width, roi_h)
                        bev_display = draw_roi_indicator(bev_display, roi_coords_display)
                    
                    # Add labels and status to each image
                    cv2.putText(bev_display, "Original BEV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(segmented_display, "Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add stop status
                    if should_stop:
                        cv2.putText(bev_display, "STOP!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(segmented_display, "STOP!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Add detected IDs info
                    if detection_info.get('all_detected_ids'):
                        ids_text = f"IDs: {detection_info['all_detected_ids']}"
                        cv2.putText(segmented_display, ids_text, (10, display_height - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Combine side by side
                    combined_display = np.hstack((bev_display, segmented_display))
                    
                    cv2.imshow("BEV Processing Results", combined_display)
                    
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
        # This branch processes regular video files
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
            if opts.use_roi:
                print(f"Using ROI processing: top {opts.roi_height_fraction*100:.1f}% of frame")

            # Setup video writer with reduced dimensions
            if opts.output:
                os.makedirs(opts.output, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join(opts.output, f"{video_name}_segmented.mp4")
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (output_width, output_height))

            frame_count = 0
            progress_bar = tqdm(total=total_frames)
            stop_triggered = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress_bar.update(1)

                if (frame_count - 1) % opts.skip_frames != 0:
                    continue

                # Process the frame and get result
                segmented_frame, raw_pred, detection_info, should_stop = process_bev_frame(
                    frame, model, transform, device, decode_fn,
                    show_mask_ids=opts.show_mask_ids,
                    use_roi=opts.use_roi,
                    roi_height_fraction=opts.roi_height_fraction,
                    stop_ids=stop_ids
                )

                # Handle stop condition
                if should_stop and not stop_triggered:
                    stop_triggered = True
                    print(f"🛑 STOP CONDITION TRIGGERED at frame {frame_count}!")
                    # Here you can add code to handle the stop condition
                    # For example: pause processing, save frame, alert user, etc.

                # Resize frames for display and output
                if opts.show_preview:
                    h_seg, w_seg = segmented_frame.shape[:2]
                    h_orig, w_orig = frame.shape[:2]
                    
                    # Resize both frames to same size for side-by-side display
                    display_width = min(w_seg, w_orig) // 2
                    display_height = min(h_seg, h_orig) // 2
                    
                    original_display = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    segmented_display = cv2.resize(segmented_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
                    
                    # Draw ROI indicator if using ROI
                    if opts.use_roi:
                        roi_h = int(display_height * opts.roi_height_fraction)
                        roi_coords_display = (0, 0, display_width, roi_h)
                        original_display = draw_roi_indicator(original_display, roi_coords_display)
                    
                    # Add labels to each image
                    cv2.putText(original_display, "Original Video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(segmented_display, "Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add frame counter
                    cv2.putText(original_display, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add stop status
                    if should_stop:
                        cv2.putText(original_display, "STOP!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(segmented_display, "STOP!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Add detected IDs info
                    if detection_info.get('all_detected_ids'):
                        ids_text = f"IDs: {detection_info['all_detected_ids']}"
                        cv2.putText(segmented_display, ids_text, (10, display_height - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Show stop IDs if detected
                        if detection_info.get('stop_ids_detected'):
                            stop_ids_text = f"Stop IDs: {detection_info['stop_ids_detected']}"
                            cv2.putText(segmented_display, stop_ids_text, (10, display_height - 40), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Combine side by side
                    combined_display = np.hstack((original_display, segmented_display))
                    
                    cv2.imshow("Video Processing Results", combined_display)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if opts.output:
                    # Resize frame before writing to output
                    h, w = segmented_frame.shape[:2]
                    resized_frame = cv2.resize(segmented_frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    out_writer.write(resized_frame)

            progress_bar.close()
            cap.release()
            
            if opts.output:
                out_writer.release()
                print(f"Output video saved to: {out_path} (50% size reduction)")

            if opts.show_preview:
                cv2.destroyAllWindows()
            
            # Print summary for this video
            if stop_triggered:
                print(f"⚠️  Stop condition was triggered during processing of {video_name}")
            else:
                print(f"✅ Completed processing {video_name} without stop conditions")
    
    else:
        print("Error: Either --use_camera or --input must be specified")

if __name__ == '__main__':
    main()
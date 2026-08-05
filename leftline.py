from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2
import pickle

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, CustomSegmentation
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a video file or video directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes','custom'], help='Name of training set')
    parser.add_argument("--output", type=str, default=None,
                        help="path to save the segmented video output")
    parser.add_argument("--fps", type=int, default=5,
                        help="frames per second for output video")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="process every n-th frame")
    parser.add_argument("--show_preview", action='store_true', default=False,
                        help="show video preview during processing")

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
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

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
    # Cityscapes: Road=0, Sidewalk=1, Lane=8
    # Note: These IDs may vary by dataset!
    if road_class_id is None and lane_class_id is None:
        # Default road classes for Cityscapes
        road_candidates = [0, 1, 8]
        for class_id in road_candidates:
            mask = np.logical_or(mask, pred == class_id)
    
    return mask.astype(np.uint8) * 255

def find_navigation_line(road_mask, frame_height, frame_width):
    """
    Generate a navigation line based on the road mask
    
    Args:
        road_mask: Binary mask with road/lane areas
        frame_height: Height of the video frame
        frame_width: Width of the video frame
    
    Returns:
        List of points forming the navigation line
    """
    # Define region of interest (bottom half of the image)
    roi_height = frame_height // 2
    roi_y_start = frame_height - roi_height
    
    # We'll generate several points for our navigation line
    nav_points = []
    
    # Start from the bottom of the image
    for y_offset in range(0, roi_height, roi_height // 10):
        y = frame_height - y_offset
        row = road_mask[y-1, :]  # Get the row at this y position
        
        # Find the non-zero elements (road pixels)
        road_pixels = np.where(row > 0)[0]
        
        if len(road_pixels) > 0:
            # Find the middle of the road
            middle_x = int((road_pixels[0] + road_pixels[-1]) / 2)
            nav_points.append((middle_x, y))
    
    # If we don't have enough points, add a default central line
    if len(nav_points) < 2:
        center_x = frame_width // 2
        nav_points = [(center_x, frame_height), (center_x, frame_height - roi_height)]
    
    return nav_points


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

def compute_row_centroids_offset(mask, num_points=3, offset_inside_area=250): #for area (target line)
    """
    For each row, return a point that is 100 pixels *into* the area
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

            # Offset 200 pixels into the area, but not beyond its right edge
            x_offset = min(x_left + offset_inside_area, x_right)
            centroids.append((x_offset, y))

            # Print the centroid coordinates to terminal
            # print(f"x={x_offset}, y={y}")

    return centroids

def compute_row_centroids_center(mask2, num_points=2): #for golf cart (heading vector)
    """
    For each row, compute the horizontal center (mean x) of the mask.
    Suitable for tracking object center like the golf cart.
    """
    height, width = mask2.shape
    centroids = []
    step = max(1, height // num_points)

    for y in range(0, height, step):
        row = mask2[y, :]
        x_indices = np.where(row > 0)[0]
        if len(x_indices) > 0:
            x_center = int(np.mean(x_indices))
            centroids.append((x_center, y))

    return centroids


def process_frame(frame, model, transform, device, decode_fn, show_mask_ids=False, 
                 show_navigation=False, nav_line_color='green', nav_line_thickness=5, 
                 road_class_id=None, lane_class_id=None):
    """Process a frame, return segmented frame with Line1 and Line2 drawn"""
    # Convert input to tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]

    # Decode and convert to OpenCV BGR
    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)

    # Create masks
    drivable_mask = (pred == 0).astype(np.uint8)
    parking_mask = (pred == 5).astype(np.uint8)
    crosswalk_mask = (pred == 6).astype(np.uint8)
    golf_cart_mask = (pred == 8).astype(np.uint8)

    combined_mask = ((pred == 0) | (pred == 5) | (pred == 6)).astype(np.uint8)

    # Calculate centroids across full height
    line1_points = compute_row_centroids_offset(combined_mask, offset_inside_area=250)
    line2_points = compute_row_centroids_center(golf_cart_mask)


    #ไม่จำเป็นต้องแสดงค่าใน terminal ---> ควรให้ ros เป็นคนดึงค่าเพื่อไปใช้ต่อดีกว่า
    # print("\n📌 Line 1: จุด offset จากขอบซ้ายเข้ามาในเลน")
    # for i, (x, y) in enumerate(line1_points):
    #     print(f"Point {i+1}: x={x}, y={y}")

    # print("\n📌 Line 2: จุดกึ่งกลางของเลน")
    # for i, (x, y) in enumerate(line2_points):
    #     print(f"Point {i+1}: x={x}, y={y}")

    # Draw on the segmented image only
    draw_navigation_line(colorized_pred_bgr, line1_points, (0, 255, 0), 2)  # Green
    draw_navigation_line(colorized_pred_bgr, line2_points, (0, 0, 255), 2)      # Red

    return colorized_pred_bgr, pred, None, None  # nav_frame, road_mask_vis not needed


def main():
    opts = get_argparser().parse_args()

    # Set number of classes and decode function
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'custom':
        opts.num_classes = 9
        decode_fn = Cityscapes.decode_target

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Setup video input
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

    # Load model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        state_dict = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)#pickle_module=pickle)
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
        print(f"Processing: {video_name}, {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup video writer
        os.makedirs(opts.output, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(opts.output, f"{video_name}_segmented.mp4")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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
            segmented_frame, raw_pred, _, _ = process_frame(
                frame, model, transform, device, decode_fn,
                show_mask_ids=opts.show_mask_ids,
                show_navigation=True,
                nav_line_color=opts.nav_line_color,
                nav_line_thickness=opts.nav_line_thickness
            )

            output_frame = segmented_frame  # Already has Line1 and Line2 drawn

            if opts.show_preview:
                cv2.imshow("Segmented Output", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out_writer.write(output_frame)

        progress_bar.close()
        cap.release()
        out_writer.release()

        print(f"Output video saved to: {out_path}")

        if opts.show_preview:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, Bool
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

# Import needed modules from your existing code
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h

class IDPublisher(Node):
    def __init__(self):
        super().__init__('id_publisher')
        
        # ROS2 Publishers and Subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detected_ids_publisher = self.create_publisher(Int32MultiArray, '/detected_ids', 10)
        self.stop_status_publisher = self.create_publisher(Bool, '/stop_status', 10)
        
        # Parameters
        self.declare_parameter('stop_ids', [0, 2, 4, 6])  # Default stop IDs
        self.declare_parameter('normal_velocity', 1.0)     # Normal forward velocity
        self.declare_parameter('stop_velocity', 0.0)       # Stop velocity
        self.declare_parameter('publish_rate', 10.0)       # Hz
        self.declare_parameter('use_camera', True)         # Use camera or video file
        self.declare_parameter('video_path', '')           # Path to video file if not using camera
        self.declare_parameter('model_path', './checkpoints/best2700.pth')
        self.declare_parameter('dataset', 'custom')
        self.declare_parameter('model_name', 'deeplabv3plus_mobilenet')
        self.declare_parameter('num_classes', 9)
        self.declare_parameter('use_roi', True)            # Use ROI processing
        self.declare_parameter('roi_height_fraction', 0.33) # ROI height fraction
        
        # Get parameters
        self.stop_ids = self.get_parameter('stop_ids').get_parameter_value().integer_array_value
        self.normal_velocity = self.get_parameter('normal_velocity').get_parameter_value().double_value
        self.stop_velocity = self.get_parameter('stop_velocity').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.use_camera = self.get_parameter('use_camera').get_parameter_value().bool_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.dataset = self.get_parameter('dataset').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.num_classes = self.get_parameter('num_classes').get_parameter_value().integer_value
        self.use_roi = self.get_parameter('use_roi').get_parameter_value().bool_value
        self.roi_height_fraction = self.get_parameter('roi_height_fraction').get_parameter_value().double_value
        
        self.get_logger().info(f'Stop IDs: {self.stop_ids}')
        self.get_logger().info(f'Normal velocity: {self.normal_velocity} m/s')
        self.get_logger().info(f'Stop velocity: {self.stop_velocity} m/s')
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        self.get_logger().info(f'Use camera: {self.use_camera}')
        self.get_logger().info(f'Use ROI: {self.use_roi}')
        
        # Initialize segmentation model
        self.setup_model()
        
        # Initialize BEV processor or video capture
        if self.use_camera:
            self.setup_bev_processor()
        else:
            self.setup_video_capture()
        
        # Control variables
        self.current_velocity = self.normal_velocity
        self.detected_ids = []
        self.stop_status = False
        self.last_detection_time = time.time()
        
        # Create timer for main processing loop
        self.timer = self.create_timer(1.0 / self.publish_rate, self.process_and_publish)
        
        self.get_logger().info('ID Publisher initialized successfully')
    
    def setup_model(self):
        """Initialize the segmentation model"""
        try:
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.get_logger().info(f'Using device: {self.device}')
            
            # Set decode function based on dataset
            if self.dataset.lower() == 'voc':
                from datasets import VOCSegmentation
                self.decode_fn = VOCSegmentation.decode_target
            elif self.dataset.lower() == 'cityscapes':
                from datasets import Cityscapes
                self.decode_fn = Cityscapes.decode_target
            elif self.dataset.lower() == 'custom':
                from datasets import Cityscapes
                self.decode_fn = Cityscapes.decode_target
            
            # Load model
            self.model = network.modeling.__dict__[self.model_name](
                num_classes=self.num_classes, output_stride=16
            )
            utils.set_bn_momentum(self.model.backbone, momentum=0.01)
            
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                self.model = nn.DataParallel(self.model)
                self.model.to(self.device)
                self.model.eval()
                self.get_logger().info(f'Model loaded from: {self.model_path}')
            else:
                self.get_logger().error(f'Model file not found: {self.model_path}')
                raise FileNotFoundError(f'Model file not found: {self.model_path}')
            
            # Setup transforms
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup model: {str(e)}')
            raise
    
    def setup_bev_processor(self):
        """Initialize BEV processor for camera input"""
        try:
            # Default camera IDs - you may need to adjust these
            video_paths = {
                "front": 2,
                "left": 4,
                "rear": 0,
                "right": 6,
            }
            
            from object import BEVProcessor  # Import from your existing code
            self.bev_processor = BEVProcessor(
                video_paths, 
                img_car,
                display_width=800,
                display_height=600
            )
            self.get_logger().info('BEV processor initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup BEV processor: {str(e)}')
            raise
    
    def setup_video_capture(self):
        """Initialize video capture for file input"""
        try:
            if not self.video_path:
                self.get_logger().error('Video path not specified')
                raise ValueError('Video path not specified')
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.get_logger().error(f'Failed to open video: {self.video_path}')
                raise ValueError(f'Failed to open video: {self.video_path}')
            
            self.get_logger().info(f'Video capture initialized: {self.video_path}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup video capture: {str(e)}')
            raise
    
    def extract_roi(self, frame, roi_height_fraction=0.33):
        """Extract Region of Interest from the top portion of the frame"""
        h, w = frame.shape[:2]
        roi_height = int(h * roi_height_fraction)
        roi_frame = frame[0:roi_height, 0:w]
        roi_coords = (0, 0, w, roi_height)
        return roi_frame, roi_coords
    
    def create_full_frame_result(self, roi_result, original_frame_shape, roi_coords):
        """Create a full-frame result by placing ROI result in correct position"""
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
    
    def process_frame(self, frame):
        """Process a single frame and return detected IDs"""
        try:
            roi_coords = None
            
            if self.use_roi:
                # Extract ROI from the frame
                roi_frame, roi_coords = self.extract_roi(frame, self.roi_height_fraction)
                processing_frame = roi_frame
            else:
                processing_frame = frame
            
            # Convert input to tensor
            frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                pred = self.model(input_tensor).max(1)[1].cpu().numpy()[0]
            
            # If using ROI, create full-frame result
            if self.use_roi:
                pred = self.create_full_frame_result(pred, frame.shape[:2], roi_coords)
            
            # Get unique IDs from prediction
            unique_ids = np.unique(pred).tolist()
            
            return unique_ids, pred
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')
            return [], None
    
    def check_stop_condition(self, detected_ids):
        """Check if any stop IDs are detected"""
        stop_ids_detected = [id for id in detected_ids if id in self.stop_ids]
        should_stop = len(stop_ids_detected) > 0
        return should_stop, stop_ids_detected
    
    def get_frame(self):
        """Get the next frame from either BEV processor or video capture"""
        try:
            if self.use_camera:
                bev_frame, _ = self.bev_processor.get_bev_frame()
                return bev_frame
            else:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        return None
                return frame
        except Exception as e:
            self.get_logger().error(f'Error getting frame: {str(e)}')
            return None
    
    def process_and_publish(self):
        """Main processing loop - get frame, process, and publish commands"""
        try:
            # Get frame
            frame = self.get_frame()
            if frame is None:
                self.get_logger().warn('No frame available')
                return
            
            # Process frame to get detected IDs
            detected_ids, pred_mask = self.process_frame(frame)
            
            # Check stop condition
            should_stop, stop_ids_detected = self.check_stop_condition(detected_ids)
            
            # Update control variables
            self.detected_ids = detected_ids
            self.stop_status = should_stop
            self.last_detection_time = time.time()
            
            # Determine velocity
            if should_stop:
                self.current_velocity = self.stop_velocity
                self.get_logger().warn(f'STOP condition triggered! Detected stop IDs: {stop_ids_detected}')
            else:
                self.current_velocity = self.normal_velocity
            
            # Create and publish cmd_vel message
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = self.current_velocity
            cmd_vel_msg.linear.y = 0.0
            cmd_vel_msg.linear.z = 0.0
            cmd_vel_msg.angular.x = 0.0
            cmd_vel_msg.angular.y = 0.0
            cmd_vel_msg.angular.z = 0.0
            
            self.cmd_vel_publisher.publish(cmd_vel_msg)
            
            # Publish detected IDs
            ids_msg = Int32MultiArray()
            ids_msg.data = detected_ids
            self.detected_ids_publisher.publish(ids_msg)
            
            # Publish stop status
            stop_msg = Bool()
            stop_msg.data = should_stop
            self.stop_status_publisher.publish(stop_msg)
            
            # Log status (throttled)
            if int(time.time()) % 5 == 0:  # Log every 5 seconds
                self.get_logger().info(
                    f'Velocity: {self.current_velocity:.2f} m/s, '
                    f'Detected IDs: {detected_ids}, '
                    f'Stop Status: {should_stop}'
                )
            
        except Exception as e:
            self.get_logger().error(f'Error in processing loop: {str(e)}')
            # Publish stop command as safety measure
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = 0.0
            self.cmd_vel_publisher.publish(cmd_vel_msg)
    
    def destroy_node(self):
        """Clean up resources when shutting down"""
        try:
            if hasattr(self, 'bev_processor'):
                self.bev_processor.release()
            if hasattr(self, 'cap'):
                self.cap.release()
            self.get_logger().info('Resources cleaned up')
        except Exception as e:
            self.get_logger().error(f'Error cleaning up resources: {str(e)}')
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        id_publisher = IDPublisher()
        
        # Run the node
        rclpy.spin(id_publisher)
        
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        if 'id_publisher' in locals():
            id_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
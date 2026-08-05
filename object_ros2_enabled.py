# ========== IMPORTS ==========
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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray, Bool

from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h

# ========== ROS2 CMD_VEL CLASS ==========
class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher_from_object')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detected_ids_publisher = self.create_publisher(Int32MultiArray, '/detected_ids', 10)
        self.stop_status_publisher = self.create_publisher(Bool, '/stop_status', 10)

    def publish_cmd(self, detected_ids):
        stop_ids = [2, 3]
        should_stop = any(i in stop_ids for i in detected_ids)

        msg = Twist()
        msg.linear.x = 0.0 if should_stop else 1.0
        self.cmd_vel_publisher.publish(msg)

        ids_msg = Int32MultiArray()
        ids_msg.data = detected_ids
        self.detected_ids_publisher.publish(ids_msg)

        stop_msg = Bool()
        stop_msg.data = should_stop
        self.stop_status_publisher.publish(stop_msg)

        print(f"[ROS2] Published cmd_vel.linear.x = {msg.linear.x} (IDs: {detected_ids})")

# ========== MAIN FUNCTION ==========
def main():
    from object import get_argparser, process_bev_frame, BEVProcessor
    opts = get_argparser().parse_args()

    stop_ids = []
    if opts.stop_on_ids:
        stop_ids = [int(id.strip()) for id in opts.stop_on_ids.split(',') if id.strip()]

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

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        state_dict = torch.load(opts.ckpt, map_location=device)
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

    # ROS2 START
    rclpy.init()
    cmd_node = CmdVelPublisher()

    try:
        while True:
            bev_frame, camera_display = bev_processor.get_bev_frame()
            if bev_frame is None:
                break

            segmented_frame, raw_pred, detection_info, should_stop = process_bev_frame(
                bev_frame, model, transform, device, decode_fn,
                show_mask_ids=opts.show_mask_ids,
                use_roi=opts.use_roi,
                roi_height_fraction=opts.roi_height_fraction,
                stop_ids=stop_ids
            )

            detected_ids = np.unique(raw_pred).tolist()
            print(f"[ROS2 DEBUG] Publishing cmd_vel.linear.x based on IDs: {detected_ids}")
            cmd_node.publish_cmd(detected_ids)
            rclpy.spin_once(cmd_node, timeout_sec=0.01)

            if opts.show_preview:
                # Calculate new dimensions for 50% resize
                original_width_seg = segmented_frame.shape[1]
                original_height_seg = segmented_frame.shape[0]
                new_width_seg = int(original_width_seg * 0.5)
                new_height_seg = int(original_height_seg * 0.5)

                original_width_bev = bev_frame.shape[1]
                original_height_bev = bev_frame.shape[0]
                new_width_bev = int(original_width_bev * 0.5)
                new_height_bev = int(original_height_bev * 0.5)

                seg_resized = cv2.resize(segmented_frame, (new_width_seg, new_height_seg))
                bev_resized = cv2.resize(bev_frame, (new_width_bev, new_height_bev))
                combo = np.hstack((bev_resized, seg_resized))
                cv2.imshow("BEV and Segmentation", combo)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        bev_processor.release()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
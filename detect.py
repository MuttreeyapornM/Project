import torch
import torch.nn as nn
import argparse
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms as T
import rclpy
from ros2_publish import PathPublisher

from image_processing import ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h

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
    parser.add_argument("--show_preview", action='store_true', default=True)
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
    return parser

def process_bev_frame(bev_frame, model, transform, device, decode_fn, show_mask_ids=False, ros_node=None):
    frame_rgb = cv2.cvtColor(bev_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]

    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)

    height = pred.shape[0]
    roi_height = height // 3
    roi_pred = pred[:roi_height, :]

    colorized_pred_bgr[:roi_height, :, 0] = 0
    colorized_pred_bgr[:roi_height, :, 2] = 255

    print("🔁 Processing new frame...")
    if show_mask_ids:
        unique_ids = np.unique(roi_pred)
        print(f"\n📌 ROI (Top 1/3): Mask IDs detected → {unique_ids}")
        if ros_node:
            ros_node.publish_roi_ids(unique_ids.tolist())

    return colorized_pred_bgr, pred, bev_frame.copy(), None

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

    def initialize_video_capture(self, path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    def get_bev_frame(self):
        cam_names = ["Front", "Left", "Rear", "Right"]
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
            images.append(frame)
            warped_rgba_.append(frame)

        final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
        merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)
        return merged_car_image, None

    def release(self):
        for cap in self.caps.values():
            cap.release()

def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == 'custom':
        opts.num_classes = 9
        from datasets import CustomSegmentation
        decode_fn = CustomSegmentation.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from network import modeling
    model = modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        from network import convert_to_separable_conv
        convert_to_separable_conv(model.classifier)
    from utils import set_bn_momentum
    set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt and os.path.isfile(opts.ckpt):
        state_dict = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rclpy.init()
    publisher_node = PathPublisher()

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
                    ros_node=publisher_node
                )

                if opts.show_preview:
                    resized = cv2.resize(segmented_frame, (opts.display_width, opts.display_height))
                    cv2.imshow("Segmentation", resized)
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
                    ros_node=publisher_node
                )

                if opts.show_preview:
                    resized = cv2.resize(segmented_frame, (opts.display_width, opts.display_height))
                    cv2.imshow("Segmentation", resized)
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



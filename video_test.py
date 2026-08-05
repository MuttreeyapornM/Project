# from torch.utils.data import dataset
# from tqdm import tqdm
# import network
# import utils
# import os
# import random
# import argparse
# import numpy as np
# import cv2

# from torch.utils import data
# from datasets import VOCSegmentation, Cityscapes, CustomSegmentation
# from torchvision import transforms as T
# from metrics import StreamSegMetrics

# import torch
# import torch.nn as nn

# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
# from glob import glob

# def get_argparser():
#     parser = argparse.ArgumentParser()

#     # Datset Options
#     parser.add_argument("--input", type=str, required=True,
#                         help="path to a single image or video file")
#     parser.add_argument("--dataset", type=str, default='voc',
#                         choices=['voc', 'cityscapes','custom'], help='Name of training set')

#     # Deeplab Options
#     available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
#                               not (name.startswith("__") or name.startswith('_')) and callable(
#                               network.modeling.__dict__[name])
#                               )

#     parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
#                         choices=available_models, help='model name')
#     parser.add_argument("--separable_conv", action='store_true', default=False,
#                         help="apply separable conv to decoder and aspp")
#     parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

#     # Train Options
#     parser.add_argument("--save_val_results_to", default=None,
#                         help="save segmentation results to the specified dir")

#     parser.add_argument("--crop_val", action='store_true', default=False,
#                         help='crop validation (default: False)')
#     parser.add_argument("--crop_size", type=int, default=513)
    
#     parser.add_argument("--ckpt", default=None, type=str,
#                         help="resume from checkpoint")
#     parser.add_argument("--gpu_id", type=str, default='0',
#                         help="GPU ID")
#     return parser

# def preprocess(frame, input_size):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image)
#     transform = T.Compose([
#         T.Resize(input_size),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225])
#     ])
#     return transform(image).unsqueeze(0)  # shape: (1, 3, H, W)

# def decode_segmap(output, num_classes=21):
#     label_colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
#     r = label_colors[output][:, :, 0]
#     g = label_colors[output][:, :, 1]
#     b = label_colors[output][:, :, 2]
#     return np.stack([r, g, b], axis=2)

# def main():
#     opts = get_argparser().parse_args()

#     if opts.dataset.lower() == 'voc':
#         opts.num_classes = 21
#         decode_fn = VOCSegmentation.decode_target
#     elif opts.dataset.lower() == 'cityscapes':
#         opts.num_classes = 19
#         decode_fn = Cityscapes.decode_target
#     elif opts.dataset.lower() == 'custom':
#         opts.num_classes = 9
#         decode_fn = Cityscapes.decode_target

#     os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device: %s" % device)

#     # Set up model
#     model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
#     if opts.separable_conv and 'plus' in opts.model:
#         network.convert_to_separable_conv(model.classifier)
#     utils.set_bn_momentum(model.backbone, momentum=0.01)

#     if opts.ckpt is not None and os.path.isfile(opts.ckpt):
#         checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
#         model.load_state_dict(checkpoint["model_state"])
#         model = nn.DataParallel(model)
#         model.to(device)
#         print("Resume model from %s" % opts.ckpt)
#         del checkpoint
#     else:
#         print("[!] Retrain")
#         model = nn.DataParallel(model)
#         model.to(device)

#     # Load video
#     cap = cv2.VideoCapture(opts.input)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     out_path = os.path.join(opts.save_val_results_to, "output_video.mp4")
#     os.makedirs(opts.save_val_results_to, exist_ok=True)

#     out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     # Define image transformation
#     if opts.crop_val:
#         transform = T.Compose([
#             T.Resize(opts.crop_size),
#             T.CenterCrop(opts.crop_size),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
#         ])
#     else:
#         transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
#         ])

#     with torch.no_grad():
#         model = model.eval()
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             img = preprocess(frame, (height, width)).to(device)

#             # Get predictions
#             pred = model(img).max(1)[1].cpu().numpy()[0]  # HW

#             # Show mask IDs
#             print(f"Mask IDs: {np.unique(pred)}")  # แสดง ID ของ mask ที่เป็น unique

#             # Colorize prediction mask
#             colorized_preds = decode_segmap(pred, num_classes=opts.num_classes)
#             colorized_preds = cv2.resize(colorized_preds, (width, height))
#             overlay = cv2.addWeighted(frame, 0.5, colorized_preds, 0.5, 0)
#             out_writer.write(overlay)

#     cap.release()
#     out_writer.release()
#     print(f"✅ วิดีโอผลลัพธ์บันทึกที่: {out_path}")

# if __name__ == '__main__':
#     main()

#---------------------------------v2-----------------------------------------
from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2

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
    parser.add_argument("--dataset", type=str, default='voc',
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

def process_frame(frame, model, transform, device, decode_fn, show_mask_ids=False):
    """Process a single frame through the segmentation model"""
    # Convert from BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Apply transformations
    input_tensor = transform(pil_image).unsqueeze(0)  # To tensor of NCHW
    input_tensor = input_tensor.to(device)
    
    # Get predictions
    with torch.no_grad():
        pred = model(input_tensor).max(1)[1].cpu().numpy()[0]  # HW
    
    # Show mask IDs if requested
    if show_mask_ids:
        unique_ids = np.unique(pred)
        print(f"Frame mask IDs: {unique_ids}")
    
    # Colorize prediction mask
    colorized_pred = decode_fn(pred).astype('uint8')
    
    # Convert back to OpenCV format (BGR)
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)
    
    return colorized_pred_bgr, pred

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'custom':
        opts.num_classes = 9
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup video input
    video_files = []
    if os.path.isdir(opts.input):
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            files = glob(os.path.join(opts.input, f"*.{ext}"))
            if len(files) > 0:
                video_files.extend(files)
    elif os.path.isfile(opts.input):
        video_files.append(opts.input)
    
    if not video_files:
        print("No video files found!")
        return
    
    # Set up model (all models are constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
    
    model = model.eval()
    
    for video_path in video_files:
        # Open video file
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Couldn't open video file {video_path}")
            continue
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if opts.fps is None else opts.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_name}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup video writer if output path is specified
        video_writer = None
        if opts.output is not None:
            os.makedirs(opts.output, exist_ok=True)
            output_path = os.path.join(opts.output, f"{video_name}_segmented.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        progress_bar = tqdm(total=total_frames)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress_bar.update(1)
            
            # Skip frames if needed
            if (frame_count - 1) % opts.skip_frames != 0:
                continue
            
            # Process frame
            segmented_frame, raw_pred = process_frame(frame, model, transform, device, decode_fn, opts.show_mask_ids)
            
            # Create output frame
            if opts.overlay:
                # Blend segmentation with original frame
                output_frame = cv2.addWeighted(frame, 1 - opts.overlay_alpha, segmented_frame, opts.overlay_alpha, 0)
            else:
                output_frame = segmented_frame
            
            # Show preview if requested
            if opts.show_preview:
                cv2.imshow('Segmentation Preview', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write to output video
            if video_writer is not None:
                video_writer.write(output_frame)
        
        progress_bar.close()
        cap.release()
        
        if video_writer is not None:
            video_writer.release()
            print(f"Segmented video saved to: {output_path}")
        
        if opts.show_preview:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

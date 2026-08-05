#-------------------------------Vehicle Heading-------------------------------
from tqdm import tqdm
import network
import utils
import os
import argparse
import numpy as np
import cv2
import json
from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image
from glob import glob
from datasets import VOCSegmentation, Cityscapes, CustomSegmentation
from scipy.interpolate import CubicSpline

# ----------------- GLOBALS -----------------
prev_spline_points = None
alpha_smooth = 0.6  # ค่า smooth spline ระหว่างเฟรม (0-1)


# ===============================================================
# 🧱 ฟังก์ชันเตรียม argument
# ===============================================================
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to a video file, video directory, or image file/directory")
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['voc', 'cityscapes', 'custom'])
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--fps", type=int, default=5, help="fps for video output (ignored for images)")
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--show_preview", action='store_true', default=False)

    available_models = sorted(name for name in network.modeling.__dict__
                              if name.islower() and callable(network.modeling.__dict__[name]))

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet', choices=available_models)
    parser.add_argument("--separable_conv", action='store_true', default=False)
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--gpu_id", type=str, default='0')

    parser.add_argument("--conf_thresh", type=float, default=0.7, help="minimum confidence threshold for drivable mask")
    parser.add_argument("--lookahead_distance", type=int, default=200, help="lookahead distance for waypoint selection (pixels)")
    return parser


# ===============================================================
# 🧹 ฟังก์ชันทำความสะอาด mask
# ===============================================================
def clean_drivable_mask(mask, valid_ids=[0, 5, 6]):
    combined_mask = np.isin(mask, valid_ids).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_mask = np.zeros_like(combined_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(largest_mask, [largest], -1, 255, -1)
    else:
        largest_mask = combined_mask

    return (largest_mask > 0).astype(np.uint8)


# ===============================================================
# 🚗 ฟังก์ชันตรวจจับตัวรถและหาจุดกึ่งกลาง + มุม heading (ใช้ ROI 100x100)
# ===============================================================
def detect_vehicle_and_heading(pred, height, width):
    """
    ตรวจจับพื้นที่รถ (label 8) และคำนวณจุดกึ่งกลางพร้อมทิศทาง heading
    โดยใช้ ROI ขนาด 100x100 พิกเซลตรงกึ่งกลางเฟรม
    
    Returns:
        vehicle_center: (x, y) จุดกึ่งกลางของรถ
        heading_vector: (dx, dy) vector ทิศทาง heading (ชี้ขึ้น)
        vehicle_mask: binary mask ของตัวรถ (เฉพาะใน ROI)
        roi_rect: (x1, y1, x2, y2) พิกัด ROI สำหรับวาดกรอบ
    """
    # กำหนดขนาด ROI
    roi_size = 100
    
    # คำนวณตำแหน่งกึ่งกลางเฟรม
    center_x = width // 2
    center_y = height // 2
    
    # คำนวณขอบเขต ROI 
    x1 = max(0, center_x - roi_size // 2)
    y1 = max(0, center_y - roi_size // 2)
    x2 = min(width, center_x + roi_size // 2)
    y2 = min(height, center_y + roi_size // 2)
    
    # สร้าง ROI จาก prediction mask
    roi_pred = pred[y1:y2, x1:x2]
    
    # สร้าง mask ของรถ (label 8) เฉพาะใน ROI
    vehicle_mask_roi = (roi_pred == 8).astype(np.uint8) * 255
    
    # สร้าง full-size mask สำหรับ visualization (ทั้งภาพ แต่มีเฉพาะ ROI)
    vehicle_mask = np.zeros((height, width), dtype=np.uint8)
    vehicle_mask[y1:y2, x1:x2] = vehicle_mask_roi
    
    if vehicle_mask_roi.sum() == 0:
        # ถ้าไม่เจอรถใน ROI ให้ใช้จุดกลาง ROI แทน
        vehicle_center = (center_x, center_y)
        heading_vector = (0, -50)  # ชี้ขึ้น
        return vehicle_center, heading_vector, vehicle_mask, (x1, y1, x2, y2)
    
    # หา contour ของรถใน ROI
    contours, _ = cv2.findContours(vehicle_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        vehicle_center = (center_x, center_y)
        heading_vector = (0, -50)
        return vehicle_center, heading_vector, vehicle_mask, (x1, y1, x2, y2)
    
    # เลือก contour ที่ใหญ่ที่สุด
    largest_contour = max(contours, key=cv2.contourArea)
    
    # หา bounding box ของรถใน ROI (coordinate ท้องถิ่น)
    x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(largest_contour)
    
    # แปลงกลับเป็น coordinate ของภาพเต็ม
    vehicle_center = (x1 + x_roi + w_roi // 2, y1 + y_roi + h_roi // 2)
    
    # สร้าง heading vector (เส้นแบ่งครึ่งรถในแนวตั้ง ชี้ขึ้น)
    # ความยาวลูกศรประมาณ 40% ของความสูงรถ (หรืออย่างน้อย 30 พิกเซล)
    arrow_length = max(100, int(h_roi * 0.4))
    heading_vector = (0, -arrow_length)  # ชี้ขึ้น (y ติดลบ)
    
    return vehicle_center, heading_vector, vehicle_mask, (x1, y1, x2, y2)


# ===============================================================
# 🧭 ฟังก์ชันหลักสำหรับประมวลผลแต่ละเฟรม (อัพเดต)
# ===============================================================
def process_frame(frame, model, transform, device, decode_fn, conf_thresh=0.7, lookahead_distance=200):
    """
    ประมวลผลเฟรมจากวิดีโอ/ภาพ → segmentation → ตรวจจับรถใน ROI 100x100 → สร้าง spline → overlay
    
    ROI 100x100: สร้างพื้นที่สี่เหลี่ยมขนาด 100x100 พิกเซลตรงกึ่งกลางเฟรม
                 ใช้สำหรับตรวจจับตำแหน่งและทิศทางรถ
    """
    global prev_spline_points

    # แปลงภาพเป็น RGB และ tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Inference ด้วยโมเดล
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        pred = np.argmax(probs, axis=0)

    # แปลงเป็นภาพสีตาม class
    colorized_pred = decode_fn(pred).astype('uint8')
    colorized_pred_bgr = cv2.cvtColor(colorized_pred, cv2.COLOR_RGB2BGR)

    # ✅ สร้าง overlay แบบโปร่งใส
    overlay = cv2.addWeighted(frame, 0.35, colorized_pred_bgr, 0.65, 0)

    height, width = pred.shape
    
    # ===============================================================
    # 🛣️ คำนวณ mask พื้นที่ขับขี่ได้
    # ===============================================================
    # คำนวณ mask สำหรับพื้นที่ขับเคลื่อน
    drivable_probs = probs[[0, 5, 6], :, :].max(axis=0)
    conf_mask = (drivable_probs > conf_thresh).astype(np.uint8)

    # ทำความสะอาด mask
    clean_mask = clean_drivable_mask(pred, valid_ids=[0, 5, 6])
    final_mask = cv2.bitwise_and(clean_mask, conf_mask)
    
    # ===============================================================
    # 🚗 1. ตรวจจับตัวรถและวาดลูกศร heading ภายใน ROI
    # ===============================================================
    vehicle_center, heading_vector, vehicle_mask, roi_rect = detect_vehicle_and_heading(pred, height, width)
    
    # ===============================================================
    # 🚗 2. วาดลูกศร heading และจุดกึ่งกลางรถ
    # ===============================================================
    # คำนวณจุดปลายลูกศร
    arrow_end = (vehicle_center[0] + heading_vector[0], 
                 vehicle_center[1] + heading_vector[1])
    
    # วาดลูกศร heading (สีม่วง)
    cv2.arrowedLine(overlay, vehicle_center, arrow_end, (255, 0, 255), 3, tipLength=0.3)
    
    # วาดจุดกึ่งกลางรถ (สีเขียวสว่าง)
    cv2.circle(overlay, vehicle_center, 8, (0, 255, 0), -1)
    cv2.putText(overlay, "Vehicle", (vehicle_center[0] + 10, vehicle_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ===============================================================
    # 🛣️ 3. สร้าง Path Planning
    # ===============================================================
    # เลือกพื้นที่สนใจ (ROI) ด้านล่างของภาพ
    roi_height = height // 3
    roi_mask = final_mask[:roi_height, :]

    # หาจุดนำทาง (navigation points)
    nav_points = []
    num_points = 5
    step = max(1, roi_height // num_points)
    for y in range(0, roi_height, step):
        row = roi_mask[y, :]
        x_indices = np.where(row > 0)[0]
        if len(x_indices) > 0:
            x_left, x_right = x_indices[0], x_indices[-1]
            x_offset = min(x_left + lookahead_distance, x_right)
            nav_points.append((x_offset, y))

    spline_points = []
    interesting_point = None

    # สร้างเส้น spline
    if len(nav_points) >= 3:
        pts = np.array(nav_points)
        distances = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        t = np.insert(np.cumsum(distances), 0, 0)

        spline_x = CubicSpline(t, pts[:, 0])
        spline_y = CubicSpline(t, pts[:, 1])

        t_new = np.linspace(t[0], t[-1], 200)
        spline_points = np.stack((spline_x(t_new), spline_y(t_new)), axis=-1).astype(int)

        # Temporal smoothing
        if prev_spline_points is not None and len(prev_spline_points) == len(spline_points):
            spline_points = (alpha_smooth * prev_spline_points + (1 - alpha_smooth) * spline_points).astype(int)
        prev_spline_points = spline_points

        # วาดเส้น spline (สีฟ้าสว่าง)
        for i in range(len(spline_points) - 1):
            cv2.line(overlay, tuple(spline_points[i]), tuple(spline_points[i + 1]), (255, 255, 0), 2)

        # จุดสนใจ (waypoint)
        mid_idx = len(spline_points) // 2
        interesting_point = tuple(spline_points[mid_idx])
        cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
        cv2.putText(overlay, "Waypoint", (interesting_point[0] + 10, interesting_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # ===============================================================
        # 📐 คำนวณมุมระหว่าง Waypoint และ Vehicle Center
        # ===============================================================
        # วาดเส้นเชื่อมจาก Waypoint (สีแดง) ไป Vehicle Center (สีเขียว)
        cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)  # สีม่วงอ่อน
        
        # คำนวณมุม
        # Vector จาก vehicle center ไป waypoint
        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        
        # Vector ของ heading (ชี้ขึ้น = 0 องศา)
        heading_dx = heading_vector[0]
        heading_dy = heading_vector[1]
        
        # คำนวณมุมด้วย atan2
        # มุมของเส้นไป waypoint (เทียบกับแกน x)
        angle_to_waypoint = np.arctan2(-dy, dx)  # ลบ dy เพราะ y เพิ่มลงล่าง
        
        # มุมของ heading (เทียบกับแกน x)
        angle_heading = np.arctan2(-heading_dy, heading_dx)
        
        # คำนวณมุมสัมพัทธ์ (ต่างมุม)
        angle_diff = angle_to_waypoint - angle_heading
        
        # แปลงเป็นองศาและปรับให้อยู่ในช่วง -180 ถึง 180
        angle_deg = np.degrees(angle_diff)
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360
        
        # แสดงมุม (ซ้าย = บวก, ขวา = ลบ)
        angle_text = f"Angle: {angle_deg:.1f} deg"
        text_position = (vehicle_center[0] - 80, vehicle_center[1] + 30)
        cv2.putText(overlay, angle_text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    elif len(nav_points) > 0:
        interesting_point = nav_points[0]
        cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
        
        # คำนวณมุมสำหรับกรณีมี 1 จุด
        cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)
        
        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        heading_dx = heading_vector[0]
        heading_dy = heading_vector[1]
        
        angle_to_waypoint = np.arctan2(-dy, dx)
        angle_heading = np.arctan2(-heading_dy, heading_dx)
        angle_diff = angle_to_waypoint - angle_heading
        angle_deg = np.degrees(angle_diff)
        
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360
        
        angle_text = f"Angle: {angle_deg:.1f} deg"
        text_position = (vehicle_center[0] - 80, vehicle_center[1] - 30)
        cv2.putText(overlay, angle_text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        direction = "LEFT" if angle_deg > 0 else "RIGHT" if angle_deg < 0 else "STRAIGHT"
        cv2.putText(overlay, direction, (text_position[0], text_position[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # ===============================================================
    # ส่งค่ากลับ
    # ===============================================================
    if isinstance(spline_points, np.ndarray):
        spline_points_out = spline_points.tolist()
    else:
        spline_points_out = spline_points

    return overlay, pred, spline_points_out, vehicle_center, interesting_point


# ===============================================================
# 📸 ฟังก์ชันประมวลผลภาพ (ใหม่)
# ===============================================================
def process_image(image_path, model, transform, device, decode_fn, opts):
    """
    ประมวลผลภาพเดียว และบันทึกผล visualization
    """
    print(f"Processing image: {image_path}")
    
    # อ่านภาพ
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot read image: {image_path}")
        return
    
    # ประมวลผล
    result_frame, _, _, _, _ = process_frame(
        frame, model, transform, device, decode_fn,
        conf_thresh=opts.conf_thresh,
        lookahead_distance=opts.lookahead_distance
    )
    
    # บันทึกผล
    image_name = os.path.basename(image_path).split('.')[0]
    out_path = os.path.join(opts.output, f"{image_name}_navigation.jpg")
    cv2.imwrite(out_path, result_frame)
    print(f"✅ Image saved: {out_path}")
    
    # แสดงตัวอย่าง (ถ้าเปิด)
    if opts.show_preview:
        cv2.imshow("Navigation View", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===============================================================
# 🎬 ฟังก์ชันประมวลผลวิดีโอ (อัพเดต - ไม่บันทึก JSON)
# ===============================================================
def process_video(video_path, model, transform, device, decode_fn, opts):
    """
    ประมวลผลวิดีโอ และบันทึกเฉพาะ visualization
    """
    global prev_spline_points
    prev_spline_points = None  # Reset สำหรับวิดีโอใหม่
    
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = opts.fps if opts.fps else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(opts.output, f"{video_name}_navigation.mp4")
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    progress = tqdm(total=total_frames, desc=f"Processing {video_name}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        progress.update(1)

        if (frame_count - 1) % opts.skip_frames != 0:
            continue

        result_frame, _, _, _, _ = process_frame(
            frame, model, transform, device, decode_fn,
            conf_thresh=opts.conf_thresh,
            lookahead_distance=opts.lookahead_distance
        )

        out_writer.write(result_frame)

        if opts.show_preview:
            cv2.imshow("Navigation View", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    progress.close()
    cap.release()
    out_writer.release()

    print(f"✅ Video saved: {out_path}")

    if opts.show_preview:
        cv2.destroyAllWindows()


# ===============================================================
# 🧩 main
# ===============================================================
def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    else:
        opts.num_classes = 9
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.ckpt and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model).to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(opts.output, exist_ok=True)

    # ===============================================================
    # 📂 ตรวจสอบ input type (วิดีโอ หรือ ภาพ)
    # ===============================================================
    input_files = []
    
    if os.path.isdir(opts.input):
        # Directory: รวบรวมทั้งวิดีโอและภาพ
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            input_files.extend([(f, 'video') for f in glob(os.path.join(opts.input, f"*.{ext}"))])
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            input_files.extend([(f, 'image') for f in glob(os.path.join(opts.input, f"*.{ext}"))])
    
    elif os.path.isfile(opts.input):
        # Single file: ตรวจสอบนามสกุล
        ext = opts.input.split('.')[-1].lower()
        if ext in ['mp4', 'avi', 'mov', 'mkv']:
            input_files.append((opts.input, 'video'))
        elif ext in ['jpg', 'jpeg', 'png', 'bmp']:
            input_files.append((opts.input, 'image'))
        else:
            print(f"❌ Unsupported file format: {ext}")
            return

    if not input_files:
        print("❌ No valid input files found!")
        return

    # ===============================================================
    # 🔄 ประมวลผลแต่ละไฟล์
    # ===============================================================
    for file_path, file_type in input_files:
        if file_type == 'video':
            process_video(file_path, model, transform, device, decode_fn, opts)
        elif file_type == 'image':
            process_image(file_path, model, transform, device, decode_fn, opts)


if __name__ == "__main__":
    main()
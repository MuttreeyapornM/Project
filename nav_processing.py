import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from scipy.interpolate import CubicSpline

try:
    import rclpy  # noqa: F401

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


prev_spline_points = None
alpha_smooth = 0.6

WHEELBASE = 1.67
MAX_STEERING_ANGLE = 50.0
ROI_W = 200
ROI_H = 310

SAFETY_CHECK_INTERVAL = 1.0
SAFETY_TIMEOUT = 3.0
MODE_LABEL = "FULL"
METERS_PER_PIXEL = 0.01
DEFAULT_DRIVABLE_CLASS_IDS = [0, 5, 6]


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("CUDA GPU is not available. Falling back to CPU.")
    return torch.device("cpu")


def calculate_steering_angle(alpha_deg, target_distance_m, wheelbase=WHEELBASE):
    alpha_rad = np.radians(alpha_deg)
    ld = float(target_distance_m)
    steering_rad = (
        np.arctan2(2 * wheelbase * np.sin(alpha_rad), ld) if ld > 0 else 0.0
    )
    steering_deg = np.clip(
        np.degrees(steering_rad), -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE
    )
    return steering_deg, np.radians(steering_deg)


def get_fixed_roi(height, width, roi_width=ROI_W, roi_height=ROI_H):
    cx, cy = width // 2, height // 2
    roi_width = max(1, int(roi_width))
    roi_height = max(1, int(roi_height))
    return (
        max(0, cx - roi_width // 2),
        max(0, cy - roi_height // 2),
        min(width, cx + roi_width // 2),
        min(height, cy + roi_height // 2),
    )


def measure_continuous_drivable_width_x(mask, offset_x=0, offset_y=0):
    col_has_drivable = np.any(mask > 0, axis=0)
    if not np.any(col_has_drivable):
        return 0, (offset_x, offset_y, offset_x + 10, offset_y + mask.shape[0])

    start_idx = int(np.argmax(col_has_drivable))
    end_idx = start_idx
    while end_idx + 1 < len(col_has_drivable) and col_has_drivable[end_idx + 1]:
        end_idx += 1

    run_mask = mask[:, start_idx : end_idx + 1]
    ys, _ = np.where(run_mask > 0)
    y1 = int(ys.min()) + offset_y if len(ys) > 0 else offset_y
    y2 = int(ys.max()) + offset_y if len(ys) > 0 else offset_y + mask.shape[0]
    x1 = start_idx + offset_x
    x2 = end_idx + offset_x
    drivable_width_px = x2 - x1 + 1
    return drivable_width_px, (x1, y1, x2, y2)


def measure_continuous_drivable_height_y(mask, offset_x=0, offset_y=0):
    row_has_drivable = np.any(mask > 0, axis=1)
    if not np.any(row_has_drivable):
        return 0, (offset_x, offset_y, offset_x + mask.shape[1], offset_y + 10)

    start_idx = int(np.argmax(row_has_drivable))
    end_idx = start_idx
    while end_idx + 1 < len(row_has_drivable) and row_has_drivable[end_idx + 1]:
        end_idx += 1

    run_mask = mask[start_idx : end_idx + 1, :]
    _, xs = np.where(run_mask > 0)
    x1 = int(xs.min()) + offset_x if len(xs) > 0 else offset_x
    x2 = int(xs.max()) + offset_x if len(xs) > 0 else offset_x + mask.shape[1]
    y1 = start_idx + offset_y
    y2 = end_idx + offset_y
    drivable_height_px = y2 - y1 + 1
    return drivable_height_px, (x1, y1, x2, y2)


def check_drivable_width_ahead(final_mask, roi_rect):
    height, width = final_mask.shape
    zone_y1, zone_y2 = 0, height // 3
    vehicle_x1, _, vehicle_x2, _ = roi_rect
    vehicle_width = max(1, vehicle_x2 - vehicle_x1)
    zone_x1 = max(0, min(vehicle_x1, width - 1))
    zone_mask = final_mask[zone_y1:zone_y2, zone_x1:width]
    drivable_width_px, roi = measure_continuous_drivable_width_x(
        zone_mask, zone_x1, zone_y1
    )
    linear_x = 1.0 if drivable_width_px >= vehicle_width else 0.0
    return linear_x, drivable_width_px, roi


def check_drivable_height_ahead(final_mask, roi_rect, min_height_px=300):
    height, width = final_mask.shape
    zone_y1, zone_y2 = 0, height // 3
    vehicle_x1, _, _, _ = roi_rect
    zone_x1 = max(0, min(vehicle_x1, width - 1))
    zone_mask = final_mask[zone_y1:zone_y2, zone_x1:width]
    drivable_height_px, roi = measure_continuous_drivable_height_y(
        zone_mask, zone_x1, zone_y1
    )
    linear_x = 1.0 if drivable_height_px >= int(min_height_px) else 0.0
    return linear_x, drivable_height_px, roi


def clean_drivable_mask(mask, valid_ids=None):
    if valid_ids is None:
        valid_ids = DEFAULT_DRIVABLE_CLASS_IDS
    combined = np.isin(mask, valid_ids).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out = np.zeros_like(combined)
    if contours:
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    else:
        out = combined
    return (out > 0).astype(np.uint8)


def detect_vehicle_and_heading(pred, height, width, vehicle_size=None):
    if vehicle_size is None:
        vehicle_size = (ROI_W, ROI_H)
    vehicle_width, vehicle_height = vehicle_size
    return (
        (width // 2, height // 2),
        (0, -250),
        np.zeros((height, width), dtype=np.uint8),
        get_fixed_roi(height, width, vehicle_width, vehicle_height),
    )


def draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px):
    cx1, cy1, cx2, cy2 = cmp_roi
    cmp_color = (0, 255, 0) if linear_x >= 1.0 else (0, 0, 255)
    fill_color = (0, 200, 0) if linear_x >= 1.0 else (0, 0, 200)
    mask_crop = final_mask[cy1 : cy2 + 1, cx1 : cx2 + 1]
    layer = np.zeros_like(overlay)
    layer[cy1 : cy2 + 1, cx1 : cx2 + 1][mask_crop > 0] = fill_color
    cv2.addWeighted(layer, 0.35, overlay, 1.0, 0, overlay)
    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), cmp_color, 2)


def draw_vehicle_roi(overlay, height, width, vehicle_size=None):
    vehicle_center, heading_vector, _, roi_rect = detect_vehicle_and_heading(
        None, height, width, vehicle_size
    )
    vx1, vy1, vx2, vy2 = roi_rect
    arrow_end = (
        vehicle_center[0] + heading_vector[0],
        vehicle_center[1] + heading_vector[1],
    )
    cv2.arrowedLine(
        overlay, vehicle_center, arrow_end, (255, 0, 255), 3, tipLength=0.3
    )
    cv2.circle(overlay, vehicle_center, 8, (0, 255, 0), -1)
    cv2.putText(
        overlay,
        "Vehicle",
        (vehicle_center[0] + 10, vehicle_center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    cv2.rectangle(overlay, (vx1, vy1), (vx2, vy2), (255, 255, 255), 2)
    return vehicle_center, heading_vector, roi_rect


def compute_roi_linear_x(final_mask, roi_rect):
    vx1, vy1, vx2, vy2 = roi_rect
    roi_region = final_mask[vy1:vy2, vx1:vx2]
    vehicle_roi_width = max(1, vx2 - vx1)
    drivable_width_px, _ = measure_continuous_drivable_width_x(
        roi_region, vx1, vy1
    )
    linear_x = 1.0 if drivable_width_px >= vehicle_roi_width else 0.0
    cmp_roi = (vx1, vy1, vx2, vy2)
    return linear_x, drivable_width_px, cmp_roi


def select_target_point(path_points, vehicle_center, target_lookahead_m):
    pts = np.asarray(path_points)
    if len(pts) == 0:
        return None, 0.0

    target_px = max(float(target_lookahead_m) / METERS_PER_PIXEL, 1.0)
    vehicle = np.asarray(vehicle_center)
    dists_px = np.linalg.norm(pts - vehicle, axis=1)
    idx = int(np.argmin(np.abs(dists_px - target_px)))
    target_point = tuple(pts[idx].astype(int))
    target_distance_m = float(dists_px[idx] * METERS_PER_PIXEL)
    return target_point, target_distance_m


def process_frame(
    frame,
    model,
    transform,
    device,
    decode_fn,
    conf_thresh=0.7,
    lookahead_distance=200,
    target_lookahead_m=2.5,
    ros2_node=None,
    drivable_class_ids=None,
    inference_size=None,
    vehicle_size=None,
    use_fp16=False,
    render_overlay=True,
):
    global prev_spline_points

    input_frame = frame
    if inference_size:
        infer_w, infer_h = inference_size
        if infer_w > 0 and infer_h > 0:
            input_frame = cv2.resize(
                frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA
            )

    frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(PILImage.fromarray(frame_rgb)).unsqueeze(0).to(device)
    if use_fp16 and device.type == "cuda":
        input_tensor = input_tensor.half()

    with torch.inference_mode():
        logits = model(input_tensor)
        pred_gpu = torch.argmax(logits, dim=1)[0]
        pred = pred_gpu.cpu().numpy().astype(np.int64)
    if pred.shape[:2] != frame.shape[:2]:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    if render_overlay:
        colorized = cv2.cvtColor(decode_fn(pred).astype("uint8"), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 0.35, colorized, 0.65, 0)
    else:
        overlay = None

    height, width = pred.shape
    final_mask = clean_drivable_mask(pred, drivable_class_ids)

    if render_overlay:
        vehicle_center, heading_vector, roi_rect = draw_vehicle_roi(
            overlay, height, width, vehicle_size
        )
    else:
        vehicle_center, heading_vector, _, roi_rect = detect_vehicle_and_heading(
            None, height, width, vehicle_size
        )
    linear_x, drivable_width_px, cmp_roi = compute_roi_linear_x(final_mask, roi_rect)

    roi_height = height // 3
    roi_mask = final_mask[:roi_height, :]
    nav_points = []
    for y in range(0, roi_height, max(1, roi_height // 5)):
        xs = np.where(roi_mask[y, :] > 0)[0]
        if len(xs) > 0:
            nav_points.append((min(xs[0] + lookahead_distance, xs[-1]), y))

    spline_points = []
    interesting_point = None
    steering_deg = 0.0
    steering_rad = 0.0
    angle_diff = 0.0
    target_distance_m = 0.0

    if len(nav_points) >= 3:
        pts = np.array(nav_points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        t = np.insert(np.cumsum(dist), 0, 0)
        t_new = np.linspace(t[0], t[-1], 200)
        spline_points = np.stack(
            (CubicSpline(t, pts[:, 0])(t_new), CubicSpline(t, pts[:, 1])(t_new)),
            axis=-1,
        ).astype(int)
        if prev_spline_points is not None and len(prev_spline_points) == len(
            spline_points
        ):
            spline_points = (
                alpha_smooth * prev_spline_points
                + (1 - alpha_smooth) * spline_points
            ).astype(int)
        prev_spline_points = spline_points

        if render_overlay:
            for i in range(len(spline_points) - 1):
                cv2.line(
                    overlay,
                    tuple(spline_points[i]),
                    tuple(spline_points[i + 1]),
                    (255, 255, 0),
                    2,
                )

        interesting_point, target_distance_m = select_target_point(
            spline_points, vehicle_center, target_lookahead_m
        )
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                "Target",
                (interesting_point[0] + 10, interesting_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx)
            - np.arctan2(-heading_vector[1], heading_vector[0])
        )
        if angle_diff > 180:
            angle_diff -= 360
        if angle_diff < -180:
            angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE
        )

        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect
        )

        if render_overlay:
            cv2.putText(
                overlay,
                f"Alpha: {angle_diff:.1f} deg",
                (vehicle_center[0] - 80, vehicle_center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f}° | "
                f"linear_x={linear_x:.1f} | drivable_width_px={drivable_width_px}"
            )

    elif len(nav_points) > 0:
        interesting_point, target_distance_m = select_target_point(
            nav_points, vehicle_center, target_lookahead_m
        )
        if render_overlay:
            cv2.circle(overlay, interesting_point, 8, (0, 0, 255), -1)
            cv2.line(overlay, interesting_point, vehicle_center, (255, 0, 255), 2)

        dx = interesting_point[0] - vehicle_center[0]
        dy = interesting_point[1] - vehicle_center[1]
        angle_diff = np.degrees(
            np.arctan2(-dy, dx)
            - np.arctan2(-heading_vector[1], heading_vector[0])
        )
        if angle_diff > 180:
            angle_diff -= 360
        if angle_diff < -180:
            angle_diff += 360

        steering_deg, steering_rad = calculate_steering_angle(
            angle_diff, target_distance_m, WHEELBASE
        )
        linear_x, drivable_width_px, cmp_roi = check_drivable_width_ahead(
            final_mask, roi_rect
        )

        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(steering_deg, linear_x=linear_x)
            ros2_node.publish_status(
                f"NAVIGATING: delta={steering_deg:.2f}° | linear_x={linear_x:.1f}"
            )

    else:
        linear_x = 0.0
        if ros2_node is not None and ROS2_AVAILABLE:
            ros2_node.publish_cmd_vel(0.0, linear_x=0.0)
            ros2_node.publish_status("NO_PATH_FOUND")

    if render_overlay:
        draw_compare_roi(overlay, final_mask, cmp_roi, linear_x, drivable_width_px)

        h_orig, w_orig = frame.shape[:2]
        info_lines = [
            f"PURE PURSUIT  [{MODE_LABEL}]",
            f"Steering: {steering_deg:+.4f} deg  ({steering_rad:+.4f} rad)",
            f"Target Ld: {target_distance_m:.2f} m / set {target_lookahead_m:.2f} m",
            f"linear.x: {'GO' if linear_x >= 1.0 else 'STOP'} ({linear_x:.1f})",
        ]
        box_h = 25 * len(info_lines) + 10
        box_w = 380
        bx, by = w_orig - box_w - 10, 10
        sub = overlay[by : by + box_h, bx : bx + box_w]
        white = np.ones(sub.shape, dtype=np.uint8) * 255
        overlay[by : by + box_h, bx : bx + box_w] = cv2.addWeighted(
            sub, 0.4, white, 0.6, 1.0
        )
        for i, line in enumerate(info_lines):
            color = (0, 0, 0)
            if i == 3:
                color = (0, 150, 0) if linear_x >= 1.0 else (0, 0, 180)
            cv2.putText(
                overlay,
                line,
                (bx + 8, by + 20 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    return (
        overlay if render_overlay else frame,
        pred,
        spline_points.tolist()
        if isinstance(spline_points, np.ndarray)
        else spline_points,
        vehicle_center,
        interesting_point,
    )

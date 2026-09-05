import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


Golf_img_Path = 'Dataset/golf_car.png'
img_car = cv2.imread(Golf_img_Path, cv2.IMREAD_UNCHANGED)

# =========================
# Layout / Canvas CONFIG
# =========================
shift_w = 400
shift_h = 400

Cal_size_w = 120
Cal_size_h = 120

Car_size_w = 120
Car_size_h = 200

inn_shift_w = 45
inn_shift_h = 45

# Car top-left in BEV canvas
top_left_x = shift_w + Cal_size_w + inn_shift_w
top_left_y = shift_h + Cal_size_h + inn_shift_h

# --- Car overlay placement -------------------------------------------------
# Car_size_w/h describe the VEHICLE. The region no camera can see is larger
# than that and is not centred on it, so drawing the car image only over
# Car_size leaves a black ring. CAR_OVERLAY_BOX places the overlay over the
# measured blind zone instead. It affects ONLY
# ImageAdjuster.overlay_image_perspective(); it does NOT touch xl/xr/yt/yb,
# which define the stitch section crops, so the homographies stay valid.
# Set to None to use the plain Car_size rectangle.
# Re-measure with stitch_probe.py after any recalibration.
CAR_OVERLAY_BOX = None   # (x0, y0, x1, y1) in canvas px

if CAR_OVERLAY_BOX is None:
    _cx0, _cy0 = top_left_x, top_left_y
    _cx1, _cy1 = top_left_x + Car_size_w, top_left_y + Car_size_h
else:
    _cx0, _cy0, _cx1, _cy1 = CAR_OVERLAY_BOX

Car_dst_points = np.float32([
    [_cx0, _cy0],
    [_cx1, _cy0],
    [_cx0, _cy1],
    [_cx1, _cy1],
])

# Total canvas size
total_w = (2 * (shift_w + Cal_size_w + inn_shift_w)) + Car_size_w
total_h = (2 * (shift_h + Cal_size_h + inn_shift_h)) + Car_size_h

# Car rectangle
xl = shift_w + Cal_size_w + inn_shift_w
xr = xl + Car_size_w
yt = shift_h + Cal_size_h + inn_shift_h
yb = yt + Car_size_h

# Outer boundary
outer_xl = shift_w
outer_xr = total_w - shift_w
outer_yt = shift_h
outer_yb = total_h - shift_h

# Inner boundary
inner_xl = xl - inn_shift_w
inner_xr = xr + inn_shift_w
inner_yt = yt - inn_shift_h
inner_yb = yb + inn_shift_h

# =========================
# Chessboard patterns
# rows/cols = number of SQUARES
# =========================
PATTERN_TB = dict(rows=5, cols=14, chessboard_width=280, chessboard_height=100)   # front+rear
PATTERN_LR = dict(rows=21, cols=5, chessboard_width=100, chessboard_height=420)   # left+right

# =========================
# Helper: build dst rect from top-left + size
# order: TL, TR, BL, BR
# =========================
def dst_rect_from_tl(x_tl, y_tl, w, h):
    return np.array([
        [x_tl,     y_tl],
        [x_tl + w, y_tl],
        [x_tl,     y_tl + h],
        [x_tl + w, y_tl + h],
    ], dtype=np.float32)

# =========================
# Anchors (top-left positions)
# =========================
shift_wa = 350

TL_FRONT = (shift_w + 80, shift_h - 45)
TL_LEFT  = (shift_w + 0,  shift_h + 70)
# TL_RIGHT = (shift_w + 345, shift_h + 70)
TL_RIGHT = (shift_w + 340, shift_h + 70)
TL_REAR  = (shift_w + 85, shift_h + 511)

DST_TB_W = int(PATTERN_TB["chessboard_width"])
DST_TB_H = int(PATTERN_TB["chessboard_height"])
DST_LR_W = int(PATTERN_LR["chessboard_width"])
DST_LR_H = int(PATTERN_LR["chessboard_height"])

# =========================
# Chessboard destination config
# =========================
chessboard_config = {
    "front": {
        "inner_dst_pts": dst_rect_from_tl(*TL_FRONT, DST_TB_W, DST_TB_H),
        **PATTERN_TB
    },
    "left": {
        "inner_dst_pts": dst_rect_from_tl(*TL_LEFT, DST_LR_W, DST_LR_H),
        **PATTERN_LR
    },
    "right": {
        "inner_dst_pts": dst_rect_from_tl(*TL_RIGHT, DST_LR_W, DST_LR_H),
        **PATTERN_LR
    },
    "rear": {
        "inner_dst_pts": dst_rect_from_tl(*TL_REAR, DST_TB_W, DST_TB_H),
        **PATTERN_TB
    }
}

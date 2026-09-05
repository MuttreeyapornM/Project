"""One-shot: grab 4 cameras, warp, stitch, save everything for inspection."""
import sys, cv2, numpy as np
sys.path.insert(0, ".")
from image_processing import ImageStitcher, ImageAdjuster
from param_settings import (img_car, Car_dst_points, total_w, total_h,
                            xl, xr, yt, yb, chessboard_config, shift_w, shift_h)
import importlib.util as iu
spec = iu.spec_from_file_location("rt", "RealTime.py")

IDS = {"front": 0, "left": 4, "rear": 2, "right": 6}
CAMS = ["front", "left", "rear", "right"]

def calib(c):
    fs = cv2.FileStorage(f"yaml/calibration_data_{c}.yaml", cv2.FILE_STORAGE_READ)
    d = {k: fs.getNode(k).mat() for k in ("camera_matrix", "dist_coeffs", "homography")}
    fs.release(); return d

print(f"canvas {total_w}x{total_h}  car x{xl}-{xr} y{yt}-{yb}")
for c in CAMS:
    p = chessboard_config[c if c != "rear" else "rear"]
    d = p["inner_dst_pts"]
    print(f"  {c:6s} board dst x {d[:,0].min():.0f}-{d[:,0].max():.0f} "
          f"y {d[:,1].min():.0f}-{d[:,1].max():.0f}  "
          f"({p['chessboard_width']}x{p['chessboard_height']} cm)")

warped = []
for c in CAMS:
    cap = cv2.VideoCapture(IDS[c], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(8): cap.read()
    ok, f = cap.read(); cap.release()
    if not ok: sys.exit(f"read failed {c}")
    cv2.imwrite(f"probe_raw_{c}.png", f)
    d = calib(c)
    u = cv2.undistort(f, d["camera_matrix"], d["dist_coeffs"])
    if c in ("rear", "right"): u = cv2.rotate(u, cv2.ROTATE_180)
    w = cv2.warpPerspective(u, d["homography"], (total_w, total_h))
    cv2.imwrite(f"probe_warp_{c}.png", w)
    cov = (w.sum(axis=2) > 0)
    ys, xs = np.where(cov)
    print(f"  {c:6s} warped covers x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}  "
          f"{100*cov.mean():.1f}% of canvas")
    warped.append(w)

merged = ImageStitcher.get_weights_and_masks_liverun(warped)
cv2.imwrite("probe_merged.png", merged)
if img_car is not None:
    over = ImageAdjuster.overlay_image_perspective(merged.copy(), img_car, Car_dst_points)
    cv2.imwrite("probe_merged_car.png", over)

# quad view of the four warped layers, colour-coded
vis = np.zeros((total_h, total_w, 3), np.uint8)
cols = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
for w, col in zip(warped, cols):
    m = (w.sum(axis=2) > 0)
    vis[m] = (vis[m] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
cv2.rectangle(vis, (xl, yt), (xr, yb), (255,255,255), 3)
cv2.imwrite("probe_coverage.png", vis)
print("wrote probe_raw_*.png probe_warp_*.png probe_merged.png probe_merged_car.png probe_coverage.png")

import cv2, numpy as np, sys
sys.path.insert(0, ".")
from image_processing import ImageStitcher
from param_settings import total_w, total_h
CAMS = ["front", "left", "rear", "right"]

def cal(c):
    fs = cv2.FileStorage(f"yaml/calibration_data_{c}.yaml", cv2.FILE_STORAGE_READ)
    d = {k: fs.getNode(k).mat() for k in ("camera_matrix", "dist_coeffs", "homography")}
    fs.release(); return d

dev = {}
for i in (0, 2, 4, 6):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(8): cap.read()
    ok, f = cap.read(); cap.release()
    dev[i] = f if ok else None
    print("  video%d: %s" % (i, "ok" if ok else "FAILED"))

def build(mapping, name):
    ws = []
    for c in CAMS:
        f = dev[mapping[c]]; d = cal(c)
        u = cv2.undistort(f, d["camera_matrix"], d["dist_coeffs"])
        if c in ("rear", "right"): u = cv2.rotate(u, cv2.ROTATE_180)
        ws.append(cv2.warpPerspective(u, d["homography"], (total_w, total_h)))
    ImageStitcher.reset_weights()
    m = ImageStitcher.get_weights_and_masks_liverun(ws)
    cv2.imwrite("probe_map_%s.png" % name, m)
    cov = np.zeros((total_h, total_w), bool)
    for w in ws: cov |= (w.sum(axis=2) > 0)
    print("  %-16s coverage %.1f%%" % (name, 100 * cov.mean()))

build({"front": 0, "left": 4, "rear": 2, "right": 6}, "A_left4_rear2")
build({"front": 0, "left": 2, "rear": 4, "right": 6}, "B_left2_rear4")

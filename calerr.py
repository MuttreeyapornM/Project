import cv2, numpy as np, sys
sys.path.insert(0, ".")
from param_settings import chessboard_config as CC
CAMS = ["front", "left", "rear", "right"]

def cal(c):
    fs = cv2.FileStorage(f"yaml/calibration_data_{c}.yaml", cv2.FILE_STORAGE_READ)
    d = {k: fs.getNode(k).mat() for k in ("camera_matrix", "dist_coeffs", "homography")}
    fs.release(); return d

def variants(g):
    yield g, "raw"
    yield cv2.createCLAHE(2.0, (8, 8)).apply(g), "clahe"
    yield cv2.GaussianBlur(g, (5, 5), 0), "blur"
    yield 255 - g, "invert"

for c in CAMS:
    img = cv2.imread(f"probe_raw_{c}.png")
    d = cal(c)
    u = cv2.undistort(img, d["camera_matrix"], d["dist_coeffs"])
    if c in ("rear", "right"):
        u = cv2.rotate(u, cv2.ROTATE_180)
    g = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
    cfg = CC[c]
    # squares -> inner corners, both orientations
    cands = [(cfg["cols"] - 1, cfg["rows"] - 1), (cfg["rows"] - 1, cfg["cols"] - 1),
             (13, 4), (4, 13), (4, 20), (20, 4), (6, 4), (4, 6)]
    hit = None
    for gv, vn in variants(g):
        for cols, rows in cands:
            ok, cor = cv2.findChessboardCornersSB(gv, (cols, rows),
                        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                hit = (cols, rows, cor, vn); break
        if hit: break
    if not hit:
        print(f"{c:6s} board NOT detected in the undistorted source"); continue
    cols, rows, cor, vn = hit
    cor = cv2.cornerSubPix(g, cor, (11, 11), (-1, -1),
                           (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
    pts = cor.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, d["homography"]).reshape(-1, 2)
    dst = cfg["inner_dst_pts"]
    x0, x1 = dst[:, 0].min(), dst[:, 0].max()
    y0, y1 = dst[:, 1].min(), dst[:, 1].max()
    # inner corners sit inset by half a square from the board edge
    sw, sh = (x1 - x0) / cfg["cols"], (y1 - y0) / cfg["rows"]
    ex0, ex1, ey0, ey1 = x0 + sw, x1 - sw, y0 + sh, y1 - sh
    mx0, mx1 = mapped[:, 0].min(), mapped[:, 0].max()
    my0, my1 = mapped[:, 1].min(), mapped[:, 1].max()
    print(f"{c:6s} detected {cols}x{rows} inner ({vn})")
    print(f"       lands at   x {mx0:7.1f}-{mx1:7.1f} ({mx1-mx0:6.1f})  y {my0:7.1f}-{my1:7.1f} ({my1-my0:6.1f})")
    print(f"       config says x {ex0:7.1f}-{ex1:7.1f} ({ex1-ex0:6.1f})  y {ey0:7.1f}-{ey1:7.1f} ({ey1-ey0:6.1f})")
    print(f"       ERROR       dx {mx0-ex0:+7.1f} cm  dy {my0-ey0:+7.1f} cm   size ratio x{(mx1-mx0)/(ex1-ex0):.2f} y{(my1-my0)/(ey1-ey0):.2f}")

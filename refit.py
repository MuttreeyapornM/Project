"""Re-fit each camera's homography so its board maps at exactly 1 px = 1 cm.

Keeps each board where it currently lands (position is already close) and fixes
the SCALE, which is where the error is. Ordering ambiguity is resolved the way
Step4 does it: try all four rotations of the ideal grid, keep the lowest
reprojection RMSE. Writes yaml/ only with --apply.
"""
import cv2, numpy as np, sys, shutil, os
sys.path.insert(0, ".")

SQUARE_CM = 20.0
CAMS = ["front", "left", "rear", "right"]
APPLY = "--apply" in sys.argv

def cal(c):
    fs = cv2.FileStorage(f"yaml/calibration_data_{c}.yaml", cv2.FILE_STORAGE_READ)
    d = {k: fs.getNode(k).mat() for k in ("camera_matrix", "dist_coeffs", "homography", "resolution")}
    fs.release(); return d

def variants(g):
    yield g, "raw"
    yield cv2.createCLAHE(2.0, (8, 8)).apply(g), "clahe"
    yield cv2.GaussianBlur(g, (5, 5), 0), "blur"
    yield 255 - g, "invert"

GRIDS = [(13,4),(4,13),(12,4),(4,12),(6,4),(4,6),(7,4),(4,7),(10,4),(4,10),(6,3),(3,6)]

def rmse(H, src, dst):
    p = cv2.perspectiveTransform(src.reshape(-1,1,2), H).reshape(-1,2)
    return float(np.sqrt(np.mean(np.sum((p-dst)**2, axis=1))))

report = []
for c in CAMS:
    img = cv2.imread(f"probe_raw_{c}.png")
    d = cal(c)
    u = cv2.undistort(img, d["camera_matrix"], d["dist_coeffs"])
    if c in ("rear", "right"):
        u = cv2.rotate(u, cv2.ROTATE_180)
    g = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
    # Search EVERY variant x grid and keep the LARGEST grid found. Taking the
    # first hit lets a spurious small sub-grid in the raw image beat the real
    # full grid that only shows up after CLAHE.
    hit = None
    for gv, vn in variants(g):
        for cols, rows in GRIDS:
            if hit is not None and cols * rows <= hit[0] * hit[1]:
                continue
            ok, cor = cv2.findChessboardCornersSB(gv, (cols, rows),
                        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                hit = (cols, rows, cor, vn)
    if not hit:
        report.append((c, None)); print(f"{c:6s} NOT DETECTED - skipped"); continue
    cols, rows, cor, vn = hit
    cor = cv2.cornerSubPix(g, cor, (11,11), (-1,-1),
                           (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER, 30, 0.001))
    src = cor.reshape(-1,2).astype(np.float32)

    # where it lands today, with the existing H
    old = cv2.perspectiveTransform(src.reshape(-1,1,2), d["homography"]).reshape(-1,2)
    cx, cy = old[:,0].mean(), old[:,1].mean()
    old_rmse_scale = ((old[:,0].max()-old[:,0].min())/((cols-1)*SQUARE_CM),
                      (old[:,1].max()-old[:,1].min())/((rows-1)*SQUARE_CM))

    # ideal grid: exact 20 cm spacing, 1 px = 1 cm, centred on the current centroid
    gx, gy = np.meshgrid(np.arange(cols)*SQUARE_CM, np.arange(rows)*SQUARE_CM)
    ideal = np.stack([gx, gy], -1).astype(np.float32)
    ideal[...,0] -= ideal[...,0].mean(); ideal[...,1] -= ideal[...,1].mean()
    ideal[...,0] += cx;                  ideal[...,1] += cy

    # Pick the grid ORIENTATION by agreement with where the board currently
    # lands, NOT by fit residual. A 180-degree flip fits a near-symmetric
    # chessboard just as well but reverses the whole camera-to-canvas mapping,
    # and RMSE measured on the board alone cannot see that. The existing
    # homographies have the correct global orientation; only their scale is
    # wrong, so the current mapping is the right thing to agree with.
    pick = None
    for k in range(4):
        cand = np.rot90(ideal, k).reshape(-1,2).astype(np.float32)
        if cand.shape[0] != src.shape[0]: continue
        agree = float(np.mean(np.linalg.norm(cand - old, axis=1)))
        if pick is None or agree < pick[0]: pick = (agree, k, cand)
    if pick is None:
        print(f"{c:6s} fit failed - no usable orientation"); continue
    agree, k, cand = pick
    H, _ = cv2.findHomography(src, cand, cv2.RANSAC, 3.0)
    if H is None:
        print(f"{c:6s} fit failed - findHomography returned None"); continue
    e = rmse(H, src, cand)
    print(f"       orientation rot{k*90} chosen, mean shift from current mapping {agree:.1f} cm")
    new = cv2.perspectiveTransform(src.reshape(-1,1,2), H).reshape(-1,2)
    new_scale = ((new[:,0].max()-new[:,0].min())/((cols-1)*SQUARE_CM),
                 (new[:,1].max()-new[:,1].min())/((rows-1)*SQUARE_CM))
    print(f"{c:6s} grid {cols}x{rows} ({vn})  rot{k*90}  RMSE {e:6.2f} cm")
    print(f"       scale px/cm  before x{old_rmse_scale[0]:.3f} y{old_rmse_scale[1]:.3f}"
          f"   after x{new_scale[0]:.3f} y{new_scale[1]:.3f}")
    report.append((c, (e, H)))

if APPLY:
    os.makedirs("pre_refit_2026-09-05", exist_ok=True)
    for c, r in report:
        if r is None: continue
        shutil.copy2(f"yaml/calibration_data_{c}.yaml", f"pre_refit_2026-09-05/calibration_data_{c}.yaml")
        d = cal(c)
        fs = cv2.FileStorage(f"yaml/calibration_data_{c}.yaml", cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", d["camera_matrix"]); fs.write("dist_coeffs", d["dist_coeffs"])
        fs.write("resolution", d["resolution"]);       fs.write("homography", r[1])
        fs.release()
        print(f"  wrote yaml/calibration_data_{c}.yaml (backup in pre_refit_2026-09-05/)")
else:
    print("\ndry run - pass --apply to write the YAMLs")

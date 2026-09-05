"""Live BEV on the Xavier: capture the four cameras, run the patched pipeline, time it."""
import sys, time, statistics, numpy as np, cv2
sys.path.insert(0, ".")
from bev_processor import BEVProcessor          # patched (LUT)
import bev_processor_orig as ORIG               # pre-patch backup
from image_processing import ImageStitcher

# Command.md used --front_cam 2 --left_cam 4 --rear_cam 0 --right_cam 6;
# index 6 no longer enumerates, /dev/video7 is the Logitech C615.
IDX = {"front": 2, "left": 4, "rear": 0, "right": 7}
CAMS = ["front", "left", "rear", "right"]
N = 20

caps = {}
for c in CAMS:
    cap = cv2.VideoCapture(IDX[c], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"  {c:6s} /dev/video{IDX[c]}  FAILED TO OPEN"); cap = None
    else:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        print(f"  {c:6s} /dev/video{IDX[c]}  "
              f"{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps  "
              f"fourcc {''.join(chr((fourcc>>8*i)&0xFF) for i in range(4))}")
    caps[c] = cap
if any(v is None for v in caps.values()):
    sys.exit("could not open all four cameras")

for _ in range(10):                      # let auto-exposure settle
    for c in CAMS: caps[c].read()

frames = {}
for c in CAMS:
    ok, f = caps[c].read()
    if not ok: sys.exit(f"read failed on {c}")
    if f.shape[:2] != (720, 1280): f = cv2.resize(f, (1280, 720))
    frames[c] = f
    cv2.imwrite(f"live_raw_{c}.png", f)
print(f"\ncaptured 4 live frames at {frames['front'].shape[1]}x{frames['front'].shape[0]}")

paths = {c: "" for c in CAMS}
pat = BEVProcessor(paths, img_car=None)
org = ORIG.BEVProcessor(paths, img_car=None)

def bev(proc):
    return ImageStitcher.get_weights_and_masks_liverun(
        [proc.process_image(frames[c], c)[1] for c in CAMS])

bp, bo = bev(pat), bev(org)
cv2.imwrite("live_bev_patched.png", bp)
cv2.imwrite("live_bev_unpatched.png", bo)
d = np.abs(bo.astype(np.int16) - bp.astype(np.int16)).max(axis=2)
print(f"live BEV {bp.shape}  patched vs unpatched: max {d.max()} mean {d.mean():.3f} "
      f">10 on {100.0*(d>10).sum()/d.size:.3f}%")

# ---- timing on live frames --------------------------------------------
def bench(proc, warm=3):
    for _ in range(warm): bev(proc)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); bev(proc); ts.append((time.perf_counter()-t0)*1000)
    ts.sort(); return statistics.median(ts), ts[0], ts[int(.95*(len(ts)-1))]

for name, proc in (("unpatched", org), ("patched", pat)):
    m, lo, hi = bench(proc)
    print(f"{name:10s} 4-camera process + stitch: median {m:7.2f} ms  "
          f"min {lo:7.2f}  p95 {hi:7.2f}   -> {1000/m:5.2f} FPS")

# ---- end-to-end including capture -------------------------------------
ts = []
for _ in range(N):
    t0 = time.perf_counter()
    fr = {}
    for c in CAMS:
        ok, f = caps[c].read()
        fr[c] = f if (ok and f.shape[:2] == (720, 1280)) else frames[c]
    ImageStitcher.get_weights_and_masks_liverun(
        [pat.process_image(fr[c], c)[1] for c in CAMS])
    ts.append((time.perf_counter()-t0)*1000)
ts.sort()
print(f"patched, INCLUDING serial capture of 4 cameras: median {statistics.median(ts):7.2f} ms"
      f"  -> {1000/statistics.median(ts):5.2f} FPS")
for c in CAMS: caps[c].release()

panel = np.hstack([bo, bp])
panel = cv2.resize(panel, (int(panel.shape[1]*760/panel.shape[0]), 760))
cv2.imwrite("live_bev_panel.png", panel)
grid = np.vstack([np.hstack([cv2.resize(frames["front"],(640,360)), cv2.resize(frames["left"],(640,360))]),
                  np.hstack([cv2.resize(frames["rear"],(640,360)),  cv2.resize(frames["right"],(640,360))])])
cv2.imwrite("live_raw_grid.png", grid)
print("wrote live_bev_panel.png, live_raw_grid.png")

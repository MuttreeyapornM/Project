"""Stitcher patch: correctness + timing, on the Xavier, on live camera frames."""
import sys, time, statistics, numpy as np, cv2
sys.path.insert(0, ".")
from bev_processor import BEVProcessor
import image_processing as ORIG
import image_processing_fixed as FIX

CAMS = ["front", "left", "rear", "right"]
fr = {c: cv2.imread(f"live_raw_{c}.png") for c in CAMS}
assert all(v is not None for v in fr.values()), "live_raw_*.png missing"
p = BEVProcessor({c: "" for c in CAMS}, img_car=None)
W = [p.process_image(fr[c], c)[1] for c in CAMS]

o = ORIG.ImageStitcher.get_weights_and_masks_liverun(W)
FIX.ImageStitcher.reset_weights()
f1 = FIX.ImageStitcher.get_weights_and_masks_liverun(W)   # cold: builds cache
f2 = FIX.ImageStitcher.get_weights_and_masks_liverun(W)   # warm: uses cache
print("cached corners:", FIX.ImageStitcher.cached_corners())
print("cold vs warm patched output identical:", np.array_equal(f1, f2))

d = np.abs(o.astype(np.int16) - f2.astype(np.int16)).max(axis=2)
print(f"patched vs original stitch: max {d.max()}  mean {d.mean():.3f}  "
      f">10 on {100.0*(d>10).sum()/d.size:.3f}%  >30 on {100.0*(d>30).sum()/d.size:.3f}%")
cv2.imwrite("stitch_orig.png", o); cv2.imwrite("stitch_fixed.png", f2)
cv2.imwrite("stitch_diff_x8.png",
            cv2.applyColorMap(np.clip(d.astype(np.int16)*8,0,255).astype(np.uint8),
                              cv2.COLORMAP_INFERNO))

def med(fn, n=20, w=5):
    for _ in range(w): fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1000)
    ts.sort(); return statistics.median(ts)

print()
for r in range(3):
    a = med(lambda: ORIG.ImageStitcher.get_weights_and_masks_liverun(W))
    b = med(lambda: FIX.ImageStitcher.get_weights_and_masks_liverun(W))
    warp = med(lambda: [p.process_image(fr[c], c)[1] for c in CAMS])
    full_o = med(lambda: ORIG.ImageStitcher.get_weights_and_masks_liverun(
        [p.process_image(fr[c], c)[1] for c in CAMS]))
    full_f = med(lambda: FIX.ImageStitcher.get_weights_and_masks_liverun(
        [p.process_image(fr[c], c)[1] for c in CAMS]))
    print(f"round {r+1}: stitch {a:7.2f} -> {b:6.2f} ms ({a/b:4.2f}x) | warp4 {warp:6.2f} | "
          f"full frame {full_o:7.2f} -> {full_f:6.2f} ms ({full_o/full_f:4.2f}x) = "
          f"{1000/full_o:5.2f} -> {1000/full_f:5.2f} FPS")

# cold-start cost of building the cache
FIX.ImageStitcher.reset_weights()
t0 = time.perf_counter(); FIX.ImageStitcher.get_weights_and_masks_liverun(W)
print(f"\ncache build (first frame only): {(time.perf_counter()-t0)*1000:.2f} ms")

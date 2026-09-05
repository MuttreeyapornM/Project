"""Compare blend implementations for ImageStitcher.merge on real crop sizes."""
import sys, time, statistics, numpy as np, cv2
sys.path.insert(0, ".")
from image_processing import ImageStitcher
from param_settings import xl, yt

# real corner-crop size used by get_weights_and_masks_liverun
H, W = yt, xl
rng = np.random.default_rng(0)
imA = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
imB = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
xs = np.linspace(0, 1, W, dtype=np.float32)
G = np.repeat(xs[None, :], H, axis=0)          # smooth 0..1 ramp, like a real weight map
G3 = np.repeat(G[:, :, None], 3, axis=2).astype(np.float32)
GB3 = 1.0 - G3

# fixed point, 7-bit: 255*128 + 255*128 = 65280 < 65535, so uint16 cannot overflow
Q = 128
Gq = np.repeat(np.round(G * Q).astype(np.uint16)[:, :, None], 3, axis=2)
GqB = (Q - Gq).astype(np.uint16)

def ref():
    return cv2.add(cv2.multiply(imA.astype(np.float32), G3),
                   cv2.multiply(imB.astype(np.float32), GB3)).astype(np.uint8)

def np_fixed():
    return ((imA.astype(np.uint16) * Gq + imB.astype(np.uint16) * GqB) >> 7).astype(np.uint8)

def cv_fixed():
    s = cv2.add(cv2.multiply(imA, Gq, dtype=cv2.CV_16U),
                cv2.multiply(imB, GqB, dtype=cv2.CV_16U))
    return cv2.convertScaleAbs(s, alpha=1.0 / Q)

def med(fn, n=30, w=6):
    for _ in range(w): fn()
    ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append((time.perf_counter() - a) * 1000)
    ts.sort(); return statistics.median(ts)

r = ref()
print(f"  crop {W}x{H}x3")
for name, fn in (("float32 (current)", ref), ("numpy uint16 fixed", np_fixed), ("cv2 uint16 fixed", cv_fixed)):
    t = med(fn)
    d = np.abs(fn().astype(np.int16) - r.astype(np.int16))
    print(f"  {name:22s} {t:6.2f} ms   max diff {d.max():3d}  mean {d.mean():.3f}  >1 on {100*(d>1).mean():.2f}%")
print(f"\n  x4 corners: float32 {4*med(ref):6.2f} ms   cv2 fixed {4*med(cv_fixed):6.2f} ms")

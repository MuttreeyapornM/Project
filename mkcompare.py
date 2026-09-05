"""Render the stitched BEV from the unpatched and patched processors, side by side."""
import sys, time, numpy as np, cv2
sys.path.insert(0, ".")
import bev_processor_orig as ORIG      # the pre-patch backup
import bev_processor as PATCHED        # now the LUT version
from image_processing import ImageStitcher

CAMS = ["front", "left", "rear", "right"]
SRC = {c: f"Dataset/Img_distortion_Testing/{c.capitalize()}.jpg" for c in CAMS}
imgs = {}
for c in CAMS:
    im = cv2.imread(SRC[c])
    if im.shape[:2] != (720, 1280):
        im = cv2.resize(im, (1280, 720))
    imgs[c] = im

def build(mod, label):
    b = mod.BEVProcessor({c: SRC[c] for c in CAMS}, img_car=None)
    warped, t = [], 0.0
    for c in CAMS:                      # bev_processor orders caps front,left,rear,right
        t0 = time.perf_counter()
        _, w = b.process_image(imgs[c], c)
        t += (time.perf_counter() - t0) * 1000
        warped.append(w)
    merged = ImageStitcher.get_weights_and_masks_liverun(warped)
    print(f"{label:10s} process_image total {t:7.2f} ms   merged {merged.shape}")
    return merged

o = build(ORIG, "unpatched")
p = build(PATCHED, "patched")

cv2.imwrite("cmp_bev_unpatched.png", o)
cv2.imwrite("cmp_bev_patched.png", p)

d = np.abs(o.astype(np.int16) - p.astype(np.int16)).max(axis=2).astype(np.uint8)
print(f"difference: max {d.max()}  mean {d.mean():.3f}  "
      f">10 on {100.0*(d>10).sum()/d.size:.3f}% of the canvas")
cv2.imwrite("cmp_bev_diff_raw.png", d)
heat = cv2.applyColorMap(np.clip(d.astype(np.int16) * 8, 0, 255).astype(np.uint8),
                         cv2.COLORMAP_INFERNO)
cv2.imwrite("cmp_bev_diff_x8.png", heat)

def label(img, txt, sub=""):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 74), (0, 0, 0), -1)
    cv2.putText(img, txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    if sub:
        cv2.putText(img, sub, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 220, 255), 1)
    return img

panel = np.hstack([
    label(o, "UNPATCHED", "kornia GPU warp, maps rebuilt per frame - 251.71 ms"),
    label(p, "PATCHED",   "single remap LUT, maps built once - 48.09 ms"),
    label(heat, "DIFFERENCE x8", "inferno; black = identical"),
])
h = 900
panel = cv2.resize(panel, (int(panel.shape[1] * h / panel.shape[0]), h))
cv2.imwrite("cmp_bev_panel.png", panel)
print("wrote cmp_bev_panel.png", panel.shape)

# zoom on the busiest edge region to show what the difference actually is
ys, xs = np.where(d > 30)
if len(ys):
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = max(0, cy - 110), max(0, cx - 110)
    y1, x1 = min(o.shape[0], y0 + 220), min(o.shape[1], x0 + 220)
    z = lambda im: cv2.resize(im[y0:y1, x0:x1], (440, 440), interpolation=cv2.INTER_NEAREST)
    zoom = np.hstack([label(z(o), "UNPATCHED"), label(z(p), "PATCHED"),
                      label(z(heat), "DIFF x8")])
    cv2.imwrite("cmp_bev_zoom.png", zoom)
    print(f"wrote cmp_bev_zoom.png  (220x220 crop at y={y0} x={x0})")

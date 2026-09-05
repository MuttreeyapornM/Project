"""Per-stage timing for the saty5 pipeline, without cameras."""
import sys, time, statistics, argparse, numpy as np, cv2, torch
sys.path.insert(0, ".")
import numpy, numpy.core
for n in list(sys.modules):
    if n.startswith("numpy.core"):
        sys.modules[n.replace("numpy.core", "numpy._core", 1)] = sys.modules[n]
sys.modules["numpy._core"] = numpy.core

import importlib.util as iu
spec = iu.spec_from_file_location("s5", "saty5.py"); m = iu.module_from_spec(spec)
sys.modules["s5"] = m; spec.loader.exec_module(m)

from bev_processor import BEVProcessor
from image_processing import ImageStitcher

CAMS = ["front", "left", "rear", "right"]
fr = {c: cv2.imread(f"live_raw_{c}.png") for c in CAMS}
p = BEVProcessor({c: "" for c in CAMS}, img_car=None)

def bev():
    return ImageStitcher.get_weights_and_masks_liverun(
        [p.process_image(fr[c], c, want_undistorted=False)[1] for c in CAMS])

parser = m.get_argparser()
opts = parser.parse_args([
    "--dataset", "custom", "--ckpt", "./checkpoints_zoo/best27000.pth",
    "--fp16", "--inference_width", "640", "--inference_height", "360",
    "--vehicle_width", "200", "--vehicle_height", "300",
    "--lookahead_distance", "150", "--target_lookahead_m", "4.0",
    "--spline_curvature_threshold", "0.3",
    "--straight_linear_x", "1.5", "--curved_linear_x", "1.0",
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform, decode_fn = m.build_model(opts, device)
m.warmup_model(model, device, opts)
inference_size = (opts.inference_width, opts.inference_height)
vehicle_size = (opts.vehicle_width, opts.vehicle_height)

B = bev()
def frame(render):
    return m.process_frame(
        B, model, transform, device, decode_fn,
        conf_thresh=opts.conf_thresh, lookahead_distance=opts.lookahead_distance,
        target_lookahead_m=opts.target_lookahead_m, ros2_node=None,
        drivable_class_ids=opts.drivable_class_ids, inference_size=inference_size,
        vehicle_size=vehicle_size, use_fp16=opts.fp16, render_overlay=render,
        spline_curvature_threshold=opts.spline_curvature_threshold,
        straight_linear_x=opts.straight_linear_x, curved_linear_x=opts.curved_linear_x,
        min_nav_path_points=opts.min_nav_path_points,
        straight_path_bias=opts.straight_path_bias,
        nav_corridor_width=opts.nav_corridor_width,
        nav_path_offset_px=opts.nav_path_offset_px,
        nav_path_lateral_ratio=opts.nav_path_lateral_ratio)

def med(fn, n=15, w=5):
    for _ in range(w): fn()
    torch.cuda.synchronize() if device.type == "cuda" else None
    ts = []
    for _ in range(n):
        a = time.perf_counter(); fn()
        torch.cuda.synchronize() if device.type == "cuda" else None
        ts.append((time.perf_counter() - a) * 1000)
    ts.sort(); return statistics.median(ts)

b = med(bev); pf = med(lambda: frame(True)); pn = med(lambda: frame(False))
print()
print(f"  BEV (4 warps + stitch, patched) : {b:7.2f} ms")
print(f"  process_frame, overlay ON       : {pf:7.2f} ms   <- --show_preview")
print(f"  process_frame, overlay OFF      : {pn:7.2f} ms")
print(f"  {'-'*52}")
print(f"  TOTAL with preview              : {b+pf:7.2f} ms  = {1000/(b+pf):5.2f} FPS")
print(f"  TOTAL headless                  : {b+pn:7.2f} ms  = {1000/(b+pn):5.2f} FPS")
print()
print(f"  NOTE: async_capture overlaps BEV with inference, so the real loop rate")
print(f"        is bounded by max(BEV, process_frame), not their sum:")
print(f"        with preview  {max(b,pf):7.2f} ms = {1000/max(b,pf):5.2f} FPS")
print(f"        headless      {max(b,pn):7.2f} ms = {1000/max(b,pn):5.2f} FPS")

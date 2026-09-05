"""Split process_frame: model forward vs everything else."""
import sys, time, statistics, numpy as np, cv2, torch
sys.path.insert(0, ".")
import importlib.util as iu
spec = iu.spec_from_file_location("s5", "saty5.py"); m = iu.module_from_spec(spec)
sys.modules["s5"] = m; spec.loader.exec_module(m)
from bev_processor import BEVProcessor
from image_processing import ImageStitcher

CAMS = ["front", "left", "rear", "right"]
fr = {c: cv2.imread(f"probe_raw_{c}.png") for c in CAMS}
p = BEVProcessor({c: "" for c in CAMS}, img_car=None)
BEV = ImageStitcher.get_weights_and_masks_liverun(
    [p.process_image(fr[c], c, want_undistorted=False)[1] for c in CAMS])

opts = m.get_argparser().parse_args([
    "--dataset","custom","--ckpt","./checkpoints_zoo/best27000.pth","--fp16",
    "--inference_width","640","--inference_height","360",
    "--vehicle_width","200","--vehicle_height","300",
    "--lookahead_distance","150","--target_lookahead_m","4.0",
    "--spline_curvature_threshold","0.3","--straight_linear_x","1.0","--curved_linear_x","0.5"])
device = torch.device("cuda")
model, transform, decode_fn = m.build_model(opts, device)
m.warmup_model(model, device, opts)

def med(fn, n=20, w=6):
    for _ in range(w): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-a)*1000)
    ts.sort(); return statistics.median(ts)

def full(render):
    return m.process_frame(BEV, model, transform, device, decode_fn,
        conf_thresh=opts.conf_thresh, lookahead_distance=opts.lookahead_distance,
        target_lookahead_m=opts.target_lookahead_m, ros2_node=None,
        drivable_class_ids=opts.drivable_class_ids,
        inference_size=(opts.inference_width, opts.inference_height),
        vehicle_size=(opts.vehicle_width, opts.vehicle_height), use_fp16=opts.fp16,
        render_overlay=render,
        spline_curvature_threshold=opts.spline_curvature_threshold,
        straight_linear_x=opts.straight_linear_x, curved_linear_x=opts.curved_linear_x,
        min_nav_path_points=opts.min_nav_path_points,
        straight_path_bias=opts.straight_path_bias,
        nav_corridor_width=opts.nav_corridor_width,
        nav_path_offset_px=opts.nav_path_offset_px,
        nav_path_lateral_ratio=opts.nav_path_lateral_ratio)

# isolate the pieces
from PIL import Image as PILImage
def prep(size):
    small = cv2.resize(BEV, size, interpolation=cv2.INTER_AREA)
    pil = PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    t = transform(pil).unsqueeze(0).to(device)
    return t.half() if opts.fp16 else t

def fwd(size):
    t = prep(size)
    with torch.no_grad():
        return model(t).max(1)[1].cpu().numpy()[0]

print()
for size in [(640,360),(512,288),(448,256),(384,216)]:
    tp = med(lambda: prep(size))
    tf = med(lambda: fwd(size))
    print(f"  {size[0]}x{size[1]:>4}  preprocess {tp:6.2f} ms   preprocess+forward+argmax {tf:6.2f} ms   model {tf-tp:6.2f} ms")
pf_off = med(lambda: full(False))
pf_on  = med(lambda: full(True))
base = med(lambda: fwd((640,360)))
print(f"\n  process_frame overlay OFF {pf_off:6.2f} ms   of which inference {base:6.2f} ms"
      f"   -> nav/post {pf_off-base:6.2f} ms")
print(f"  process_frame overlay ON  {pf_on:6.2f} ms   -> overlay {pf_on-pf_off:6.2f} ms")

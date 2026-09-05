"""navigate.py LUT port: correctness + timing on live Xavier camera frames."""
import sys, time, statistics, numpy as np, cv2, importlib.util as iu
sys.path.insert(0, ".")

def load(name, path):
    spec = iu.spec_from_file_location(name, path); m = iu.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

ORIG = load("nav_orig", "navigate_orig.py")
FIX  = load("nav_fix",  "navigate.py")

CAMS = ["front", "left", "rear", "right"]
fr = {c: cv2.imread(f"live_raw_{c}.png") for c in CAMS}
assert all(v is not None for v in fr.values())
paths = {c: "" for c in CAMS}
car = None
try:
    from param_settings import img_car as car
except Exception:
    pass

t0 = time.perf_counter(); o = ORIG.BEVProcessor(paths, car); ti_o = (time.perf_counter()-t0)*1000
t0 = time.perf_counter(); f = FIX.BEVProcessor(paths, car);  ti_f = (time.perf_counter()-t0)*1000
print(f"constructor: original {ti_o:7.1f} ms   patched {ti_f:7.1f} ms (builds LUTs)")

print("\nequivalence on live frames:")
for c in CAMS:
    ou, ow, _ = o.process_image(fr[c], c)
    fu, fw, _ = f.process_image(fr[c], c)
    du = np.abs(ou.astype(np.int16)-fu.astype(np.int16))
    dw = np.abs(ow.astype(np.int16)-fw.astype(np.int16))
    co = int((ow.sum(axis=2) > 0).sum()); cf = int((fw.sum(axis=2) > 0).sum())
    print(f"  {c:6s} undist max {du.max():3d} p99 {np.percentile(du,99):4.1f} | "
          f"bev max {dw.max():3d} p99 {np.percentile(dw,99):4.1f} | "
          f"coverage {co} vs {cf} ({100*(cf-co)/max(co,1):+.2f}%)")

def med(fn, n=20, w=5):
    for _ in range(w): fn()
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1000)
    ts.sort(); return statistics.median(ts)

print()
for r in range(3):
    a = med(lambda: [o.process_image(fr[c], c)[1] for c in CAMS])
    b = med(lambda: [f.process_image(fr[c], c)[1] for c in CAMS])
    d = med(lambda: [f.process_image(fr[c], c, want_undistorted=False)[1] for c in CAMS])
    print(f"round {r+1}: warp4 original {a:7.2f} ms -> patched {b:6.2f} ms ({a/b:4.2f}x) | "
          f"BEV only {d:6.2f} ms ({a/d:4.2f}x)")

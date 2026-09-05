"""Benchmark BEVProcessor.process_image on the Jetson AGX Xavier.

Compares the original (cv2.undistort + cv2.warpPerspective, maps rebuilt every
frame) against the patched version (maps built once, undistort+rotate+homography
composed into a single remap LUT).

Run from ~/model/main so that yaml/ and param_settings.py resolve.
"""
import os, sys, time, statistics, json
import numpy as np
import cv2

sys.path.insert(0, ".")
import bev_processor as ORIG
import bev_processor_fixed as FIXED

CAMS = ["front", "left", "rear", "right"]
SRC = {c: f"Dataset/Img_distortion_Testing/{c.capitalize()}.jpg" for c in CAMS}
WARMUP, ITERS = 5, 25


def sysinfo(tag):
    def rd(p):
        try:
            return open(p).read().strip()
        except Exception:
            return "?"
    zones = {}
    for i in range(12):
        t = rd(f"/sys/class/thermal/thermal_zone{i}/type")
        if t != "?":
            zones[t] = int(rd(f"/sys/class/thermal/thermal_zone{i}/temp") or 0) / 1000.0
    gpu = rd("/sys/devices/gpu.0/devfreq/17000000.gv11b/cur_freq")
    cpu = [int(rd(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq") or 0) // 1000
           for i in range(8)]
    print(f"[{tag}] CPU MHz {cpu}")
    print(f"[{tag}] GPU {int(gpu)/1e6:.0f} MHz   " +
          "  ".join(f"{k}={v:.1f}C" for k, v in zones.items() if "therm" in k.lower()))


def load_images():
    imgs = {}
    for c in CAMS:
        im = cv2.imread(SRC[c])
        if im is None:
            sys.exit(f"cannot read {SRC[c]}")
        if im.shape[:2] != (720, 1280):
            im = cv2.resize(im, (1280, 720))
        imgs[c] = im
    return imgs


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "median": statistics.median(ts),
        "min": ts[0],
        "p95": ts[int(0.95 * (len(ts) - 1))],
        "mean": statistics.fmean(ts),
    }


def main():
    print(f"OpenCV {cv2.__version__}   cv2.cuda devices "
          f"{cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,'cuda') else 'n/a'}")
    print(f"threads {cv2.getNumThreads()}   iters {ITERS} (+{WARMUP} warmup)\n")
    sysinfo("before")

    imgs = load_images()
    paths = {c: SRC[c] for c in CAMS}

    t0 = time.perf_counter()
    o = ORIG.BEVProcessor(paths, img_car=None)
    t_o_init = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    f = FIXED.BEVProcessor(paths, img_car=None)
    t_f_init = (time.perf_counter() - t0) * 1000
    print(f"\nconstructor: original {t_o_init:7.1f} ms   patched {t_f_init:7.1f} ms"
          f"   (patched builds the LUTs here, once)\n")

    # ---- equivalence -------------------------------------------------
    print("equivalence (patched vs original, same input):")
    for c in CAMS:
        ou, ow = o.process_image(imgs[c], c)
        fu, fw = f.process_image(imgs[c], c)
        du = np.abs(ou.astype(np.int16) - fu.astype(np.int16))
        dw = np.abs(ow.astype(np.int16) - fw.astype(np.int16))
        cov_o = int((ow.sum(axis=2) > 0).sum())
        cov_f = int((fw.sum(axis=2) > 0).sum())
        print(f"  {c:6s} undist max {du.max():3d} p99 {np.percentile(du,99):5.1f} | "
              f"bev max {dw.max():3d} p99 {np.percentile(dw,99):5.1f} | "
              f"coverage {cov_o} vs {cov_f} ({100*(cov_f-cov_o)/max(cov_o,1):+.2f}%)")

    # ---- timing ------------------------------------------------------
    res = {}
    print(f"\nprocess_image, per camera (ms):")
    print(f"  {'cam':6s} {'original':>28s} {'patched':>28s} {'speedup':>8s}")
    tot_o = tot_f = 0.0
    for c in CAMS:
        img = imgs[c]
        for _ in range(WARMUP):
            o.process_image(img, c); f.process_image(img, c)
        ro = timeit(lambda: o.process_image(img, c), ITERS)
        rf = timeit(lambda: f.process_image(img, c), ITERS)
        res[c] = {"orig": ro, "fixed": rf}
        tot_o += ro["median"]; tot_f += rf["median"]
        print(f"  {c:6s} "
              f"med {ro['median']:7.2f} min {ro['min']:7.2f} p95 {ro['p95']:7.2f} | "
              f"med {rf['median']:7.2f} min {rf['min']:7.2f} p95 {rf['p95']:7.2f} | "
              f"{ro['median']/rf['median']:7.2f}x")
    print(f"  {'SUM':6s} {'':4s}{tot_o:7.2f}{'':22s}{tot_f:7.2f}{'':16s}"
          f"{tot_o/tot_f:7.2f}x")

    # ---- BEV-only (patched can skip the undistorted output) -----------
    tot_f2 = 0.0
    print(f"\npatched, BEV output only (want_undistorted=False):")
    for c in CAMS:
        img = imgs[c]
        for _ in range(WARMUP):
            f.process_image(img, c, want_undistorted=False)
        r = timeit(lambda: f.process_image(img, c, want_undistorted=False), ITERS)
        tot_f2 += r["median"]
        print(f"  {c:6s} med {r['median']:7.2f} ms")
    print(f"  {'SUM':6s}     {tot_f2:7.2f} ms   vs original {tot_o:7.2f} ms"
          f"   = {tot_o/tot_f2:.2f}x")

    sysinfo("\nafter")
    print(f"\nNOTE: the SUM row is the serial cost of all four cameras. "
          f"get_bev_frame() runs them in threads, and OpenCV releases the GIL, "
          f"so wall-clock per BEV frame is lower than the sum.")
    json.dump(res, open("bench_bev_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()

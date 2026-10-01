"""Which JPEG decode routes exist on THIS machine, and how fast are they?

    python probe_decoders.py                 # uses live_raw_front.png if present
    python probe_decoders.py --image x.png --iters 50

Reports, for a 1280x720 JPEG at quality 85 (what the cameras emit):
  - cv2.imdecode (CPU)            : what VideoCapture does today
  - torchvision decode_jpeg CPU   : plumbing check
  - torchvision decode_jpeg CUDA  : nvjpeg on the GPU  <- the one we want
  - GStreamer nvjpegdec           : only if OpenCV was built with GStreamer
Each is checked for correctness against cv2.imdecode and timed p50/p95.
"""
import argparse
import os
import tempfile
import time

import cv2
import numpy as np

from jpeg_decoder import JpegDecoder


def timed(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[min(len(ts) - 1, int(len(ts) * 0.95))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="live_raw_front.png")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--quality", type=int, default=85)
    a = ap.parse_args()

    img = cv2.imread(a.image) if os.path.exists(a.image) else None
    if img is None:
        print(f"(no {a.image}; using a synthetic 1280x720 frame)")
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (720, 1280, 3), np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 3)          # JPEG-realistic, not noise
    if img.shape[:2] != (720, 1280):
        img = cv2.resize(img, (1280, 720))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    buf = enc.reshape(-1)
    ref = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    print(f"test JPEG: 1280x720 q{a.quality}, {buf.size/1024:.0f} KB\n")

    try:
        build = cv2.getBuildInformation()
        gst = [l.strip() for l in build.splitlines() if "GStreamer" in l]
        print("OpenCV", cv2.__version__, "|", (gst[0] if gst else "GStreamer: (not listed)"))
    except Exception:
        pass

    p50, p95 = timed(lambda: cv2.imdecode(buf, cv2.IMREAD_COLOR), a.iters)
    print(f"  cv2.imdecode (CPU, = VideoCapture today) : p50 {p50:6.2f} ms  p95 {p95:6.2f} ms")

    for b in ("torchvision_cpu", "torchvision_cuda"):
        try:
            d = JpegDecoder(b)
            out = d.decode(buf)
            if out is None:
                raise RuntimeError("decode returned None")
            diff = int(np.abs(out.astype(np.int16) - ref.astype(np.int16)).max())
            p50, p95 = timed(lambda: d.decode(buf), a.iters)
            print(f"  {b:<40}: p50 {p50:6.2f} ms  p95 {p95:6.2f} ms   "
                  f"shape {out.shape}  max|diff| vs cv2 = {diff}")
        except Exception as e:
            print(f"  {b:<40}: unavailable ({e})")

    # GStreamer / nvjpegdec route: decode the same JPEG from a file through it.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(buf.tobytes()); jpg = f.name
    pipe = (f"filesrc location={jpg} ! jpegparse ! nvjpegdec ! nvvidconv ! "
            f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink")
    try:
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        ok, out = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and out is not None:
            diff = int(np.abs(out.astype(np.int16) - ref.astype(np.int16)).max())
            print(f"  GStreamer nvjpegdec                      : AVAILABLE  shape {out.shape}  max|diff| vs cv2 = {diff}")
        else:
            print("  GStreamer nvjpegdec                      : not available (OpenCV lacks GStreamer, or no nvjpegdec element)")
    except Exception as e:
        print(f"  GStreamer nvjpegdec                      : not available ({e})")
    finally:
        os.unlink(jpg)

    print("\n  auto-selected:", (JpegDecoder.probe(verbose=False) or type("x", (), {"backend": "none (CPU via VideoCapture)"})).backend)


if __name__ == "__main__":
    main()

"""Shared V4L2 capture configuration for the surround-view cameras.

Kept in its own module so that `bev_processor.py` (used by the saty* driving
scripts) and the standalone `navigate.py` entry point can apply identical
settings without either importing the other.

WHY THIS EXISTS
---------------
The four cameras enumerate on USB 2.0 links (480 Mbit/s), behind separate
Renesas uPD720202 controllers. At 1280x720 an uncompressed YUYV frame is
14.75 Mbit, so 30 fps would need 442 Mbit/s against a 480 Mbit/s theoretical
bus. It does not fit, and the driver caps the negotiation at 10 fps:

    v4l2-ctl --list-formats-ext        # 1280x720
      YUYV (uncompressed)  -> 10 fps
      MJPG (compressed)    -> 25 fps   (30 fps on the Logitech C615)

`cv2.VideoCapture` does not request a pixel format, so V4L2 negotiates YUYV
by default and the entire pipeline is capped at 10 fps no matter how fast the
downstream processing is. Requesting MJPG raises that ceiling to 25 fps.

Set BEV_DISABLE_MJPG=1 in the environment to fall back to the previous
behaviour if a camera misbehaves in MJPG mode.
"""

import os

import cv2

# Requested frame rate. The driver grants what the format and link allow;
# 30 is asked for so the C615 is not artificially held to 25.
CAP_FPS = 30

_DISABLE_MJPG = os.environ.get("BEV_DISABLE_MJPG", "").strip() != ""

# Where udev publishes stable, port-derived names for V4L2 devices.
V4L_DIR = "/dev/v4l"


def is_camera_source(path):
    """True when `path` refers to a live V4L2 camera rather than a video file.

    MJPG must only be forced on live cameras. Video files carry their own
    codec and setting CAP_PROP_FOURCC on them can break decoding.

    Accepts /dev/v4l/... as well as /dev/video*, so a stable by-path name is
    still recognised as a camera. Missing this would silently disable MJPG and
    the grabber thread for anything addressed by stable path.
    """
    if isinstance(path, bool):          # guard: bool is a subclass of int
        return False
    if isinstance(path, int):
        return True
    if not isinstance(path, str):
        return False
    return path.startswith("/dev/video") or path.startswith(V4L_DIR + "/")


def list_cameras():
    """Return [(stable_path, /dev/videoN)] for every capture device, sorted.

    Reads /dev/v4l/by-path, which names devices by the USB port they are
    plugged into rather than by enumeration order.
    """
    out = []
    by_path = os.path.join(V4L_DIR, "by-path")
    if not os.path.isdir(by_path):
        return out
    for name in sorted(os.listdir(by_path)):
        # Each camera exposes a capture node and a metadata node; keep capture.
        if not name.endswith("-video-index0"):
            continue
        link = os.path.join(by_path, name)
        out.append((link, os.path.realpath(link)))
    return out


def resolve_camera(spec):
    """Resolve a camera spec to something cv2.VideoCapture can open.

    Accepts, in order of preference:

      - a stable path      "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
      - a by-path basename "platform-3610000.usb-usb-0:2.1:1.0-video-index0"
      - a device path      "/dev/video2"
      - an integer index   2  or  "2"   (legacy; NOT stable across reboots)

    Stable paths are resolved to their /dev/videoN target, because that is what
    the V4L2 backend and `is_camera_source` expect.

    WHY: /dev/videoN indices are assigned in enumeration order and move between
    boots and replugs - observed on this rig going from 0,2,4,7 to 2,4,5,8.
    With four physically identical cameras there is nothing in the image to say
    which is which, so a reordering silently applies each camera's homography to
    the wrong view and the stitch is wrong with no error raised.
    """
    if isinstance(spec, int) and not isinstance(spec, bool):
        return spec
    if not isinstance(spec, str):
        return spec

    spec = spec.strip()
    if spec.isdigit():                      # legacy "--front_cam 2"
        return int(spec)
    if spec.startswith("/dev/video"):
        return spec
    if spec.startswith(V4L_DIR + "/"):
        return os.path.realpath(spec)

    candidate = os.path.join(V4L_DIR, "by-path", spec)
    if os.path.exists(candidate):
        return os.path.realpath(candidate)

    # Not recognised - hand it back untouched so video files still work.
    return spec


def fourcc_name(cap):
    """Return the capture's negotiated FOURCC as a 4-character string."""
    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))


def configure_capture(cap, path, width, height, verbose=True, raw_mjpg=False):
    """Apply resolution, and MJPG + frame rate for live cameras.

    raw_mjpg=True asks the backend NOT to decode: read()/retrieve() then return
    the compressed JPEG buffer (1xN uint8), for a jpeg_decoder.JpegDecoder to
    decode on the GPU. Only honoured when MJPG was actually negotiated.

    Returns the same `cap` so it can be used inline.
    """
    camera = is_camera_source(path) and not _DISABLE_MJPG

    # FOURCC must be set BEFORE the frame size: V4L2 renegotiates the format
    # on each set(), and a resolution is only meaningful within a format.
    if camera:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if camera:
        cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
        # Keep the driver queue shallow. cap.read() returns the OLDEST queued
        # frame, so a deep queue puts the display permanently behind reality by
        # queue_depth / fps. live_bev.py already sets this; the driving path
        # never did, which is why the monitor lags. Not every V4L2 backend
        # honours it, so camera_grabber.py bounds the latency properly - this
        # just reduces how much there is to drain.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got = fourcc_name(cap)
        raw = False
        if raw_mjpg and got == "MJPG":
            # Hand us the compressed buffer; decode happens in the grabber.
            raw = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
        if verbose:
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      cap.get(cv2.CAP_PROP_FPS))
            if got == "MJPG":
                print(f"[capture] {path}: MJPG {actual[0]}x{actual[1]} "
                      f"@ {actual[2]:.0f} fps" + ("  (raw, GPU decode)" if raw else ""))
            else:
                print(f"[capture] WARNING {path}: negotiated {got!r}, not "
                      f"MJPG - capture stays capped near 10 fps at 720p")
    return cap

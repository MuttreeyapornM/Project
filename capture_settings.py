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


def is_camera_source(path):
    """True when `path` refers to a live V4L2 camera rather than a video file.

    MJPG must only be forced on live cameras. Video files carry their own
    codec and setting CAP_PROP_FOURCC on them can break decoding.
    """
    if isinstance(path, bool):          # guard: bool is a subclass of int
        return False
    if isinstance(path, int):
        return True
    return isinstance(path, str) and path.startswith("/dev/video")


def fourcc_name(cap):
    """Return the capture's negotiated FOURCC as a 4-character string."""
    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))


def configure_capture(cap, path, width, height, verbose=True):
    """Apply resolution, and MJPG + frame rate for live cameras.

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
        if verbose:
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      cap.get(cv2.CAP_PROP_FPS))
            if got == "MJPG":
                print(f"[capture] {path}: MJPG {actual[0]}x{actual[1]} "
                      f"@ {actual[2]:.0f} fps")
            else:
                print(f"[capture] WARNING {path}: negotiated {got!r}, not "
                      f"MJPG - capture stays capped near 10 fps at 720p")
    return cap

"""Per-camera grabber thread that keeps only the newest frame.

WHY THIS EXISTS
---------------
`cv2.VideoCapture.read()` returns the OLDEST frame in the driver queue, not the
newest. Whenever the pipeline consumes slower than the camera produces, unread
frames accumulate and the displayed image sits permanently behind reality by

    latency = queue_depth / camera_fps

It reaches a steady state rather than growing without bound, which is why the
lag on the cart is a stable 2-3 s rather than something that gets steadily
worse.

`CAP_PROP_BUFFERSIZE = 1` (see capture_settings.py) helps, but not every V4L2
backend honours it. This bounds the latency regardless: a dedicated thread
consumes frames as fast as the camera emits them, so the driver queue never
backs up, and the consumer always gets the most recent frame.

NOTE ON `LatestBEVCapture`
-------------------------
The saty*.py scripts already wrap the pipeline in `LatestBEVCapture`, which
keeps the newest *processed BEV frame*. That fixes staleness on the consumer
side only - its producer thread still called `cap.read()` and so was itself
being handed stale frames off the front of the queue. It delivered the freshest
available *stale* frame. The backlog is upstream of it, which is what this fixes.

COST, STATED HONESTLY
---------------------
This thread decodes at the CAMERA's rate rather than the consumer's, so when
the pipeline runs slower than the camera it does strictly more decode work than
before - roughly (camera_fps / loop_fps) times as many decodes. With MJPG that
is JPEG decode on the CPU. It buys bounded latency, which is the safety-relevant
property; moving the decode to `nvjpegdec` on the GPU is the follow-up that
removes the cost.

Set BEV_DISABLE_GRABBER=1 to fall back to direct cap.read() for A/B testing.
"""

import os
import threading
import time

DISABLED = os.environ.get("BEV_DISABLE_GRABBER", "").strip() != ""


class CameraGrabber:
    """Owns one VideoCapture and publishes only its most recent frame.

    Only ever attach this to a LIVE CAMERA. On a video file it would race
    through the file at thread speed rather than following the consumer.
    """

    def __init__(self, cap, name=""):
        self.cap = cap
        self.name = str(name)
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._running = False
        self._thread = None
        self.fail_count = 0

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"grab:{self.name}"
        )
        self._thread.start()
        return self

    def _run(self):
        while self._running:
            # grab() blocks until the next frame is available, so this loop
            # runs at exactly the camera rate and keeps the queue drained.
            if not self.cap.grab():
                self.fail_count += 1
                time.sleep(0.005)
                continue
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                self.fail_count += 1
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, timeout=1.0):
        """Return (ok, frame) for the newest frame received.

        Mirrors the cv2 read() signature so it can be swapped in directly.
        Never returns a stale queue entry. Blocks only until the first frame
        arrives after start-up.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._frame is not None:
                    return True, self._frame
            if time.time() >= deadline or not self._running:
                return False, None
            time.sleep(0.001)

    @property
    def seq(self):
        """Frame counter, for detecting a stalled camera."""
        with self._lock:
            return self._seq

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def release(self):
        self.stop()
        try:
            self.cap.release()
        except Exception:
            pass

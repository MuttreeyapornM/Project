"""MJPEG frame decode off the CPU.

WHY
---
PR #1 forces MJPG on the cameras (the only way past 10 fps on their USB 2.0
links). The cost moved rather than vanished: every frame now arrives as a JPEG
and cv2.VideoCapture decodes it in libjpeg on a Carmel core. At 720p that is
roughly 4-6 ms per frame per camera; four cameras at 25-30 fps is 100-120
decodes a second, on the same cores that run the warp, the stitch and nav -
while the GPU idles at ~115 MHz.

HOW
---
Ask V4L2 for the raw compressed buffer (CAP_PROP_CONVERT_RGB = 0) and decode it
ourselves on the GPU. The grabber thread hands each raw buffer to a JpegDecoder
and publishes the decoded BGR frame, so nothing downstream changes.

Backends, best first:

  torchvision_cuda  torchvision.io.decode_jpeg(device="cuda") - nvjpeg on the
                    GPU. NVIDIA's Jetson torchvision wheels are built with it.
  torchvision_cpu   same call on the CPU (libjpeg-turbo). No faster than cv2;
                    exists so the plumbing can be tested where there is no GPU.
  cv2               cv2.imdecode - identical to what VideoCapture did before.

Not implemented here: the `nvjpegdec` GStreamer element (NVJPG hardware
block). It needs an OpenCV built with GStreamer; the venv's pip wheel almost
certainly is not. probe_decoders.py reports whether that route exists on the
machine so the decision is made on evidence.

Selection: BEV_JPEG_DECODER = auto (default) | off | cv2 | torchvision_cpu |
torchvision_cuda. `off` restores the previous behaviour (VideoCapture decodes).

COST STATED PLAINLY
-------------------
The decoded frame still comes back to the CPU (`.cpu()`), because the remap
and stitch are CPU OpenCV today. On Jetson's unified memory that copy is
~1 ms for 720p. The full win arrives when the warp moves to the GPU/PVA too and
the frame never leaves.
"""

import os
import time

import numpy as np

_MODE = os.environ.get("BEV_JPEG_DECODER", "auto").strip().lower() or "auto"
VALID_MODES = ("auto", "off", "cv2", "torchvision_cpu", "torchvision_cuda")
JPEG_DECODER_MODE = _MODE if _MODE in VALID_MODES else "auto"


def is_raw_jpeg(frame):
    """True if `frame` is a raw compressed buffer rather than a decoded image.

    With CAP_PROP_CONVERT_RGB=0 the V4L2 backend returns the JPEG bytes as a
    1xN (or Nx1) uint8 Mat. A decoded frame is HxWx3.
    """
    if frame is None or frame.dtype != np.uint8:
        return False
    if frame.ndim == 1:
        return frame.size > 4 and frame[0] == 0xFF and frame[1] == 0xD8
    if frame.ndim == 2 and (frame.shape[0] == 1 or frame.shape[1] == 1):
        flat = frame.reshape(-1)
        return flat.size > 4 and flat[0] == 0xFF and flat[1] == 0xD8
    return False


class JpegDecoder:
    def __init__(self, backend):
        if backend not in ("cv2", "torchvision_cpu", "torchvision_cuda"):
            raise ValueError(backend)
        self.backend = backend
        self.is_gpu = backend == "torchvision_cuda"
        self.n = 0
        self.fail = 0
        self.total_ms = 0.0
        if backend.startswith("torchvision"):
            import torch
            from torchvision.io import ImageReadMode, decode_jpeg
            self._torch = torch
            self._decode = decode_jpeg
            self._mode = ImageReadMode.RGB
            self._device = "cuda" if self.is_gpu else "cpu"
            if self.is_gpu and not torch.cuda.is_available():
                raise RuntimeError("torchvision_cuda requested but CUDA is unavailable")
        else:
            import cv2
            self._cv2 = cv2

    # --- one frame ------------------------------------------------------
    def decode(self, buf):
        """Raw JPEG bytes (np.uint8 array, any shape, or bytes) -> BGR HxWx3 uint8.

        Returns None on a corrupt buffer instead of raising; the grabber counts
        it and keeps the previous frame.
        """
        t0 = time.perf_counter()
        try:
            if self.backend == "cv2":
                arr = np.frombuffer(buf, np.uint8) if isinstance(buf, (bytes, bytearray)) else buf.reshape(-1)
                out = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
            else:
                if isinstance(buf, (bytes, bytearray)):
                    data = self._torch.frombuffer(bytearray(buf), dtype=self._torch.uint8)
                else:
                    data = self._torch.from_numpy(np.ascontiguousarray(buf.reshape(-1)))
                img = self._decode(data, mode=self._mode, device=self._device)   # (3,H,W) RGB
                # RGB->BGR and CHW->HWC, then to host. flip(0) does the channel swap.
                out = img.flip(0).permute(1, 2, 0).contiguous()
                out = out.cpu().numpy() if self.is_gpu else out.numpy()
        except Exception:
            out = None
        self.n += 1
        if out is None:
            self.fail += 1
        self.total_ms += (time.perf_counter() - t0) * 1000.0
        return out

    @property
    def mean_ms(self):
        return self.total_ms / self.n if self.n else 0.0

    # --- choose a backend -----------------------------------------------
    @classmethod
    def probe(cls, mode=None, verbose=True):
        """Return a working decoder for `mode` (default: env BEV_JPEG_DECODER),
        or None for `off`. `auto` tries torchvision_cuda, then falls back to
        None (= keep VideoCapture's own decode; no point re-doing it on CPU)."""
        mode = (mode or JPEG_DECODER_MODE)
        if mode == "off":
            return None
        order = ["torchvision_cuda"] if mode == "auto" else [mode]
        for b in order:
            try:
                d = cls(b)
                # Smoke-test on a tiny synthetic JPEG so a broken install fails
                # here, at start-up, not on the first camera frame.
                if d.decode(_tiny_jpeg()) is None:
                    raise RuntimeError("smoke decode returned None")
                if verbose:
                    print(f"[jpeg] decoder: {b}" + (" (GPU)" if d.is_gpu else ""))
                return d
            except Exception as e:
                if verbose:
                    print(f"[jpeg] {b} unavailable: {e}")
        if verbose and mode == "auto":
            print("[jpeg] no GPU decoder; VideoCapture keeps decoding on the CPU")
        return None


def _tiny_jpeg():
    import cv2
    ok, enc = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    return enc.reshape(-1)

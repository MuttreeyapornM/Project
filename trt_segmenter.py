"""DeepLabV3+ (MobileNet) as a TensorRT FP16 engine, drop-in for the PyTorch model.

WHY
---
The PyTorch path already runs FP16, so this is not FP32->FP16. It is
PyTorch->TensorRT at the same precision: layer fusion (conv+BN+ReLU into one
kernel), kernels auto-tuned for THIS GPU, a static memory plan, and no Python
dispatcher in the loop. MobileNet is dozens of small depthwise convolutions, so
launch overhead dominates in PyTorch; TensorRT collapses it. Expected 2-4x on
inference, deterministic latency (pulls the p95 in), and the Carmel cores get
back the time PyTorch spent dispatching.

DESIGN
------
- The engine outputs LOGITS (1,C,H,W) in FP16. Argmax is done in torch on the
  output buffer. TensorRT 8.5 (JetPack 5.1.x) cannot emit int64 outputs, which
  is what ONNX ArgMax produces, so folding argmax into the graph is fragile
  across versions. One torch.argmax on a (1,9,360,640) tensor is ~0.2 ms.
- I/O buffers are torch CUDA tensors and bindings are their data_ptr(). No
  pycuda dependency; torch is already installed with CUDA on the vehicle.
- Preprocessing (resize, BGR->RGB, /255, ImageNet normalise) mirrors
  saty5.build_model()'s transform exactly, but runs in torch on the GPU and
  skips the PIL round-trip.

USAGE
-----
    python export_trt.py --ckpt checkpoints_zoo/best27000.pth --out seg_640x360_fp16.engine
    python validate_trt.py --ckpt ... --engine seg_640x360_fp16.engine --images 'live_raw_*.png'
    ./run_navigate.sh ... --trt_engine seg_640x360_fp16.engine

Engines are specific to the GPU and TensorRT version they were built on. One
built on the Xavier will not load on the Orin Nano; rebuild there.
"""

import os
import sys
import time

import numpy as np
import torch

# The TensorRT python bindings on JetPack live in the system dist-packages.
# A venv created without --system-site-packages cannot see them.
try:
    import tensorrt as trt
except ImportError:
    for _p in ("/usr/lib/python3.8/dist-packages", "/usr/lib/python3.10/dist-packages"):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)
    try:
        import tensorrt as trt
    except ImportError:
        trt = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mirrors saty5.build_model(): the "custom" dataset has 9 classes.
NUM_CLASSES = {"voc": 21, "cityscapes": 19, "custom": 9}


def load_pytorch_model(ckpt, dataset="custom", arch="deeplabv3plus_mobilenet",
                       output_stride=16, device="cuda"):
    """Build the PyTorch model exactly as saty5.build_model() does.

    Kept here (rather than importing saty5) because saty5 imports ROS 2 and
    starts nothing useful for an offline export.
    """
    import network  # repo-local

    model = network.modeling.__dict__[arch](
        num_classes=NUM_CLASSES[dataset.lower()],
        output_stride=output_stride,
        pretrained_backbone=False,       # weights come from the checkpoint
    )
    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        state = ckpt_obj["model_state"]          # training checkpoint
    elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state = ckpt_obj["state_dict"]
    else:
        state = ckpt_obj
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def export_onnx(model, onnx_path, width=640, height=360, opset=13):
    """Trace the model to ONNX at a fixed input shape (FP32 graph; FP16 is a
    build-time decision in TensorRT)."""
    try:
        import onnx  # noqa: F401  - torch.onnx.export raises without it
    except ImportError as e:
        raise RuntimeError("the 'onnx' package is required for export: pip install onnx") from e
    model = model.float().eval()
    dummy = torch.zeros(1, 3, height, width, device=next(model.parameters()).device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True, dynamic_axes=None,
    )
    import onnx  # hard requirement of torch.onnx.export itself (pip install onnx)
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"[export] ONNX ok: {onnx_path}")
    return onnx_path


def build_engine(onnx_path, engine_path, fp16=True, workspace_mb=2048, dla_core=None):
    """ONNX -> serialised TensorRT engine, using the python builder so this does
    not depend on trtexec's CLI flags, which changed between 8.x releases."""
    if trt is None:
        raise RuntimeError("tensorrt python bindings not importable - run with the "
                           "system python3 or create the venv with --system-site-packages")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[build] parser error:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    except AttributeError:                       # TensorRT < 8.4
        config.max_workspace_size = workspace_mb << 20
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[build] WARNING: platform reports no fast FP16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    if dla_core is not None:
        # Xavier has two DLA cores. Layers DLA cannot run fall back to the GPU.
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla_core)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build] engine written: {engine_path}  ({os.path.getsize(engine_path)/1e6:.1f} MB, "
          f"{time.time()-t0:.0f} s)")
    return engine_path


class TRTSegmenter:
    """Runs a serialised engine. Duck-types enough of the PyTorch model for
    saty5.process_frame() to swap it in (see `is_trt`)."""

    is_trt = True

    def __init__(self, engine_path, width=640, height=360, device="cuda"):
        if trt is None:
            raise RuntimeError("tensorrt python bindings not importable")
        self.width, self.height = int(width), int(height)
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {engine_path} - was it built on this GPU/TensorRT?")
        self.context = self.engine.create_execution_context()

        # Bindings by name so binding order in the engine does not matter.
        self.in_idx = self.engine.get_binding_index("input")
        self.out_idx = self.engine.get_binding_index("logits")
        in_shape = tuple(self.engine.get_binding_shape(self.in_idx))
        out_shape = tuple(self.engine.get_binding_shape(self.out_idx))
        if in_shape[2:] != (self.height, self.width):
            raise RuntimeError(f"engine input is {in_shape}, expected (1,3,{self.height},{self.width})")
        in_dtype = trt.nptype(self.engine.get_binding_dtype(self.in_idx))
        out_dtype = trt.nptype(self.engine.get_binding_dtype(self.out_idx))
        self._in = torch.empty(in_shape, dtype=torch.from_numpy(np.zeros(1, in_dtype)).dtype, device=self.device)
        self._out = torch.empty(out_shape, dtype=torch.from_numpy(np.zeros(1, out_dtype)).dtype, device=self.device)
        self.num_classes = out_shape[1]
        self._bindings = [0] * self.engine.num_bindings
        self._bindings[self.in_idx] = self._in.data_ptr()
        self._bindings[self.out_idx] = self._out.data_ptr()
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.stream = torch.cuda.Stream(device=self.device)

    # --- inference ----------------------------------------------------
    def preprocess(self, frame_bgr):
        """uint8 BGR (H,W,3) at ANY size -> normalised (1,3,h,w) on the GPU.

        Same maths as saty5's transform (ToTensor + ImageNet Normalize) after a
        cv2.INTER_AREA resize, minus the PIL round-trip.
        """
        import cv2
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        t = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)   # BGR->RGB, NCHW
        t = (t - self._mean) / self._std
        return t.to(self._in.dtype)

    @torch.inference_mode()
    def logits(self, frame_bgr):
        """Returns the (1,C,h,w) logits tensor (a view of the output buffer)."""
        self._in.copy_(self.preprocess(frame_bgr))
        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v2(self._bindings, self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute failed")
        return self._out

    @torch.inference_mode()
    def infer(self, frame_bgr):
        """uint8 BGR frame -> (h,w) uint8 class map at the inference size."""
        return torch.argmax(self.logits(frame_bgr), dim=1)[0].to(torch.uint8).cpu().numpy()

    def warmup(self, n=10):
        dummy = np.zeros((self.height, self.width, 3), np.uint8)
        for _ in range(n):
            self.infer(dummy)

    # PyTorch-model API shims so existing call sites do not need to change.
    def eval(self):
        return self

    def half(self):
        return self

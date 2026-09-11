"""Checkpoint -> ONNX -> TensorRT FP16 engine, at the pipeline's inference size.

    python export_trt.py --ckpt checkpoints_zoo/best27000.pth --out seg_640x360_fp16.engine
    python export_trt.py ... --dla 0        # build for DLA core 0, GPU fallback

Run with an interpreter that can import tensorrt (system python3 on JetPack,
or a venv made with --system-site-packages). Rebuild per target GPU.
"""
import argparse
import os

import torch

from trt_segmenter import build_engine, export_onnx, load_pytorch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="seg_640x360_fp16.engine")
    ap.add_argument("--onnx", default=None, help="where to write the ONNX (default: next to --out)")
    ap.add_argument("--dataset", default="custom", choices=["voc", "cityscapes", "custom"])
    ap.add_argument("--model", default="deeplabv3plus_mobilenet")
    ap.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--dla", type=int, default=None, help="DLA core (0 or 1); omit for GPU")
    ap.add_argument("--workspace_mb", type=int, default=2048)
    ap.add_argument("--opset", type=int, default=13)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    onnx_path = a.onnx or os.path.splitext(a.out)[0] + ".onnx"

    print(f"[export] loading {a.ckpt} ({a.model}, os{a.output_stride}, {a.dataset}) on {device}")
    model = load_pytorch_model(a.ckpt, a.dataset, a.model, a.output_stride, device)
    export_onnx(model, onnx_path, a.width, a.height, a.opset)
    build_engine(onnx_path, a.out, fp16=not a.no_fp16, workspace_mb=a.workspace_mb, dla_core=a.dla)
    print("[export] done. Validate before driving on it:")
    print(f"  python validate_trt.py --ckpt {a.ckpt} --engine {a.out} --images 'live_raw_*.png'")


if __name__ == "__main__":
    main()

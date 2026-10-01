"""Is the engine faithful to the PyTorch model, and how much faster is it?

    python validate_trt.py --ckpt checkpoints_zoo/best27000.pth \
        --engine seg_640x360_fp16.engine --images 'live_raw_*.png'
    python validate_trt.py ... --labels path/to/masks   # also mIoU vs ground truth

Reports, over every image matched by --images:
  - pixel agreement between PyTorch-FP16 and TensorRT-FP16 argmax
  - per-class IoU of TensorRT against PyTorch (treating PyTorch as reference)
  - if --labels is given (PNG class maps, same basename): mIoU of each runtime
    against ground truth, and the delta
  - inference time for each runtime (median / p95 over the set)

Two FP16 runtimes will not agree bit-for-bit: expect >99% pixel agreement with
disagreement concentrated on class boundaries. A large drop is an export fault.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch

from trt_segmenter import IMAGENET_MEAN, IMAGENET_STD, TRTSegmenter, load_pytorch_model


def confusion(pred, ref, n):
    """n x n confusion matrix, rows = ref, cols = pred (ignores ref==255)."""
    m = ref != 255
    return np.bincount(n * ref[m].astype(np.int64) + pred[m].astype(np.int64), minlength=n * n).reshape(n, n)


def iou_from_confusion(cm):
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def pytorch_pred(model, frame_bgr, w, h, device, fp16=True):
    """Exactly saty5.process_frame()'s preprocessing + argmax."""
    from PIL import Image as PILImage
    from torchvision import transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    x = tf(PILImage.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    if fp16 and device.type == "cuda":
        x = x.half()
    with torch.inference_mode():
        out = torch.argmax(model(x), dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out.to(torch.uint8).cpu().numpy()


def timed(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[min(len(ts) - 1, int(len(ts) * 0.95))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--images", required=True, help="glob, e.g. 'live_raw_*.png'")
    ap.add_argument("--labels", default=None, help="dir of PNG class maps with the same basenames")
    ap.add_argument("--dataset", default="custom", choices=["voc", "cityscapes", "custom"])
    ap.add_argument("--model", default="deeplabv3plus_mobilenet")
    ap.add_argument("--output_stride", type=int, default=16)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--timing_iters", type=int, default=30)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(a.images))
    if not paths:
        raise SystemExit(f"no images match {a.images!r}")

    pt = load_pytorch_model(a.ckpt, a.dataset, a.model, a.output_stride, device)
    if device.type == "cuda":
        pt = pt.half()
    tr = TRTSegmenter(a.engine, a.width, a.height, device)
    n = tr.num_classes
    tr.warmup()

    agree, total = 0, 0
    cm_tr_vs_pt = np.zeros((n, n), np.int64)
    cm_pt_gt = np.zeros((n, n), np.int64)
    cm_tr_gt = np.zeros((n, n), np.int64)
    have_gt = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  skip (unreadable): {p}"); continue
        a_pt = pytorch_pred(pt, img, a.width, a.height, device)
        a_tr = tr.infer(img)
        agree += int((a_pt == a_tr).sum()); total += a_pt.size
        cm_tr_vs_pt += confusion(a_tr, a_pt, n)
        if a.labels:
            lp = os.path.join(a.labels, os.path.splitext(os.path.basename(p))[0] + ".png")
            gt = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if gt is not None:
                if gt.ndim == 3: gt = gt[..., 0]
                gt = cv2.resize(gt, (a.width, a.height), interpolation=cv2.INTER_NEAREST)
                cm_pt_gt += confusion(a_pt, gt, n); cm_tr_gt += confusion(a_tr, gt, n); have_gt += 1

    print(f"\n{len(paths)} images at {a.width}x{a.height}, {n} classes")
    print(f"  pixel agreement TensorRT vs PyTorch : {100.0 * agree / max(total, 1):6.2f} %")
    iou = iou_from_confusion(cm_tr_vs_pt)
    print("  per-class IoU (TRT vs PyTorch)       : " + "  ".join(
        f"{i}:{v:.3f}" if not np.isnan(v) else f"{i}:  -  " for i, v in enumerate(iou)))
    print(f"  mean (present classes)               : {np.nanmean(iou):.4f}")
    if have_gt:
        m_pt, m_tr = np.nanmean(iou_from_confusion(cm_pt_gt)), np.nanmean(iou_from_confusion(cm_tr_gt))
        print(f"  mIoU vs ground truth ({have_gt} labelled): PyTorch {m_pt:.4f}   TensorRT {m_tr:.4f}   "
              f"delta {m_tr - m_pt:+.4f}")

    img = cv2.imread(paths[0])
    p50, p95 = timed(lambda: pytorch_pred(pt, img, a.width, a.height, device), a.timing_iters)
    t50, t95 = timed(lambda: tr.infer(img), a.timing_iters)
    print(f"\n  inference, preprocess+forward+argmax, {a.timing_iters} iters:")
    print(f"    PyTorch FP16  p50 {p50:6.2f} ms   p95 {p95:6.2f} ms")
    print(f"    TensorRT FP16 p50 {t50:6.2f} ms   p95 {t95:6.2f} ms   -> {p50 / max(t50, 1e-6):.2f}x")


if __name__ == "__main__":
    main()

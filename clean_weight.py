import torch
import sys


def patch_numpy_core_alias():
    try:
        import numpy.core as numpy_core
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in ("multiarray", "numeric", "umath"):
        try:
            module = __import__(f"numpy.core.{name}", fromlist=[name])
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


patch_numpy_core_alias()

# โหลด checkpoint เดิม
checkpoint = torch.load("./checkpoints/iter_27000_deeplabv3plus_mobilenet_custom_os16.pth", map_location='cpu', weights_only=False)

# ดึงเฉพาะ model_state
model_state = checkpoint["model_state"]

# บันทึกเฉพาะ state_dict ให้เป็นไฟล์ใหม่
torch.save(model_state, "./checkpoints_zoo/best27000.pth")

print("✅ Saved")

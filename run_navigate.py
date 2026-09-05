"""Launcher for navigate.py that fixes the environment first.

Why this exists: the checkpoints in checkpoints/ were pickled on a machine with
numpy >= 2.0, which renamed the internal package numpy.core -> numpy._core.
This box has numpy 1.24.4, so torch.load() dies with
    ModuleNotFoundError: No module named 'numpy._core'
Aliasing the old names onto the new ones lets the pickle resolve. Nothing else
about the model changes -- the tensors are identical.

Usage: exactly like navigate.py, e.g.
    ./run_navigate.sh --use_camera --front_cam 2 ... --ckpt checkpoints/iter_27000_...pth
"""
import sys
import numpy, numpy.core, numpy.core.multiarray, numpy.core.numeric

for _name in list(sys.modules):
    if _name.startswith("numpy.core"):
        sys.modules[_name.replace("numpy.core", "numpy._core", 1)] = sys.modules[_name]
sys.modules["numpy._core"] = numpy.core

import runpy
runpy.run_path("navigate.py", run_name="__main__")

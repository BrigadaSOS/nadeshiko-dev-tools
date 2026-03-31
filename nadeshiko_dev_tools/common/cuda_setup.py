"""NVIDIA CUDA 12 compatibility setup.

onnxruntime-gpu requires CUDA 12 runtime libs. When the system ships CUDA 13+,
we use pip-installed nvidia-* packages that bundle compatible CUDA 12 libs.
LD_LIBRARY_PATH must be set before process start (ld.so caches it), so we
re-exec ourselves if needed.

Call ensure_nvidia_cuda12_libs() at the top of any CLI that needs GPU.
"""

import importlib
import os
import sys


def ensure_nvidia_cuda12_libs():
    """Re-exec with pip-installed NVIDIA CUDA 12 libs in LD_LIBRARY_PATH."""
    if os.environ.get("_NVIDIA_LIBS_SET"):
        return

    nvidia_packages = [
        "nvidia.cuda_runtime",
        "nvidia.cublas",
        "nvidia.cufft",
        "nvidia.cudnn",
        "nvidia.cuda_nvrtc",
        "nvidia.nvjitlink",
    ]
    lib_dirs = []
    for pkg in nvidia_packages:
        try:
            m = importlib.import_module(pkg)
            lib_dir = os.path.join(m.__path__[0], "lib")
            if os.path.isdir(lib_dir):
                lib_dirs.append(lib_dir)
        except ImportError:
            pass

    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = ":".join(lib_dirs)
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{existing}" if existing else new_path
    os.environ["_NVIDIA_LIBS_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

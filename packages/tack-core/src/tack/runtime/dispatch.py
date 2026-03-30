"""Tack runtime dispatch — backend selection and kernel execution."""

import platform

_current_backend = None

_BACKEND_HELP = {
    "cpu": "Requires llvmlite: pip install 'tack[cpu]'",
    "metal": "Requires macOS with Apple Silicon and pyobjc-framework-Metal",
    "cuda": "Requires NVIDIA GPU with CUDA toolkit and cuda-python>=13.2",
    "hip": "Requires AMD GPU with ROCm and hip-python",
    "level_zero": "Requires Intel GPU with Level Zero runtime and libocloc",
}


def init(arch: str = "cpu"):
    """Initialize Tack with a specific backend architecture.

    Set ``TACK_NO_REINIT=1`` to skip re-initialization when a backend
    is already active (useful when embedded in an ANARI device that
    shares the same process).
    """
    global _current_backend

    import os
    if _current_backend is not None and os.environ.get("TACK_NO_REINIT"):
        return

    _constructors = {
        "cpu": ("tack.runtime.cpu", "CPUBackend"),
        "metal": ("tack.runtime.metal", "MetalBackend"),
        "cuda": ("tack.runtime.cuda_backend", "CUDABackend"),
        "hip": ("tack.runtime.hip_backend", "HIPBackend"),
        "level_zero": ("tack.runtime.level_zero_backend", "LevelZeroBackend"),
    }

    if arch not in _constructors:
        available = ", ".join(sorted(_constructors.keys()))
        raise ValueError(f"Unknown architecture: '{arch}'. Available: {available}")

    module_name, class_name = _constructors[arch]
    help_msg = _BACKEND_HELP[arch]

    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        _current_backend = cls()
    except ImportError as e:
        raise RuntimeError(
            f"Cannot initialize '{arch}' backend: missing dependency.\n"
            f"  {e}\n"
            f"  {help_msg}"
        ) from None
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot initialize '{arch}' backend on {platform.system()}.\n"
            f"  {e}\n"
            f"  {help_msg}"
        ) from None


def get_backend():
    """Get the current backend, initializing CPU if needed."""
    global _current_backend
    if _current_backend is None:
        init("cpu")
    return _current_backend

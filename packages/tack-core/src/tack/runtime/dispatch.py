"""Tack runtime dispatch — backend selection and kernel execution."""

import platform

_current_backend = None

# What to actually do when a backend will not start. Two of these cannot be
# expressed as a pip extra at all, so the message has to carry them: hip-python
# is published on Test PyPI, and Level Zero needs system libraries rather than
# a Python package. Saying only "requires hip-python" sends people to
# `pip install hip-python`, which fails with a confusing "no matching
# distribution".
_BACKEND_HELP = {
    "cpu": "Requires llvmlite:  pip install 'tack-core[cpu]'",
    "metal": ("Requires macOS on Apple Silicon:  "
              "pip install 'tack-core[metal]'"),
    "cuda": ("Requires an NVIDIA GPU and the CUDA toolkit:  "
             "pip install 'tack-core[cuda]'"),
    "hip": (
        "Requires an AMD GPU with ROCm, plus hip-python.\n"
        "  hip-python is on Test PyPI, not PyPI, so the [hip] extra cannot\n"
        "  declare it. Install it directly:\n"
        "    pip install --pre --index-url https://test.pypi.org/simple/ \\\n"
        "      --extra-index-url https://pypi.org/simple/ 'hip-python~=7.1.0'"
    ),
    "level_zero": (
        "Requires an Intel GPU with the Level Zero runtime.\n"
        "  These are system libraries, not Python packages, so the\n"
        "  [level_zero] extra has nothing to install: you need\n"
        "  libze_loader.so (level-zero runtime) and libocloc.so\n"
        "  (intel-opencl-icd)."
    ),
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
        ) from e
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot initialize '{arch}' backend on {platform.system()}.\n"
            f"  {e}\n"
            f"  {help_msg}"
        ) from e


def get_backend():
    """Get the current backend, initializing CPU if needed."""
    global _current_backend
    if _current_backend is None:
        init("cpu")
    return _current_backend

"""Tack runtime dispatch — backend selection and kernel execution."""

import platform

_current_backend = None

# What to actually do when a backend will not start. Two of these want more
# than the extra's name, for opposite reasons: [hip] installs hip-python but
# cannot know which ROCm the machine has, and [level_zero] installs nothing
# at all because its dependencies are system libraries.
#
# This text has been wrong once already. It used to route people to Test
# PyPI, which was the only place hip-python was published; the package moved
# to PyPI proper, the extra started declaring it, and the message stayed
# behind — pointing at the wrong index, for a version the extra deliberately
# does not pin. It prints only where there is no ROCm and somebody asks for
# HIP, so nothing catches it going stale except reading it.
_BACKEND_HELP = {
    "cpu": "Requires llvmlite:  pip install 'tack-core[cpu]'",
    "metal": ("Requires macOS on Apple Silicon:  "
              "pip install 'tack-core[metal]'"),
    "cuda": ("Requires an NVIDIA GPU and the CUDA toolkit:  "
             "pip install 'tack-core[cuda]'"),
    "hip": (
        "Requires an AMD GPU with ROCm, plus hip-python:\n"
        "    pip install 'tack-core[hip]'\n"
        "  hip-python ships manylinux x86_64 wheels only, and its version\n"
        "  tracks the ROCm release it binds to — 7.1.x against ROCm 7.1,\n"
        "  7.2.x against 7.2. The extra sets a lower bound rather than a\n"
        "  pin, so a system on a different ROCm wants it spelled out:\n"
        "    pip install 'hip-python~=7.2.0'"
    ),
    "level_zero": (
        "Requires an Intel GPU with the Level Zero runtime.\n"
        "  These are system libraries, not Python packages, so the\n"
        "  [level_zero] extra has nothing to install: you need\n"
        "  libze_loader.so (level-zero runtime) and libocloc.so\n"
        "  (intel-opencl-icd)."
    ),
}


def env_flag(name: str) -> bool:
    """Whether a boolean environment variable is switched on.

    Unset, empty, ``0``, ``false``, ``no`` and ``off`` all mean off, so
    ``TACK_NO_REINIT=0`` disables the option rather than enabling it.
    """
    import os
    value = os.environ.get(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def init(arch: str = "cpu", **options):
    """Initialize Tack with a specific backend architecture.

    Keyword options go to the backend and must be ones it declares in
    `Backend.init_options`. Level Zero accepts ``external_context``, the
    driver, device and context handles of a context owned by another
    library, so both can address the same device memory.

    Set ``TACK_NO_REINIT=1`` to skip re-initialization when a backend
    is already active (useful when embedded in an ANARI device that
    shares the same process).
    """
    global _current_backend

    if _current_backend is not None and env_flag("TACK_NO_REINIT"):
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
        unknown = sorted(set(options) - cls.init_options)
        if unknown:
            accepted = ", ".join(sorted(cls.init_options)) or "none"
            raise ValueError(
                f"The '{arch}' backend does not accept the option(s) "
                f"{', '.join(unknown)}. Accepted: {accepted}.")
        _current_backend = cls(**options)
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

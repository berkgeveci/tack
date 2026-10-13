"""Which NVRTC releases the CUDA backend warns about.

Kept apart from ``cuda_backend`` so the rule can be tested without
cuda-python. NVRTC 12.0 miscompiled tack-examples' lattice Boltzmann
example on an H100: density off by up to 1.6e-2 against NumPy, where
NVRTC 13.2 agreed to 2.3e-6, as the CPU and HIP backends did. It was not
reduced to a kernel, so there is no workaround to apply, only a warning.
Tack is tested with NVRTC 13.
"""

OLDEST_TESTED_MAJOR = 13


def old_nvrtc_message(major, minor):
    """The warning for NVRTC ``major.minor``, or None if it is recent enough."""
    if major >= OLDEST_TESTED_MAJOR:
        return None
    return (
        f"NVRTC {major}.{minor} is older than the NVRTC {OLDEST_TESTED_MAJOR} Tack is "
        f"tested with, and NVRTC 12.0 miscompiled a Tack kernel. Install the "
        f"nvidia-cuda-nvrtc wheel for a CUDA version no newer than your driver's "
        f"(the 'CUDA Version' nvidia-smi reports), e.g. "
        f"pip install 'nvidia-cuda-nvrtc==13.2.*' for a CUDA 13.2 driver; "
        f"cuda-python finds it before a system toolkit."
    )

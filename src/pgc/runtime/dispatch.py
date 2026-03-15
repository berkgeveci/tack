"""PGC runtime dispatch — backend selection and kernel execution."""

_current_backend = None


def init(arch: str = "cpu"):
    """Initialize PGC with a specific backend architecture."""
    global _current_backend

    if arch == "cpu":
        from pgc.runtime.cpu import CPUBackend
        _current_backend = CPUBackend()
    elif arch == "metal":
        from pgc.runtime.metal import MetalBackend
        _current_backend = MetalBackend()
    elif arch == "cuda":
        from pgc.runtime.cuda_backend import CUDABackend
        _current_backend = CUDABackend()
    elif arch == "vulkan":
        from pgc.runtime.vulkan_backend import VulkanBackend
        _current_backend = VulkanBackend()
    elif arch == "hip":
        from pgc.runtime.hip_backend import HIPBackend
        _current_backend = HIPBackend()
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_backend():
    """Get the current backend, initializing CPU if needed."""
    global _current_backend
    if _current_backend is None:
        init("cpu")
    return _current_backend

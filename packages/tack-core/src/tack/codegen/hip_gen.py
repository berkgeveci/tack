"""Tack HIP C code generation — transforms Tack IR to HIP C source for hipRTC.

Generates an ``extern "C" __global__`` kernel function where:
  - Each Field parameter becomes a typed device pointer (``float*``, etc.)
  - The outermost parallel for-loop maps to the standard HIP thread index:
        int __idx__ = blockIdx.x * blockDim.x + threadIdx.x;
    with a bounds guard.
  - Sequential for-loops, while-loops, if/else map to standard C control flow.
  - Math builtins map to HIP device math functions (sqrtf, sinf, etc.).

HIP device code is nearly source-identical to CUDA device code.  The main
differences are in the host-side runtime API, not the kernel language.
This module reuses the CUDA codegen with a HIP-specific header.
"""

from tack.lang import ir
from tack.codegen.cuda_gen import CUDACodeGen


class HIPCodeGen(CUDACodeGen):
    """Generates HIP C source from a Tack IR function.

    HIP device kernels use the same syntax as CUDA (blockIdx, threadIdx,
    __global__, etc.).  The only difference is the required header include.
    """

    def generate(self) -> str:
        """Generate HIP C source for the kernel."""
        # HIP needs the hip_runtime header for device math functions
        body = super().generate()
        return "#include <hip/hip_runtime.h>\n\n" + body


def generate_hip_source(ir_func: ir.IRFunction) -> str:
    """Generate HIP C source for a single kernel function."""
    codegen = HIPCodeGen(ir_func)
    return codegen.generate()

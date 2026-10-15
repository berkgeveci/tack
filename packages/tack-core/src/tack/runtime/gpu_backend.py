"""What the GPU backends share: how a kernel call becomes a queued launch.

Metal, CUDA, HIP and Level Zero resolve a call's variant, evaluate its
launch range, bind textures, replace its scalar arguments with packed
buffers and queue the launch the same way; they differ in how a launch is
made, how its scalar buffers are owned, what it holds while it is made,
and whether it must complete before returning. `GPUBackend` is the shared
part, and each backend supplies those differences as the hooks below.

The CPU backend runs a range rather than queueing a launch, and does not
derive from it.
"""

import contextlib

from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    _get_launch,
    bind_textures,
    check_launch_size,
    resolve_variant,
)


class GPUBackend(Backend):
    """A backend that compiles a variant per call shape and queues launches.

    A subclass provides `_compile_kernel(packed_ir)` and `_launch(compiled,
    kernel_args, loop_end, extents)`, sets `_max_launch` (the most
    iterations one grid can index), and may override the hooks marked
    below.
    """

    #: Most iterations one launch can index (`check_launch_size`); set from
    #: the device at init.
    _max_launch: int

    def execute(self, kernel, args, kwargs):
        """Run ``kernel`` on ``args``: resolve the variant, then queue the launch.

        The IR passes and device compilation run only when this argument
        shape/type combination is new; see `resolve_variant`.
        """
        from tack.lang.array import Texture3D

        variant, effective_args = resolve_variant(
            self, kernel, args, kwargs,
            build=self._build_variant,
            store_texture_shapes=self._store_texture_shapes,
        )
        compiled, pack_info, pack_arrays = variant.payload

        # Loop range comes from the pre-packing IR: packing rewrites the
        # parameter list, and the range expression names the original args.
        kernel_args = [a.array if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end, extents = _get_launch(variant.ir, kernel_args)
        if loop_end <= 0:
            # range(0) runs nothing. No driver takes an empty grid, and
            # Level Zero's group count, a uint32, would wrap a negative one.
            return
        check_launch_size(f"Kernel '{kernel.name}'", loop_end, self._max_launch, self.label)

        # Textures bind their own snapshot, not the array they came from.
        kernel_args = bind_textures(effective_args)

        with self._launch_guard(variant):
            if pack_info:
                from tack.lang.ir_pack_scalars import split_args
                kept_args = split_args(effective_args, pack_info)
                kernel_args = bind_textures(kept_args) + self._scalar_buffers(
                    pack_info, pack_arrays, effective_args)
            self._launch(compiled, kernel_args, loop_end, extents)
        if self._must_complete(kernel_args):
            self.synchronize()

    def _build_variant(self, ir_func, effective_args):
        """Pack scalars, annotate, compile. Runs once per variant.

        Packing rewrites the parameter list, so it works on its own copy —
        the caller keeps `ir_func` for loop-range resolution.
        """
        from tack.lang.ir_pack_scalars import pack_scalars
        from tack.lang.ir_traversal import clone_ir
        from tack.lang.ir_type_annotate import annotate_types
        from tack.lang.ir_verify import verify_ir

        packed = clone_ir(ir_func)
        _, pack_info = pack_scalars(packed, effective_args)
        verify_ir(packed, 'packed')
        annotate_types(packed)
        verify_ir(packed, 'typed')
        compiled = self._compile_kernel(packed)
        return compiled, pack_info, self._variant_scalar_buffers(pack_info, effective_args)

    def _store_texture_shapes(self, ir_func, effective_args):
        """Record Texture3D extents, falling back to software sampling.

        A texture the device does not sample in hardware (no texture units,
        or past its 3D image limit; see `texture_in_hardware`) is sampled
        in software instead. That changes the generated code, so it is
        decided here, before the variant key is built, rather than at
        codegen time.
        """
        from tack.lang.array import Texture3D
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                if self.texture_in_hardware(arg.shape_3d):
                    param._texture_shape = arg.shape_3d
                else:
                    param._is_texture = False  # software fallback

    # ── Hooks ───────────────────────────────────────────────────────────

    def _compile_kernel(self, packed_ir):
        """The compiled kernel for ``packed_ir``: generate, compile, load."""
        raise NotImplementedError

    def _launch(self, compiled, kernel_args, loop_end, extents):
        """Queue one launch of ``compiled`` over ``loop_end`` iterations."""
        raise NotImplementedError

    def _variant_scalar_buffers(self, pack_info, effective_args):
        """Device buffers for the packed scalars that the variant keeps and
        every launch rewrites, or None. CUDA, HIP and Level Zero keep one
        set per variant, made from this first call's arguments."""
        if not pack_info:
            return None
        from tack.runtime.kernel_utils import _create_pack_arrays
        return _create_pack_arrays(pack_info, effective_args, self)

    def _scalar_buffers(self, pack_info, pack_arrays, effective_args) -> list:
        """This launch's packed-scalar buffers, written from its arguments.

        By default the variant's own (`_variant_scalar_buffers`), rewritten
        under `_launch_guard`, so no other launch can change them between
        the write and the launch that reads them."""
        from tack.runtime.kernel_utils import _update_pack_arrays
        _update_pack_arrays(pack_arrays, pack_info, effective_args)
        return pack_arrays

    def _launch_guard(self, variant):
        """Held from the scalar write through the launch. By default the
        variant's lock: its scalar buffers are shared by every call
        (unlocked, half of the CUDA dispatches from four threads computed
        with another thread's scalars, `test_concurrent_dispatch.py`)."""
        return variant.dispatch_lock

    def _must_complete(self, kernel_args) -> bool:
        """Whether this launch must finish before `execute` returns: it binds
        memory something outside Tack can read with no Tack call in between
        (a wrapped pointer, an exported array), or a texture image, which
        could be destroyed before a queued launch read it."""
        image = self._texture_image_type
        return any(getattr(getattr(a, '_buffer', None), '_shared', False)
                   or isinstance(a, image) for a in kernel_args)

    #: The class of this backend's hardware texture images (`_must_complete`).
    _texture_image_type: type = None   # replaced below, once the class exists


class _NoTextureImage:
    """Matches no argument: a backend without hardware texture images."""


GPUBackend._texture_image_type = _NoTextureImage

#: For a backend whose launches own their scalars and need no lock.
_no_guard = contextlib.nullcontext()

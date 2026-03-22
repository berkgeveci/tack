"""31 -- Zero-copy buffer interop between PGC and Dawn.

Demonstrates sharing GPU memory between PGC (compute) and Dawn
(WebGPU rendering engine used by VTK). PGC computes into a buffer,
then Dawn imports it via SharedBufferMemory -- no data copies.

Works on any backend: Metal (MTLBuffer sharing) or CUDA (Vulkan fd import).
The example code is backend-agnostic thanks to ExportedMemory.

This enables workflows where PGC runs GPU compute (filters, simulations)
and VTK renders the results, both operating on the same GPU memory.

Requirements:
  - pgc with Metal or CUDA backend
  - dawn-python (pip install dawn-python)

Usage:
  uv run python examples/31_dawn_interop.py
  uv run python examples/31_dawn_interop.py --arch cuda
"""

import numpy as np
import pgc

# ================================================================
# Step 1: PGC computes into a GPU buffer
# ================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="metal", choices=["metal", "cuda"])
args = parser.parse_args()
pgc.init(arch=args.arch)

N = 1024

x = pgc.field(dtype=pgc.f32, shape=(N,))
y = pgc.field(dtype=pgc.f32, shape=(N,))
out = pgc.field(dtype=pgc.f32, shape=(N,))

x.from_numpy(np.arange(N, dtype=np.float32))
y.from_numpy(np.ones(N, dtype=np.float32) * 2.0)


@pgc.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]


alpha = 3.0
saxpy(x, y, out, alpha, N)

pgc_result = out.to_numpy()
print(f"PGC computed SAXPY: first 5 = {pgc_result[:5]}")
print(f"  (expected: {alpha * np.arange(5, dtype=np.float32) + 2.0})")

# ================================================================
# Step 2: Export GPU memory (backend-agnostic)
# ================================================================

exported = out.export_memory()
buffer_size = exported.size

print(f"\nExported memory: backend={exported.backend}, "
      f"size={exported.size}, alloc_size={exported.allocation_size}")

# ================================================================
# Step 3: Import into Dawn (zero-copy)
# ================================================================

from pydawn import utils as dawn, webgpu

if exported.backend == "metal":
    features = [webgpu.WGPUFeatureName_SharedBufferMemoryMTLBuffer]
else:
    features = [webgpu.WGPUFeatureName_SharedBufferMemoryOpaqueFD]

adapter = dawn.request_adapter_sync(
    power_preference=webgpu.WGPUPowerPreference_HighPerformance)
device = dawn.request_device_sync(adapter, features)
print(f"\nDawn device created")

# Import using backend-appropriate path
if exported.backend == "metal":
    shared_mem = dawn.import_shared_buffer_memory_mtl(device, exported.handle)
else:
    shared_mem = dawn.import_shared_buffer_memory_opaque_fd(
        device, exported.handle, exported.allocation_size)
print(f"Imported SharedBufferMemory: {shared_mem}")

# Create a Dawn buffer from the shared memory
dawn_buf = dawn.create_buffer_from_shared_memory(
    device, shared_mem, buffer_size,
    webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc)

# Begin access (tells Dawn the buffer has valid data)
begin_desc = webgpu.WGPUSharedBufferMemoryBeginAccessDescriptor()
begin_desc.initialized = True
begin_desc.fenceCount = 0
webgpu.wgpuSharedBufferMemoryBeginAccess(shared_mem, dawn_buf, begin_desc)

# ================================================================
# Step 4: Verify Dawn sees the same data (zero-copy!)
# ================================================================

data = dawn.read_buffer(device, dawn_buf)
dawn_result = np.frombuffer(data, dtype=np.float32)

print(f"\nDawn reads: first 5 = {dawn_result[:5]}")
print(f"PGC wrote:  first 5 = {pgc_result[:5]}")

assert np.allclose(dawn_result, pgc_result), "Data mismatch!"
print(f"\nAll {N} values match -- zero-copy interop verified!")

# ================================================================
# Step 5: Dawn computes on the shared buffer, PGC reads back
# ================================================================

# Dawn runs a compute shader that doubles the values in-place
shader_src = """
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;

@compute @workgroup_size(256)
fn double_it(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params[0]) {
        data[i] = data[i] * 2.0;
    }
}
"""

shader = dawn.create_shader_module(device, shader_src)

# Build pipeline
layout_entries = [
    {"binding": 0, "visibility": webgpu.WGPUShaderStage_Compute,
     "buffer": {"type": webgpu.WGPUBufferBindingType_Storage}},
    {"binding": 1, "visibility": webgpu.WGPUShaderStage_Compute,
     "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage}},
]
bgl = dawn.create_bind_group_layout(device, layout_entries)
pl = dawn.create_pipeline_layout(device, [bgl])
pipeline = dawn.create_compute_pipeline(device, pl, {"module": shader, "entry_point": "double_it"})

# Params buffer
params_np = np.array([N], dtype=np.uint32)
params_buf = dawn.create_buffer(device, 4,
    webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst)
dawn.write_buffer(device, params_buf, 0, params_np.tobytes())

# Bind group — uses the shared buffer directly
bg = dawn.create_bind_group(device, bgl, [
    {"binding": 0, "resource": {"buffer": dawn_buf, "offset": 0, "size": buffer_size}},
    {"binding": 1, "resource": {"buffer": params_buf, "offset": 0, "size": 4}},
])

# Dispatch
encoder = dawn.create_command_encoder(device)
pass_enc = dawn.begin_compute_pass(encoder)
dawn.set_pipeline(pass_enc, pipeline)
dawn.set_bind_group(pass_enc, bg)
dawn.dispatch_workgroups(pass_enc, (N + 255) // 256, 1, 1)
dawn.end_compute_pass(pass_enc)
cmd = dawn.command_encoder_finish(encoder)
dawn.submit(device, [cmd])
dawn.sync(device)

# PGC reads the same buffer — should see doubled values
pgc_after = out.to_numpy()
expected = pgc_result * 2.0
print(f"\nAfter Dawn compute (double):")
print(f"  PGC reads: first 5 = {pgc_after[:5]}")
print(f"  Expected:  first 5 = {expected[:5]}")

assert np.allclose(pgc_after, expected), "Round-trip mismatch!"
print(f"\nBidirectional zero-copy interop verified! (backend={exported.backend})")
print("  PGC compute → shared buffer → Dawn compute → PGC reads back")

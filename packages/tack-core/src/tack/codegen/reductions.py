"""Shared f32 extrema semantics and native field-reduction kernel sources."""


def f32_reduction_helpers(dialect, operations=('min', 'max')):
    """NaN-propagating extrema with order-independent signed-zero ties."""
    qualifier = '__device__ inline' if dialect in ('cuda', 'hip') else 'inline'
    to_bits, to_float = {
        'cuda': ('__float_as_uint', '__uint_as_float'),
        'hip': ('__float_as_uint', '__uint_as_float'),
        'metal': ('as_type<uint>', 'as_type<float>'),
        'opencl': ('as_uint', 'as_float'),
    }[dialect]
    lines = []
    for op in sorted(operations):
        comparison, bits_op = ('<', '|') if op == 'min' else ('>', '&')
        lines.extend([
            f'{qualifier} float tack_reduce_{op}_f32(float a, float b) {{',
            f'    if (a != a || b != b) return {to_float}(0x7fc00000u);',
            '    if (a == 0.0f && b == 0.0f)',
            f'        return {to_float}({to_bits}(a) {bits_op} {to_bits}(b));',
            f'    return a {comparison} b ? a : b;',
            '}', '',
        ])
    return lines


def field_reduction_source(dialect, op):
    """A 256-lane tree followed by unordered atomic partial accumulation."""
    if op not in ('sum', 'min', 'max'):
        raise ValueError(f'Unknown reduction: {op}')
    to_bits, to_float = {
        'cuda': ('__float_as_uint', '__uint_as_float'),
        'hip': ('__float_as_uint', '__uint_as_float'),
        'metal': ('as_type<uint>', 'as_type<float>'),
        'opencl': ('as_uint', 'as_float'),
    }[dialect]
    identity = {'sum': '0.0f', 'min': f'{to_float}(0x7f800000u)',
                'max': f'{to_float}(0xff800000u)'}[op]
    tree = 'sdata[tid] + sdata[tid + s]' if op == 'sum' else (
        f'tack_reduce_{op}_f32(sdata[tid], sdata[tid + s])')
    final = ('sdata[0] + old_f' if op == 'sum' else f'tack_reduce_{op}_f32(sdata[0], old_f)')
    if dialect in ('cuda', 'hip'):
        header = '#include <hip/hip_runtime.h>\n' if dialect == 'hip' else ''
        signature = (f'extern "C" __global__ void reduce_{op}_f32('
                     'float* input, float* output, long long n)')
        # Widen before multiplying: the 32-bit product wraps past 2^32 threads.
        setup = '''extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + tid;'''
        barrier = '__syncthreads();'
        atomic = 'atomicAdd(&output[0], sdata[0]);' if op == 'sum' else f'''
        unsigned int* addr = (unsigned int*)&output[0];
        unsigned int old = atomicCAS(addr, 0u, 0u), assumed;
        do {{
            assumed = old;
            float old_f = {to_float}(assumed);
            old = atomicCAS(addr, assumed, {to_bits}({final}));
        }} while (assumed != old);'''
    elif dialect == 'metal':
        header = '#include <metal_stdlib>\nusing namespace metal;\n'
        signature = f'''kernel void reduce_{op}_f32(
    device float* input [[buffer(0)]], device float* output [[buffer(1)]],
    uint i [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])'''
        setup = 'threadgroup float sdata[256];\n    uint n = as_type<uint>(output[1]);'
        barrier = 'threadgroup_barrier(mem_flags::mem_threadgroup);'
        atomic = '''atomic_fetch_add_explicit(
            (volatile device atomic_float*)&output[0], sdata[0], memory_order_relaxed);''' if op == 'sum' else f'''
        volatile device atomic_uint* addr = (volatile device atomic_uint*)&output[0];
        uint old = atomic_load_explicit(addr, memory_order_relaxed);
        while (true) {{
            float old_f = {to_float}(old);
            uint next = {to_bits}({final});
            if (atomic_compare_exchange_weak_explicit(addr, &old, next,
                memory_order_relaxed, memory_order_relaxed)) break;
        }}'''
    else:
        header = ''
        signature = f'''__kernel void reduce_{op}_f32(
    __global float* input, __global float* output, long n)'''
        setup = ('__local float sdata[256];\n    uint tid = get_local_id(0);\n'
                 '    long i = (long)get_group_id(0) * (long)get_local_size(0) + tid;')
        barrier = 'barrier(CLK_LOCAL_MEM_FENCE);'
        atomic = f'''
        volatile __global atomic_uint* addr = (volatile __global atomic_uint*)&output[0];
        uint old = atomic_load_explicit(addr, memory_order_relaxed, memory_scope_device);
        while (true) {{
            float old_f = {to_float}(old);
            uint next = {to_bits}({final});
            if (atomic_compare_exchange_weak_explicit(addr, &old, next,
                memory_order_relaxed, memory_order_relaxed, memory_scope_device)) break;
        }}'''
    helpers = '\n'.join(f32_reduction_helpers(dialect, (op,))) if op != 'sum' else ''
    return header + helpers + f'''
{signature} {{
    {setup}
    sdata[tid] = (i < n) ? input[i] : {identity};
    {barrier}
    for (unsigned int s = 128; s > 0; s >>= 1) {{
        if (tid < s) sdata[tid] = {tree};
        {barrier}
    }}
    if (tid == 0) {{ {atomic}
    }}
}}
'''

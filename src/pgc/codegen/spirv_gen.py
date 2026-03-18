"""PGC SPIR-V code generation — emits SPIR-V compute shaders from PGC IR.

Generates SPIR-V 1.0 compute shaders with:
  - Storage buffers for field parameters (binding 0, 1, 2, ...)
  - gl_GlobalInvocationID for the parallel loop variable
  - Push constants or specialization constants for loop bounds
  - GLSL.std.450 extended instructions for math builtins

The generated shader structure:
  - Each field becomes a storage buffer (OpTypeRuntimeArray)
  - The parallel for-loop index maps to gl_GlobalInvocationID.x
  - Nested sequential loops become regular OpLoopMerge loops
"""

import struct
from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64

# SPIR-V magic number and version
SPIRV_MAGIC = 0x07230203
SPIRV_VERSION = 0x00010300  # 1.3 (needed for StorageBuffer storage class)
SPIRV_GENERATOR = 0x00000000  # unregistered

# SPIR-V opcodes (subset needed for compute shaders)
OpNop = 0
OpExtInstImport = 11
OpExtInst = 12
OpMemoryModel = 14
OpEntryPoint = 15
OpExecutionMode = 16
OpCapability = 17
OpTypeVoid = 19
OpTypeBool = 20
OpTypeInt = 21
OpTypeFloat = 22
OpTypeVector = 23
OpTypeArray = 28
OpTypeRuntimeArray = 29
OpTypeStruct = 30
OpTypePointer = 32
OpTypeFunction = 33
OpConstant = 43
OpConstantTrue = 41
OpConstantFalse = 42
OpConstantComposite = 44
OpFunction = 54
OpFunctionParameter = 55
OpFunctionEnd = 56
OpVariable = 59
OpLoad = 61
OpStore = 62
OpAccessChain = 65
OpDecorate = 71
OpMemberDecorate = 72
OpDecorationGroup = 73
OpCompositeConstruct = 80
OpCompositeExtract = 81
OpIAdd = 128
OpFAdd = 129
OpISub = 130
OpFSub = 131
OpIMul = 132
OpFMul = 133
OpUDiv = 134
OpSDiv = 135
OpFDiv = 136
OpUMod = 137
OpSRem = 138
OpSMod = 139
OpFRem = 140
OpFMod = 141
OpLogicalEqual = 164
OpLogicalNotEqual = 165
OpLogicalOr = 166
OpLogicalAnd = 167
OpLogicalNot = 168
OpSelect = 169
OpIEqual = 170
OpINotEqual = 171
OpUGreaterThan = 172
OpSGreaterThan = 173
OpUGreaterThanEqual = 174
OpSGreaterThanEqual = 175
OpULessThan = 176
OpSLessThan = 177
OpULessThanEqual = 178
OpSLessThanEqual = 179
OpFOrdEqual = 180
OpFOrdNotEqual = 182
OpFOrdLessThan = 184
OpFOrdGreaterThan = 186
OpFOrdLessThanEqual = 188
OpFOrdGreaterThanEqual = 190
OpShiftRightLogical = 194
OpShiftRightArithmetic = 195
OpShiftLeftLogical = 196
OpBitwiseOr = 197
OpBitwiseXor = 198
OpBitwiseAnd = 199
OpNot = 200
OpConvertFToU = 109
OpConvertFToU = 109
OpConvertFToS = 110
OpConvertSToF = 111
OpConvertUToF = 112
OpBitcast = 124
OpConvertUToF = 112
OpFConvert = 115
OpFNegate = 127
OpSNegate = 126
OpLabel = 248
OpBranch = 249
OpBranchConditional = 250
OpSwitch = 251
OpReturn = 253
OpReturnValue = 254
OpLoopMerge = 246
OpSelectionMerge = 247
OpPhi = 245

# Atomic operations
OpAtomicLoad = 227
OpAtomicStore = 228
OpAtomicExchange = 229
OpAtomicCompareExchange = 230
OpAtomicIAdd = 234
OpAtomicISub = 235
OpAtomicSMin = 236
OpAtomicUMin = 237
OpAtomicSMax = 238
OpAtomicUMax = 239

# Image/sampler opcodes
OpTypeImage = 25
OpTypeSampler = 26
OpTypeSampledImage = 27
OpSampledImage = 86
OpImageSampleExplicitLod = 88

# Image dimension
Dim_3D = 2

# Image operand mask
ImageOperand_Lod = 0x2

# Barrier
OpControlBarrier = 224

# Memory semantics
Scope_Device = 1
Scope_Workgroup = 2
MemorySemantics_None = 0x0
MemorySemantics_AcquireRelease = 0x8
MemorySemantics_WorkgroupMemory = 0x100

# Execution models
ExecutionModel_GLCompute = 5

# Addressing / Memory models
AddressingModel_Logical = 0
MemoryModel_GLSL450 = 1

# Storage classes
StorageClass_UniformConstant = 0
StorageClass_Input = 1
StorageClass_Uniform = 2
StorageClass_Workgroup = 4
StorageClass_Function = 7
StorageClass_PushConstant = 9
StorageClass_StorageBuffer = 12

# Decorations
Decoration_Block = 2
Decoration_BufferBlock = 3
Decoration_ArrayStride = 6
Decoration_Offset = 35
Decoration_Binding = 33
Decoration_DescriptorSet = 34
Decoration_BuiltIn = 11
Decoration_NonWritable = 24
Decoration_NonReadable = 25

# Built-in variables
BuiltIn_GlobalInvocationId = 28
BuiltIn_NumWorkgroups = 24
BuiltIn_WorkgroupId = 26
BuiltIn_LocalInvocationId = 27

# Execution modes
ExecutionMode_LocalSize = 17

# Capabilities
Capability_Shader = 1
Capability_Float64 = 10
Capability_Int64 = 11

# Selection/Loop control
SelectionControl_None = 0
LoopControl_None = 0

# Memory access
MemoryAccess_None = 0

# GLSL.std.450 extended instruction numbers
GLSL_Sqrt = 31
GLSL_Sin = 13
GLSL_Cos = 14
GLSL_Tan = 15
GLSL_Asin = 16
GLSL_Acos = 17
GLSL_Atan = 18
GLSL_Atan2 = 25
GLSL_Exp = 27
GLSL_Log = 28
GLSL_Exp2 = 29
GLSL_Log2 = 30
GLSL_Pow = 26
GLSL_Floor = 8
GLSL_Ceil = 9
GLSL_FAbs = 4
GLSL_FMin = 37
GLSL_FMax = 40
GLSL_FSign = 6
GLSL_FClamp = 43
GLSL_Log10 = 0  # Not in GLSL.std.450; we synthesize log10(x) = log(x) / log(10)

_GLSL_FUNC_MAP = {
    "sqrt": GLSL_Sqrt, "sin": GLSL_Sin, "cos": GLSL_Cos, "tan": GLSL_Tan,
    "asin": GLSL_Asin, "acos": GLSL_Acos, "atan": GLSL_Atan, "atan2": GLSL_Atan2,
    "exp": GLSL_Exp, "log": GLSL_Log, "exp2": GLSL_Exp2, "log2": GLSL_Log2,
    "pow": GLSL_Pow, "floor": GLSL_Floor, "ceil": GLSL_Ceil,
    "abs": GLSL_FAbs, "fabs": GLSL_FAbs, "min": GLSL_FMin, "max": GLSL_FMax,
    "log10": "synthesized",  # handled specially in _emit_call
}


def _encode_string(s: str) -> list[int]:
    """Encode a string as SPIR-V words (null-terminated, padded to word boundary)."""
    b = s.encode("utf-8") + b"\x00"
    # Pad to 4-byte boundary
    while len(b) % 4 != 0:
        b += b"\x00"
    return [int.from_bytes(b[i:i+4], "little") for i in range(0, len(b), 4)]


def _make_instruction(opcode: int, *operands: int) -> list[int]:
    """Build a SPIR-V instruction word list."""
    word_count = 1 + len(operands)
    return [(word_count << 16) | opcode] + list(operands)


def _make_instruction_with_string(opcode: int, operands_before: list[int],
                                   string: str, operands_after: list[int] = None) -> list[int]:
    """Build a SPIR-V instruction with an embedded string literal."""
    str_words = _encode_string(string)
    all_operands = list(operands_before) + str_words + (operands_after or [])
    word_count = 1 + len(all_operands)
    return [(word_count << 16) | opcode] + all_operands


class SPIRVModule:
    """Builds a SPIR-V binary module incrementally."""

    def __init__(self):
        self._bound = 1  # next available result ID
        self._capabilities = []
        self._extensions = []
        self._ext_inst_imports = []
        self._memory_model = []
        self._entry_points = []
        self._execution_modes = []
        self._annotations = []  # decorations
        self._types_constants = []  # types, constants, global variables
        self._functions = []  # function definitions

    def alloc_id(self) -> int:
        """Allocate and return the next result ID."""
        result = self._bound
        self._bound += 1
        return result

    def add_capability(self, cap: int):
        self._capabilities += _make_instruction(OpCapability, cap)

    def add_ext_inst_import(self, result_id: int, name: str):
        self._ext_inst_imports += _make_instruction_with_string(
            OpExtInstImport, [result_id], name)

    def set_memory_model(self, addressing: int, memory: int):
        self._memory_model = _make_instruction(OpMemoryModel, addressing, memory)

    def add_entry_point(self, execution_model: int, func_id: int, name: str,
                        interface_ids: list[int]):
        self._entry_points += _make_instruction_with_string(
            OpEntryPoint, [execution_model, func_id], name, interface_ids)

    def add_execution_mode(self, func_id: int, mode: int, *operands: int):
        self._execution_modes += _make_instruction(OpExecutionMode, func_id, mode, *operands)

    def add_annotation(self, words: list[int]):
        self._annotations += words

    def add_type_or_constant(self, words: list[int]):
        self._types_constants += words

    def add_function_words(self, words: list[int]):
        self._functions += words

    def encode(self) -> bytes:
        """Encode the module to SPIR-V binary."""
        all_words = (
            [SPIRV_MAGIC, SPIRV_VERSION, SPIRV_GENERATOR, self._bound, 0]
            + self._capabilities
            + self._extensions
            + self._ext_inst_imports
            + self._memory_model
            + self._entry_points
            + self._execution_modes
            + self._annotations
            + self._types_constants
            + self._functions
        )
        return struct.pack(f"<{len(all_words)}I", *all_words)


class SPIRVCodeGen:
    """Generates a SPIR-V compute shader from a PGC IR function.

    Each field parameter becomes a storage buffer at binding N.
    The parallel for-loop index comes from gl_GlobalInvocationID.x.
    """

    def __init__(self, ir_func: ir.IRFunction, workgroup_size: int = 256):
        self.ir_func = ir_func
        self.workgroup_size = workgroup_size
        self.module = SPIRVModule()

        # Type cache: maps type descriptions to SPIR-V IDs
        self._type_cache: dict[str, int] = {}
        # Constant cache
        self._const_cache: dict[tuple, int] = {}
        # Variable name → SPIR-V ID
        self._vars: dict[str, int] = {}
        # Parameter name → (buffer pointer ID, element type key)
        self._param_buffers: dict[str, tuple[int, str]] = {}
        # Local variable name → (pointer ID, type key)
        self._local_vars: dict[str, tuple[int, str]] = {}
        # Function body instructions
        self._body: list[int] = []
        # Deferred OpVariable instructions (must be in entry block)
        self._func_vars: list[int] = []

        # Break/continue targets
        self._break_label: int | None = None
        self._continue_label: int | None = None

        # SPIR-V ID → type key (e.g. "f32", "u32", "i32", "bool")
        self._id_types: dict[int, str] = {}

        # Texture parameters (combined image/sampler instead of storage buffer)
        self._texture_params: set[str] = {}
        # Texture sampled-image variable IDs (param_name → var_id)
        self._texture_vars: dict[str, int] = {}

        # IDs for key types/variables
        self._glsl_ext_id = 0
        self._global_invocation_id_var = 0
        self._global_invocation_id_type = 0

    def generate(self) -> bytes:
        """Generate the SPIR-V binary."""
        # Preamble
        self.module.add_capability(Capability_Shader)
        self._glsl_ext_id = self.module.alloc_id()
        self.module.add_ext_inst_import(self._glsl_ext_id, "GLSL.std.450")
        self.module.set_memory_model(AddressingModel_Logical, MemoryModel_GLSL450)

        # Declare types we'll need
        self._declare_base_types()

        # Detect texture parameters
        self._texture_params = set()
        for param in self.ir_func.params:
            if getattr(param, '_is_texture', False):
                self._texture_params.add(param.name)

        # Declare storage buffers (or combined image/samplers) for each parameter
        for i, param in enumerate(self.ir_func.params):
            if param.name in self._texture_params:
                tex_var = self._declare_texture_sampler(param, binding=i)
                # Store as param buffer with special key so scalar pre-load skips it
                self._param_buffers[param.name] = (tex_var, "sampled_image")
                self._texture_vars[param.name] = tex_var
            else:
                buf_var = self._declare_storage_buffer(param, binding=i)
                self._param_buffers[param.name] = buf_var

        # Declare gl_GlobalInvocationID
        self._declare_global_invocation_id()

        # In SPIR-V 1.3 and earlier, only Input/Output variables go in the
        # OpEntryPoint interface list.  Storage buffer variables are NOT allowed.
        interface_vars = [self._global_invocation_id_var]

        # Pre-scan: if the kernel uses thread_id/shared memory, declare LocalInvocationID
        if self._ir_uses_threadgroup(self.ir_func.body):
            ptr_type = self._get_type("ptr_uvec3_input")
            var_id = self.module.alloc_id()
            self.module.add_type_or_constant(
                _make_instruction(OpVariable, ptr_type, var_id, StorageClass_Input))
            self.module.add_annotation(
                _make_instruction(OpDecorate, var_id, Decoration_BuiltIn,
                                  BuiltIn_LocalInvocationId))
            self._local_invocation_id_var = var_id
            interface_vars.append(var_id)

        # Declare the main function
        void_type = self._get_type("void")
        func_type = self._declare_function_type(void_type, [])
        func_id = self.module.alloc_id()

        # Entry point
        self.module.add_entry_point(
            ExecutionModel_GLCompute, func_id, "main", interface_vars)
        self.module.add_execution_mode(
            func_id, ExecutionMode_LocalSize,
            self.workgroup_size, 1, 1)

        # Function begin — OpFunction + OpLabel go in preamble,
        # then deferred OpVariable instructions, then the rest of the body.
        preamble = _make_instruction(OpFunction, void_type, func_id, 0, func_type)
        entry_label = self.module.alloc_id()
        preamble += _make_instruction(OpLabel, entry_label)

        # Pre-load scalar parameters from their storage buffers at index 0.
        # The Vulkan backend wraps scalar args in 1-element storage buffers,
        # so we load them once at the start of the kernel.
        for param in self.ir_func.params:
            if param.name in self._texture_params:
                continue  # texture params are loaded at sample time
            if hasattr(param, '_is_field') and not param._is_field:
                buf_var, elem_key = self._param_buffers[param.name]
                elem_ptr_key = f"ptr_sb_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                elem_type = self._get_type(elem_key)
                zero = self._const_u32(0)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, zero, zero)
                val_id = self.module.alloc_id()
                self._body += _make_instruction(OpLoad, elem_type, val_id, ac)
                self._id_types[val_id] = elem_key
                self._vars[param.name] = val_id

        # Load gl_GlobalInvocationID.x as the loop index
        uvec3_type = self._get_type("uvec3")
        u32_type = self._get_type("u32")
        gid_load = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, uvec3_type, gid_load,
                                         self._global_invocation_id_var)
        gid_x = self.module.alloc_id()
        self._body += _make_instruction(OpCompositeExtract, u32_type, gid_x, gid_load, 0)
        self._id_types[gid_x] = "u32"

        # Bounds guard: load n from push constant, return if gid_x >= n
        pc_n_ptr, pc_n_id = self._declare_push_constant_n()
        n_val = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, u32_type, n_val, pc_n_id)
        cmp_id = self.module.alloc_id()
        self._body += _make_instruction(OpUGreaterThanEqual, self._get_type("bool"),
                                         cmp_id, gid_x, n_val)
        merge_label = self.module.alloc_id()
        early_ret_label = self.module.alloc_id()
        self._body += _make_instruction(OpSelectionMerge, merge_label, SelectionControl_None)
        self._body += _make_instruction(OpBranchConditional, cmp_id, early_ret_label, merge_label)
        self._body += _make_instruction(OpLabel, early_ret_label)
        self._body += _make_instruction(OpReturn)
        self._body += _make_instruction(OpLabel, merge_label)

        # Emit any statements before the parallel for (e.g., pre-loop setup)
        parallel_for = self._find_parallel_for()
        if parallel_for:
            for stmt in self.ir_func.body:
                if stmt is parallel_for:
                    break
                self._emit_stmt(stmt)
            self._vars[parallel_for.var] = gid_x
            # Emit the body of the parallel for (the loop itself is the dispatch)
            self._emit_body(parallel_for.body)

        # Function end
        if not self._last_is_terminator():
            self._body += _make_instruction(OpReturn)
        self._body += _make_instruction(OpFunctionEnd)

        # Assemble: preamble + hoisted OpVariables + body
        self.module.add_function_words(preamble + self._func_vars + self._body)
        return self.module.encode()

    def _find_parallel_for(self) -> ir.IRParallelFor | None:
        for stmt in self.ir_func.body:
            if isinstance(stmt, ir.IRParallelFor):
                return stmt
        return None

    def _ir_uses_threadgroup(self, stmts: list) -> bool:
        """Check if any statement uses shared memory or thread_id."""
        for stmt in stmts:
            if isinstance(stmt, (ir.IRSharedAlloc, ir.IRBarrier, ir.IRThreadId)):
                return True
            for child_list in self._stmt_children(stmt):
                if self._ir_uses_threadgroup(child_list):
                    return True
        return False

    def _stmt_children(self, stmt) -> list[list]:
        """Return child statement lists of a statement."""
        if isinstance(stmt, ir.IRParallelFor):
            return [stmt.body]
        if isinstance(stmt, ir.IRSequentialFor):
            return [stmt.body]
        if isinstance(stmt, ir.IRWhile):
            return [stmt.body]
        if isinstance(stmt, ir.IRIf):
            result = [stmt.then_body]
            if stmt.else_body:
                result.append(stmt.else_body)
            return result
        return []

    def _last_is_terminator(self) -> bool:
        """Check if the last instruction emitted is a terminator."""
        if not self._body:
            return False
        last_opcode = self._body[-1] & 0xFFFF if self._body else -1
        # Check the most recent instruction's opcode
        # Walk backwards to find the last instruction start
        for i in range(len(self._body) - 1, -1, -1):
            word = self._body[i]
            wc = (word >> 16) & 0xFFFF
            op = word & 0xFFFF
            if wc > 0:
                return op in (OpReturn, OpReturnValue, OpBranch,
                              OpBranchConditional, OpSwitch)
        return False

    # --- Type declarations ---

    def _declare_base_types(self):
        """Declare commonly used SPIR-V types."""
        # Void
        vid = self.module.alloc_id()
        self._type_cache["void"] = vid
        self.module.add_type_or_constant(_make_instruction(OpTypeVoid, vid))

        # Bool
        bid = self.module.alloc_id()
        self._type_cache["bool"] = bid
        self.module.add_type_or_constant(_make_instruction(OpTypeBool, bid))

        # Unsigned int 32
        uid = self.module.alloc_id()
        self._type_cache["u32"] = uid
        self.module.add_type_or_constant(_make_instruction(OpTypeInt, uid, 32, 0))

        # Signed int 32
        sid = self.module.alloc_id()
        self._type_cache["i32"] = sid
        self.module.add_type_or_constant(_make_instruction(OpTypeInt, sid, 32, 1))

        # Float 32
        fid = self.module.alloc_id()
        self._type_cache["f32"] = fid
        self.module.add_type_or_constant(_make_instruction(OpTypeFloat, fid, 32))

        # Declare wider types only if the kernel uses them
        used_types = {p.type_annotation for p in self.ir_func.params
                      if p.type_annotation is not None}
        if f64 in used_types:
            self.module.add_capability(Capability_Float64)
            did = self.module.alloc_id()
            self._type_cache["f64"] = did
            self.module.add_type_or_constant(_make_instruction(OpTypeFloat, did, 64))
        if i64 in used_types:
            lid = self.module.alloc_id()
            self._type_cache["i64"] = lid
            self.module.add_type_or_constant(_make_instruction(OpTypeInt, lid, 64, 1))
        if u64 in used_types:
            ulid = self.module.alloc_id()
            self._type_cache["u64"] = ulid
            self.module.add_type_or_constant(_make_instruction(OpTypeInt, ulid, 64, 0))

        # uvec3 (for gl_GlobalInvocationID)
        uvec3 = self.module.alloc_id()
        self._type_cache["uvec3"] = uvec3
        self.module.add_type_or_constant(
            _make_instruction(OpTypeVector, uvec3, uid, 3))

        # vec3 (float3) and vec4 (float4) for texture sampling
        vec3_f32 = self.module.alloc_id()
        self._type_cache["vec3_f32"] = vec3_f32
        self.module.add_type_or_constant(
            _make_instruction(OpTypeVector, vec3_f32, fid, 3))

        vec4_f32 = self.module.alloc_id()
        self._type_cache["vec4_f32"] = vec4_f32
        self.module.add_type_or_constant(
            _make_instruction(OpTypeVector, vec4_f32, fid, 4))

        # Pointer to uvec3 (Input)
        ptr_uvec3_input = self.module.alloc_id()
        self._type_cache["ptr_uvec3_input"] = ptr_uvec3_input
        self.module.add_type_or_constant(
            _make_instruction(OpTypePointer, ptr_uvec3_input,
                              StorageClass_Input, uvec3))

    def _get_type(self, key: str) -> int:
        """Get or create a SPIR-V type ID."""
        if key in self._type_cache:
            return self._type_cache[key]
        raise KeyError(f"Type '{key}' not declared")

    def _get_or_create_type(self, key: str, creator) -> int:
        if key not in self._type_cache:
            self._type_cache[key] = creator()
        return self._type_cache[key]

    def _pgc_type_to_spirv_key(self, pgc_type: ScalarType) -> str:
        _MAP = {f32: "f32", f64: "f64", i32: "i32", i64: "i64", u32: "u32", u64: "u64"}
        key = _MAP.get(pgc_type)
        if key is None:
            raise TypeError(f"Unsupported PGC type for SPIR-V: {pgc_type}")
        return key

    def _declare_function_type(self, return_type: int, param_types: list[int]) -> int:
        key = f"functype_{return_type}_{'_'.join(str(p) for p in param_types)}"
        if key in self._type_cache:
            return self._type_cache[key]
        fid = self.module.alloc_id()
        self._type_cache[key] = fid
        self.module.add_type_or_constant(
            _make_instruction(OpTypeFunction, fid, return_type, *param_types))
        return fid

    def _declare_storage_buffer(self, param: ir.IRParam, binding: int) -> tuple[int, str]:
        """Declare a storage buffer for a field parameter.

        Returns (variable_id, element_type_key).
        """
        pgc_type = param.type_annotation
        elem_key = self._pgc_type_to_spirv_key(pgc_type)
        elem_type = self._get_type(elem_key)

        # RuntimeArray of element type
        ra_key = f"runtime_array_{elem_key}"
        ra_id = self._get_or_create_type(ra_key, lambda: self._make_runtime_array(elem_type, elem_key))

        # Struct wrapping the runtime array
        struct_key = f"struct_buf_{binding}"
        struct_id = self.module.alloc_id()
        self._type_cache[struct_key] = struct_id
        self.module.add_type_or_constant(
            _make_instruction(OpTypeStruct, struct_id, ra_id))

        # Decorate struct as Block, member 0 offset 0
        self.module.add_annotation(
            _make_instruction(OpDecorate, struct_id, Decoration_Block))
        self.module.add_annotation(
            _make_instruction(OpMemberDecorate, struct_id, 0, Decoration_Offset, 0))

        # Pointer to struct (StorageBuffer)
        ptr_key = f"ptr_buf_{binding}"
        ptr_id = self.module.alloc_id()
        self._type_cache[ptr_key] = ptr_id
        self.module.add_type_or_constant(
            _make_instruction(OpTypePointer, ptr_id, StorageClass_StorageBuffer, struct_id))

        # Variable
        var_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, ptr_id, var_id, StorageClass_StorageBuffer))

        # Decorate with binding and descriptor set
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_DescriptorSet, 0))
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_Binding, binding))

        # Pointer to element type (StorageBuffer) — for AccessChain
        elem_ptr_key = f"ptr_sb_{elem_key}"
        self._get_or_create_type(elem_ptr_key, lambda: self._make_pointer_type(
            StorageClass_StorageBuffer, elem_type, elem_ptr_key))

        return (var_id, elem_key)

    def _declare_texture_sampler(self, param: ir.IRParam, binding: int) -> int:
        """Declare a combined image/sampler for a texture parameter.

        Returns the variable ID for the sampled-image uniform constant.
        """
        f32_type = self._get_type("f32")

        # OpTypeImage %f32 3D 0 0 0 1 Unknown
        #   Dim=2(3D), Depth=0, Arrayed=0, MS=0, Sampled=1, Format=0(Unknown)
        image_key = "image3d_f32"
        if image_key not in self._type_cache:
            img_id = self.module.alloc_id()
            self._type_cache[image_key] = img_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypeImage, img_id, f32_type,
                                  Dim_3D, 0, 0, 0, 1, 0))

        # OpTypeSampledImage %image_type
        sampled_key = "sampled_image3d_f32"
        if sampled_key not in self._type_cache:
            si_id = self.module.alloc_id()
            self._type_cache[sampled_key] = si_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypeSampledImage, si_id,
                                  self._type_cache[image_key]))

        # Pointer to sampled image in UniformConstant storage class
        ptr_key = "ptr_uc_sampled_image3d_f32"
        if ptr_key not in self._type_cache:
            ptr_id = self.module.alloc_id()
            self._type_cache[ptr_key] = ptr_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, ptr_id,
                                  StorageClass_UniformConstant,
                                  self._type_cache[sampled_key]))

        # Variable
        var_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, self._type_cache[ptr_key], var_id,
                              StorageClass_UniformConstant))

        # Decorate with binding and descriptor set
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_DescriptorSet, 0))
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_Binding, binding))

        return var_id

    def _make_runtime_array(self, elem_type: int, elem_key: str) -> int:
        ra_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpTypeRuntimeArray, ra_id, elem_type))
        # ArrayStride decoration
        stride = 8 if elem_key in ("f64", "i64", "u64") else 4
        self.module.add_annotation(
            _make_instruction(OpDecorate, ra_id, Decoration_ArrayStride, stride))
        return ra_id

    def _make_pointer_type(self, storage_class: int, pointee: int, key: str) -> int:
        pid = self.module.alloc_id()
        self._type_cache[key] = pid
        self.module.add_type_or_constant(
            _make_instruction(OpTypePointer, pid, storage_class, pointee))
        return pid

    def _declare_global_invocation_id(self):
        """Declare the gl_GlobalInvocationID built-in variable."""
        ptr_type = self._get_type("ptr_uvec3_input")
        var_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, ptr_type, var_id, StorageClass_Input))
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_BuiltIn,
                              BuiltIn_GlobalInvocationId))
        self._global_invocation_id_var = var_id

    def _declare_push_constant_n(self) -> tuple[int, int]:
        """Declare a push constant block containing the loop bound n (uint32).

        Returns (struct_var_id, access_chain_id_for_n).
        The access chain is deferred to body emission; here we declare the type
        and variable, and emit the AccessChain in the body instructions.
        """
        u32_type = self._get_type("u32")

        # Struct { uint n; }
        pc_struct = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpTypeStruct, pc_struct, u32_type))
        self.module.add_annotation(
            _make_instruction(OpDecorate, pc_struct, Decoration_Block))
        self.module.add_annotation(
            _make_instruction(OpMemberDecorate, pc_struct, 0, Decoration_Offset, 0))

        # Pointer to struct in PushConstant storage class
        ptr_pc = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpTypePointer, ptr_pc, StorageClass_PushConstant, pc_struct))

        # Pointer to uint in PushConstant storage class (for AccessChain result)
        ptr_u32_pc_key = "ptr_pc_u32"
        if ptr_u32_pc_key not in self._type_cache:
            ptr_u32_pc = self.module.alloc_id()
            self._type_cache[ptr_u32_pc_key] = ptr_u32_pc
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, ptr_u32_pc,
                                  StorageClass_PushConstant, u32_type))

        # Variable
        pc_var = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, ptr_pc, pc_var, StorageClass_PushConstant))

        # AccessChain to member 0 (n)
        idx_0 = self._const_i32(0)
        ptr_u32_pc = self._type_cache[ptr_u32_pc_key]
        ac_id = self.module.alloc_id()
        self._body += _make_instruction(OpAccessChain, ptr_u32_pc, ac_id, pc_var, idx_0)

        return pc_var, ac_id

    # --- Constants ---

    def _const_u32(self, value: int) -> int:
        key = ("u32", value)
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.module.alloc_id()
        self._const_cache[key] = cid
        u32_type = self._get_type("u32")
        self.module.add_type_or_constant(
            _make_instruction(OpConstant, u32_type, cid, value & 0xFFFFFFFF))
        self._id_types[cid] = "u32"
        return cid

    def _const_i32(self, value: int) -> int:
        key = ("i32", value)
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.module.alloc_id()
        self._const_cache[key] = cid
        i32_type = self._get_type("i32")
        self.module.add_type_or_constant(
            _make_instruction(OpConstant, i32_type, cid,
                              struct.unpack("<I", struct.pack("<i", value))[0]))
        self._id_types[cid] = "i32"
        return cid

    def _const_f32(self, value: float) -> int:
        # Use exact bit pattern as key to handle -0.0 vs 0.0
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        key = ("f32", bits)
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.module.alloc_id()
        self._const_cache[key] = cid
        f32_type = self._get_type("f32")
        self.module.add_type_or_constant(
            _make_instruction(OpConstant, f32_type, cid, bits))
        self._id_types[cid] = "f32"
        return cid

    def _const_f64(self, value: float) -> int:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        key = ("f64", bits)
        if key in self._const_cache:
            return self._const_cache[key]
        cid = self.module.alloc_id()
        self._const_cache[key] = cid
        f64_type = self._get_type("f64")
        lo = bits & 0xFFFFFFFF
        hi = (bits >> 32) & 0xFFFFFFFF
        self.module.add_type_or_constant(
            _make_instruction(OpConstant, f64_type, cid, lo, hi))
        self._id_types[cid] = "f64"
        return cid

    # --- Code emission ---

    def _emit_body(self, stmts: list):
        for stmt in stmts:
            self._emit_stmt(stmt)

    def _emit_stmt(self, node: ir.IRNode):
        if isinstance(node, ir.IRFieldStore):
            self._emit_field_store(node)
        elif isinstance(node, ir.IRAssign):
            self._emit_assign(node)
        elif isinstance(node, ir.IRIf):
            self._emit_if(node)
        elif isinstance(node, ir.IRSequentialFor):
            self._emit_sequential_for(node)
        elif isinstance(node, ir.IRWhile):
            self._emit_while(node)
        elif isinstance(node, ir.IRBreak):
            self._emit_break()
        elif isinstance(node, ir.IRContinue):
            self._emit_continue()
        elif isinstance(node, ir.IRReturn):
            self._body += _make_instruction(OpReturn)
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRSharedAlloc):
            self._emit_shared_alloc(node)
        elif isinstance(node, ir.IRLocalAlloc):
            self._emit_local_alloc(node)
        elif isinstance(node, ir.IRBarrier):
            self._emit_barrier()
        elif isinstance(node, ir.IRCall):
            self._emit_expr(node)
        elif isinstance(node, ir.IRPrint):
            pass  # print not supported in SPIR-V
        else:
            raise NotImplementedError(f"SPIR-V stmt: {type(node).__name__}")

    def _emit_expr(self, node: ir.IRNode) -> int:
        """Emit an expression, return its SPIR-V result ID."""
        if isinstance(node, ir.IRConstant):
            return self._emit_constant(node)
        if isinstance(node, ir.IRName):
            return self._emit_name(node)
        if isinstance(node, ir.IRBinOp):
            return self._emit_binop(node)
        if isinstance(node, ir.IRUnaryOp):
            return self._emit_unaryop(node)
        if isinstance(node, ir.IRCompare):
            return self._emit_compare(node)
        if isinstance(node, ir.IRBoolOp):
            return self._emit_boolop(node)
        if isinstance(node, ir.IRFieldLoad):
            return self._emit_field_load(node)
        if isinstance(node, ir.IRCall):
            return self._emit_call(node)
        if isinstance(node, ir.IRCast):
            return self._emit_cast(node)
        if isinstance(node, ir.IRIfExp):
            return self._emit_ifexp(node)
        if isinstance(node, ir.IRAttribute):
            return self._emit_attribute(node)
        if isinstance(node, ir.IRTextureSample):
            return self._emit_texture_sample(node)
        if isinstance(node, ir.IRThreadId):
            return self._emit_thread_id()
        if isinstance(node, ir.IRBlockReduce):
            raise NotImplementedError(
                "pgc.block_sum/max/min are not yet supported on Vulkan/SPIR-V. "
                "Use manual shared memory reductions instead.")
        raise NotImplementedError(f"SPIR-V expr: {type(node).__name__}")

    def _emit_constant(self, node: ir.IRConstant) -> int:
        if isinstance(node.value, float):
            return self._const_f32(node.value)
        if isinstance(node.value, int):
            # Use u32 for non-negative, i32 for negative
            if node.value >= 0:
                return self._const_u32(node.value)
            return self._const_i32(node.value)
        raise TypeError(f"Unsupported constant: {type(node.value)}")

    def _emit_name(self, node: ir.IRName) -> int:
        if node.name in self._vars:
            val = self._vars[node.name]
            # If it's a local variable (pointer), load it
            if node.name in self._local_vars:
                ptr_id, type_key = self._local_vars[node.name]
                elem_type = self._get_type(type_key)
                result = self.module.alloc_id()
                self._body += _make_instruction(OpLoad, elem_type, result, ptr_id)
                self._id_types[result] = type_key
                return result
            return val
        # Field param names can appear as bare IRName in template-inlined code
        # (e.g. `local = __tmpl_grid_data__`). Return a sentinel so _emit_assign
        # can propagate the buffer alias.
        if node.name in self._param_buffers:
            return node.name  # Return name as sentinel for buffer alias
        raise NameError(f"Undefined variable in SPIR-V: {node.name}")

    def _emit_binop(self, node: ir.IRBinOp) -> int:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        # Coerce to same type (promote int to float if needed)
        left, right, result_type_key = self._coerce_pair(left, right)
        result_type = self._get_type(result_type_key)

        op_map_float = {
            "+": OpFAdd, "-": OpFSub, "*": OpFMul, "/": OpFDiv, "%": OpFRem,
        }
        op_map_int = {
            "+": OpIAdd, "-": OpISub, "*": OpIMul, "//": OpSDiv, "%": OpSRem,
            "<<": OpShiftLeftLogical, ">>": OpShiftRightArithmetic,
            "&": OpBitwiseAnd, "|": OpBitwiseOr, "^": OpBitwiseXor,
        }

        is_float = result_type_key == "f32"
        if is_float:
            if node.op == "**":
                return self._emit_glsl_ext(GLSL_Pow, [left, right], result_type_key)
            if node.op == "//":
                # floor(a / b)
                div = self._emit_binop_raw(OpFDiv, result_type, left, right, result_type_key)
                return self._emit_glsl_ext(GLSL_Floor, [div], result_type_key)
            # Integer division on floats
            if node.op == "/":
                opcode = OpFDiv
            else:
                opcode = op_map_float.get(node.op)
        else:
            opcode = op_map_int.get(node.op)

        if opcode is None:
            raise NotImplementedError(f"SPIR-V binop '{node.op}' for {result_type_key}")

        return self._emit_binop_raw(opcode, result_type, left, right, result_type_key)

    def _emit_binop_raw(self, opcode: int, result_type: int,
                         left: int, right: int,
                         result_type_key: str = "f32") -> int:
        result = self.module.alloc_id()
        self._body += _make_instruction(opcode, result_type, result, left, right)
        self._id_types[result] = result_type_key
        return result

    def _emit_unaryop(self, node: ir.IRUnaryOp) -> int:
        operand = self._emit_expr(node.operand)
        op_type = self._id_types.get(operand, "f32")
        if node.op == "-":
            if op_type == "f32":
                res_type = self._get_type("f32")
                result = self.module.alloc_id()
                self._body += _make_instruction(OpFNegate, res_type, result, operand)
                self._id_types[result] = "f32"
            else:
                # Integer negation: 0 - operand
                res_type = self._get_type(op_type)
                zero = self._const_u32(0) if op_type == "u32" else self._const_i32(0)
                result = self.module.alloc_id()
                self._body += _make_instruction(OpISub, res_type, result, zero, operand)
                self._id_types[result] = op_type
            return result
        if node.op == "not":
            bool_type = self._get_type("bool")
            result = self.module.alloc_id()
            self._body += _make_instruction(OpLogicalNot, bool_type, result, operand)
            self._id_types[result] = "bool"
            return result
        if node.op == "~":
            i32_type = self._get_type("i32")
            result = self.module.alloc_id()
            self._body += _make_instruction(OpNot, i32_type, result, operand)
            self._id_types[result] = "i32"
            return result
        raise NotImplementedError(f"SPIR-V unary op: {node.op}")

    def _emit_compare(self, node: ir.IRCompare) -> int:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        left, right, type_key = self._coerce_pair(left, right)
        bool_type = self._get_type("bool")

        if type_key == "f32":
            cmp_map = {
                "==": OpFOrdEqual, "!=": OpFOrdNotEqual,
                "<": OpFOrdLessThan, "<=": OpFOrdLessThanEqual,
                ">": OpFOrdGreaterThan, ">=": OpFOrdGreaterThanEqual,
            }
        else:
            cmp_map = {
                "==": OpIEqual, "!=": OpINotEqual,
                "<": OpSLessThan, "<=": OpSLessThanEqual,
                ">": OpSGreaterThan, ">=": OpSGreaterThanEqual,
            }

        opcode = cmp_map.get(node.op)
        if opcode is None:
            raise NotImplementedError(f"SPIR-V compare: {node.op}")

        result = self.module.alloc_id()
        self._body += _make_instruction(opcode, bool_type, result, left, right)
        self._id_types[result] = "bool"
        return result

    def _emit_boolop(self, node: ir.IRBoolOp) -> int:
        bool_type = self._get_type("bool")
        result = self._emit_expr(node.values[0])
        opcode = OpLogicalAnd if node.op == "and" else OpLogicalOr
        for val_node in node.values[1:]:
            val = self._emit_expr(val_node)
            new_result = self.module.alloc_id()
            self._body += _make_instruction(opcode, bool_type, new_result, result, val)
            self._id_types[new_result] = "bool"
            result = new_result
        return result

    def _emit_field_load(self, node: ir.IRFieldLoad) -> int:
        """Load from storage buffer or shared memory: buffer.data[index]."""
        if isinstance(node.field, ir.IRName) and node.field.name in self._param_buffers:
            buf_var, elem_key = self._param_buffers[node.field.name]
            index = self._emit_expr(node.index)
            index = self._to_u32(index)
            elem_type = self._get_type(elem_key)

            shared_vars = getattr(self, '_shared_vars', set())
            local_arrays = getattr(self, '_local_arrays', set())
            if node.field.name in local_arrays:
                # Local (Function) memory: plain array, no struct wrapping
                elem_ptr_key = f"ptr_fn_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, index)
            elif node.field.name in shared_vars:
                # Shared memory: plain array, no struct wrapping
                elem_ptr_key = f"ptr_wg_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, index)
            else:
                # Storage buffer: struct { RuntimeArray }, access member 0
                elem_ptr_key = f"ptr_sb_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                zero = self._const_u32(0)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, zero, index)

            result = self.module.alloc_id()
            self._body += _make_instruction(OpLoad, elem_type, result, ac)
            self._id_types[result] = elem_key
            return result

        raise NotImplementedError("Field load from non-parameter not supported")

    def _emit_texture_sample(self, node: ir.IRTextureSample) -> int:
        """Emit hardware texture sampling via OpImageSampleExplicitLod.

        PGC convention: texel centers at i/(N-1), u=0 → texel 0, u=1 → texel N-1.
        Vulkan normalized+linear: texel centers at (i+0.5)/N.
        Transform: vk_u = (u * (N-1) + 0.5) / N
        """
        W, H, D = node.shape
        f32_type = self._get_type("f32")
        vec3_type = self._get_type("vec3_f32")
        vec4_type = self._get_type("vec4_f32")
        sampled_type = self._get_type("sampled_image3d_f32")

        # Load the combined image/sampler
        tex_var = self._texture_vars[node.field_name]
        loaded = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, sampled_type, loaded, tex_var)

        # Emit and transform each coordinate
        transformed = []
        for dim_size, coord_node in zip((W, H, D), node.coords):
            raw = self._to_f32(self._emit_expr(coord_node))
            # vk_coord = (raw * (N-1) + 0.5) / N
            n_minus_1 = self._const_f32(float(dim_size - 1))
            half = self._const_f32(0.5)
            n_f = self._const_f32(float(dim_size))

            t1 = self.module.alloc_id()
            self._body += _make_instruction(OpFMul, f32_type, t1, raw, n_minus_1)
            self._id_types[t1] = "f32"
            t2 = self.module.alloc_id()
            self._body += _make_instruction(OpFAdd, f32_type, t2, t1, half)
            self._id_types[t2] = "f32"
            t3 = self.module.alloc_id()
            self._body += _make_instruction(OpFDiv, f32_type, t3, t2, n_f)
            self._id_types[t3] = "f32"
            transformed.append(t3)

        # Build vec3 coordinate
        coord_vec = self.module.alloc_id()
        self._body += _make_instruction(
            OpCompositeConstruct, vec3_type, coord_vec,
            transformed[0], transformed[1], transformed[2])

        # Sample with explicit LOD 0.0
        lod_0 = self._const_f32(0.0)
        sample_result = self.module.alloc_id()
        self._body += _make_instruction(
            OpImageSampleExplicitLod, vec4_type, sample_result,
            loaded, coord_vec, ImageOperand_Lod, lod_0)

        # Extract .x (red channel)
        scalar = self.module.alloc_id()
        self._body += _make_instruction(
            OpCompositeExtract, f32_type, scalar, sample_result, 0)
        self._id_types[scalar] = "f32"
        return scalar

    def _emit_field_store(self, node: ir.IRFieldStore):
        """Store to storage buffer or shared memory: buffer.data[index] = value."""
        if isinstance(node.field, ir.IRName) and node.field.name in self._param_buffers:
            buf_var, elem_key = self._param_buffers[node.field.name]
            index = self._emit_expr(node.index)
            index = self._to_u32(index)
            value = self._emit_expr(node.value)

            shared_vars = getattr(self, '_shared_vars', set())
            local_arrays = getattr(self, '_local_arrays', set())
            if node.field.name in local_arrays:
                elem_ptr_key = f"ptr_fn_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, index)
            elif node.field.name in shared_vars:
                elem_ptr_key = f"ptr_wg_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, index)
            else:
                elem_ptr_key = f"ptr_sb_{elem_key}"
                elem_ptr_type = self._get_type(elem_ptr_key)
                zero = self._const_u32(0)
                ac = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAccessChain, elem_ptr_type, ac, buf_var, zero, index)

            self._body += _make_instruction(OpStore, ac, value)
            return

        raise NotImplementedError("Field store to non-parameter not supported")

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        """Emit a SPIR-V atomic operation on a storage buffer."""
        if not (isinstance(node.field, ir.IRName) and node.field.name in self._param_buffers):
            raise NotImplementedError("Atomic on non-parameter not supported")

        buf_var, elem_key = self._param_buffers[node.field.name]
        index = self._emit_expr(node.index)
        index = self._to_u32(index)
        value = self._emit_expr(node.value)

        scope = self._const_u32(Scope_Device)
        semantics = self._const_u32(MemorySemantics_None)

        is_float = elem_key == "f32"

        if is_float:
            # For float atomics, use a uint-aliased view of the same buffer
            # to perform CAS-based operations
            uint_buf = self._get_uint_alias(node.field.name)
            u32_type = self._get_type("u32")
            u32_ptr_key = f"ptr_sb_u32"
            self._get_or_create_type(u32_ptr_key, lambda: self._make_pointer_type(
                StorageClass_StorageBuffer, u32_type, u32_ptr_key))
            u32_ptr_type = self._get_type(u32_ptr_key)
            zero = self._const_u32(0)
            uint_ac = self.module.alloc_id()
            self._body += _make_instruction(
                OpAccessChain, u32_ptr_type, uint_ac, uint_buf, zero, index)

            self._emit_float_atomic_cas_op(uint_ac, value, self._get_type(elem_key),
                                            elem_key, scope, semantics, node.op)
        else:
            elem_ptr_key = f"ptr_sb_{elem_key}"
            elem_ptr_type = self._get_type(elem_ptr_key)
            elem_type = self._get_type(elem_key)
            zero = self._const_u32(0)
            ac = self.module.alloc_id()
            self._body += _make_instruction(
                OpAccessChain, elem_ptr_type, ac, buf_var, zero, index)

            if node.op == "add":
                result = self.module.alloc_id()
                self._body += _make_instruction(
                    OpAtomicIAdd, elem_type, result, ac, scope, semantics, value)
            elif node.op in ("min", "max"):
                spv_op = OpAtomicSMin if node.op == "min" else OpAtomicSMax
                result = self.module.alloc_id()
                self._body += _make_instruction(
                    spv_op, elem_type, result, ac, scope, semantics, value)
            else:
                raise NotImplementedError(f"SPIR-V atomic op: {node.op}")

    def _get_uint_alias(self, param_name: str) -> int:
        """Get or create a uint-typed alias of a float storage buffer (same binding).

        This allows CAS operations on float data using uint atomics.
        """
        alias_key = f"_uint_alias_{param_name}"
        if alias_key in self._param_buffers:
            return self._param_buffers[alias_key][0]

        # Find the binding index for this param
        binding = None
        for i, param in enumerate(self.ir_func.params):
            if param.name == param_name:
                binding = i
                break

        u32_type = self._get_type("u32")

        # RuntimeArray of uint
        ra_key = "runtime_array_u32"
        if ra_key not in self._type_cache:
            ra_id = self.module.alloc_id()
            self._type_cache[ra_key] = ra_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypeRuntimeArray, ra_id, u32_type))
            self.module.add_annotation(
                _make_instruction(OpDecorate, ra_id, Decoration_ArrayStride, 4))
        ra_id = self._type_cache[ra_key]

        # Struct wrapping the runtime array
        struct_key = f"struct_ubuf_{binding}"
        struct_id = self.module.alloc_id()
        self._type_cache[struct_key] = struct_id
        self.module.add_type_or_constant(
            _make_instruction(OpTypeStruct, struct_id, ra_id))
        self.module.add_annotation(
            _make_instruction(OpDecorate, struct_id, Decoration_Block))
        self.module.add_annotation(
            _make_instruction(OpMemberDecorate, struct_id, 0, Decoration_Offset, 0))

        # Pointer to struct
        ptr_key = f"ptr_ubuf_{binding}"
        ptr_id = self.module.alloc_id()
        self._type_cache[ptr_key] = ptr_id
        self.module.add_type_or_constant(
            _make_instruction(OpTypePointer, ptr_id, StorageClass_StorageBuffer, struct_id))

        # Variable (same binding, aliased)
        var_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, ptr_id, var_id, StorageClass_StorageBuffer))
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_DescriptorSet, 0))
        self.module.add_annotation(
            _make_instruction(OpDecorate, var_id, Decoration_Binding, binding))
        # Mark as aliased
        Decoration_Aliased = 1800  # NonWritable=24, Aliased is not a standard decoration
        # Actually use NonWritable on the original? No. The correct way is to
        # just have two variables at the same binding — the driver handles aliasing.

        self._param_buffers[alias_key] = (var_id, "u32")
        return var_id

    def _emit_float_atomic_cas_op(self, uint_ptr_id, new_float_val, float_type,
                                   float_key, scope, semantics, op):
        """Emit float atomic via CAS loop on a uint-aliased pointer.

        uint_ptr_id points to uint storage (aliased with the float buffer).
        new_float_val is the float value to add/min/max.
        """
        u32_type = self._get_type("u32")

        # Load current uint value atomically
        initial_uint = self.module.alloc_id()
        self._body += _make_instruction(
            OpAtomicLoad, u32_type, initial_uint, uint_ptr_id, scope, semantics)

        # Store in a local for the loop
        func_u32_ptr_key = "ptr_func_u32"
        self._get_or_create_type(func_u32_ptr_key, lambda: self._make_pointer_type(
            StorageClass_Function, u32_type, func_u32_ptr_key))
        func_u32_ptr = self._get_type(func_u32_ptr_key)
        old_var = self.module.alloc_id()
        self._func_vars += _make_instruction(OpVariable, func_u32_ptr, old_var,
                                              StorageClass_Function)
        self._body += _make_instruction(OpStore, old_var, initial_uint)

        # Loop
        loop_header = self.module.alloc_id()
        loop_body = self.module.alloc_id()
        loop_continue = self.module.alloc_id()
        loop_merge = self.module.alloc_id()

        self._body += _make_instruction(OpBranch, loop_header)
        self._body += _make_instruction(OpLabel, loop_header)
        self._body += _make_instruction(OpLoopMerge, loop_merge, loop_continue,
                                         LoopControl_None)
        self._body += _make_instruction(OpBranch, loop_body)
        self._body += _make_instruction(OpLabel, loop_body)

        # Load expected uint
        old_uint = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, u32_type, old_uint, old_var)

        # Bitcast to float
        old_float = self.module.alloc_id()
        self._body += _make_instruction(OpBitcast, float_type, old_float, old_uint)

        # Compute desired float
        desired_float = self.module.alloc_id()
        if op == "add":
            self._body += _make_instruction(OpFAdd, float_type, desired_float,
                                             old_float, new_float_val)
        elif op == "min":
            self._body += _make_instruction(
                OpExtInst, float_type, desired_float, self._glsl_ext_id,
                GLSL_FMin, old_float, new_float_val)
        elif op == "max":
            self._body += _make_instruction(
                OpExtInst, float_type, desired_float, self._glsl_ext_id,
                GLSL_FMax, old_float, new_float_val)

        # Bitcast desired to uint
        desired_uint = self.module.alloc_id()
        self._body += _make_instruction(OpBitcast, u32_type, desired_uint, desired_float)

        # CAS on uint pointer
        cas_result = self.module.alloc_id()
        self._body += _make_instruction(
            OpAtomicCompareExchange, u32_type, cas_result, uint_ptr_id,
            scope, semantics, semantics, desired_uint, old_uint)

        # Check success
        cmp = self.module.alloc_id()
        self._body += _make_instruction(OpIEqual, self._get_type("bool"),
                                         cmp, cas_result, old_uint)

        # Update old_var for retry
        self._body += _make_instruction(OpStore, old_var, cas_result)

        self._body += _make_instruction(OpBranch, loop_continue)
        self._body += _make_instruction(OpLabel, loop_continue)
        self._body += _make_instruction(OpBranchConditional, cmp, loop_merge, loop_header)

        self._body += _make_instruction(OpLabel, loop_merge)

    def _emit_shared_alloc(self, node: ir.IRSharedAlloc):
        """Emit a Workgroup (shared) memory variable."""
        # Map dtype string to SPIR-V type key
        dtype_map = {"float": "f32", "int": "i32"}
        type_key = dtype_map.get(node.dtype, "f32")
        elem_type = self._get_type(type_key)

        # Get array size as constant
        if isinstance(node.size, ir.IRConstant):
            arr_size = node.size.value
        else:
            raise NotImplementedError("Shared memory with non-constant size")

        # Declare array type
        arr_key = f"arr_{type_key}_{arr_size}"
        if arr_key not in self._type_cache:
            arr_id = self.module.alloc_id()
            self._type_cache[arr_key] = arr_id
            size_const = self._const_u32(arr_size)
            self.module.add_type_or_constant(
                _make_instruction(OpTypeArray, arr_id, elem_type, size_const))
            # ArrayStride decoration
            stride = 4  # f32/i32 = 4 bytes
            self.module.add_annotation(
                _make_instruction(OpDecorate, arr_id, Decoration_ArrayStride, stride))

        arr_type = self._type_cache[arr_key]

        # Pointer to array in Workgroup storage class
        ptr_key = f"ptr_wg_{arr_key}"
        if ptr_key not in self._type_cache:
            ptr_id = self.module.alloc_id()
            self._type_cache[ptr_key] = ptr_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, ptr_id,
                                  StorageClass_Workgroup, arr_type))

        ptr_type = self._type_cache[ptr_key]

        # Declare variable (goes in global scope, not function)
        var_id = self.module.alloc_id()
        self.module.add_type_or_constant(
            _make_instruction(OpVariable, ptr_type, var_id, StorageClass_Workgroup))

        # Pointer to element in Workgroup
        elem_ptr_key = f"ptr_wg_{type_key}"
        if elem_ptr_key not in self._type_cache:
            elem_ptr_id = self.module.alloc_id()
            self._type_cache[elem_ptr_key] = elem_ptr_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, elem_ptr_id,
                                  StorageClass_Workgroup, elem_type))

        # Store as a "param buffer" so field load/store can access it
        self._param_buffers[node.name] = (var_id, type_key)
        # Mark it as shared (different access pattern — no struct wrapping)
        self._shared_vars = getattr(self, '_shared_vars', set())
        self._shared_vars.add(node.name)

    def _emit_local_alloc(self, node: ir.IRLocalAlloc):
        """Emit a Function (per-thread local) memory variable."""
        dtype_map = {"float": "f32", "int": "i32"}
        type_key = dtype_map.get(node.dtype, "f32")
        elem_type = self._get_type(type_key)

        if isinstance(node.size, ir.IRConstant):
            arr_size = node.size.value
        else:
            raise NotImplementedError("Local array with non-constant size")

        # Declare array type
        arr_key = f"arr_{type_key}_{arr_size}"
        if arr_key not in self._type_cache:
            arr_id = self.module.alloc_id()
            self._type_cache[arr_key] = arr_id
            size_const = self._const_u32(arr_size)
            self.module.add_type_or_constant(
                _make_instruction(OpTypeArray, arr_id, elem_type, size_const))
            stride = 4
            self.module.add_annotation(
                _make_instruction(OpDecorate, arr_id, Decoration_ArrayStride, stride))

        arr_type = self._type_cache[arr_key]

        # Pointer to array in Function storage class
        ptr_key = f"ptr_fn_{arr_key}"
        if ptr_key not in self._type_cache:
            ptr_id = self.module.alloc_id()
            self._type_cache[ptr_key] = ptr_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, ptr_id,
                                  StorageClass_Function, arr_type))

        ptr_type = self._type_cache[ptr_key]

        # Declare variable in function scope
        var_id = self.module.alloc_id()
        self._body += _make_instruction(OpVariable, ptr_type, var_id,
                                         StorageClass_Function)

        # Element pointer type in Function storage class
        elem_ptr_key = f"ptr_fn_{type_key}"
        if elem_ptr_key not in self._type_cache:
            elem_ptr_id = self.module.alloc_id()
            self._type_cache[elem_ptr_key] = elem_ptr_id
            self.module.add_type_or_constant(
                _make_instruction(OpTypePointer, elem_ptr_id,
                                  StorageClass_Function, elem_type))

        # Store as a "param buffer" so field load/store can access it
        self._param_buffers[node.name] = (var_id, type_key)
        self._shared_vars = getattr(self, '_shared_vars', set())
        self._shared_vars.add(node.name)
        # Track local arrays for Function storage class access
        if not hasattr(self, '_local_arrays'):
            self._local_arrays = set()
        self._local_arrays.add(node.name)

    def _emit_thread_id(self) -> int:
        """Emit gl_LocalInvocationID.x (thread index within workgroup)."""
        uvec3_type = self._get_type("uvec3")
        u32_type = self._get_type("u32")
        lid_load = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, uvec3_type, lid_load,
                                         self._local_invocation_id_var)
        lid_x = self.module.alloc_id()
        self._body += _make_instruction(OpCompositeExtract, u32_type, lid_x, lid_load, 0)
        self._id_types[lid_x] = "u32"
        return lid_x

    def _emit_barrier(self):
        """Emit a workgroup memory barrier."""
        scope_wg = self._const_u32(Scope_Workgroup)
        scope_wg2 = self._const_u32(Scope_Workgroup)
        semantics = self._const_u32(
            MemorySemantics_AcquireRelease | MemorySemantics_WorkgroupMemory)
        self._body += _make_instruction(
            OpControlBarrier, scope_wg, scope_wg2, semantics)

    def _emit_assign(self, node: ir.IRAssign):
        value = self._emit_expr(node.value)
        # Handle buffer alias: template inlining assigns a field param to a local
        # (e.g. `local = __tmpl_grid_data__`). Propagate the buffer binding.
        if isinstance(value, str) and value in self._param_buffers:
            self._param_buffers[node.target] = self._param_buffers[value]
            # Also propagate shared var status
            shared = getattr(self, '_shared_vars', set())
            if value in shared:
                shared.add(node.target)
            return
        if node.target in self._local_vars:
            ptr_id, type_key = self._local_vars[node.target]
            # Coerce value to the variable's type if needed
            val_type = self._id_types.get(value, "f32")
            if val_type != type_key:
                if type_key == "f32":
                    value = self._to_f32(value)
                elif type_key == "u32":
                    value = self._to_u32(value)
                elif type_key == "i32":
                    value = self._to_i32(value)
            self._body += _make_instruction(OpStore, ptr_id, value)
        else:
            # First assignment — infer type from value
            type_key = self._id_types.get(value, "f32")
            elem_type = self._get_type(type_key)
            ptr_key = f"ptr_func_{type_key}"
            ptr_type = self._get_or_create_type(ptr_key, lambda: self._make_pointer_type(
                StorageClass_Function, elem_type, ptr_key))
            ptr_id = self.module.alloc_id()
            # OpVariable must be in the first block — defer to _func_vars
            self._func_vars += _make_instruction(OpVariable, ptr_type, ptr_id,
                                                  StorageClass_Function)
            self._body += _make_instruction(OpStore, ptr_id, value)
            self._local_vars[node.target] = (ptr_id, type_key)
            self._vars[node.target] = ptr_id

    def _emit_call(self, node: ir.IRCall) -> int:
        """Emit GLSL.std.450 extended instruction."""
        glsl_op = _GLSL_FUNC_MAP.get(node.func_name)
        if glsl_op is None:
            raise NotImplementedError(f"SPIR-V builtin: {node.func_name}")
        args = [self._emit_expr(a) for a in node.args]
        # Determine result type: f64 if any arg is f64, else f32
        result_key = "f32"
        for a in args:
            if self._id_types.get(a) == "f64":
                result_key = "f64"
                break
        # Coerce integer args to float
        if result_key == "f64":
            args = [self._to_f64(a) for a in args]
        else:
            args = [self._to_f32(a) for a in args]
        if node.func_name == "log10":
            # Synthesize: log10(x) = log(x) * (1/log(10))
            log_x = self._emit_glsl_ext(GLSL_Log, args, result_key)
            if result_key == "f64":
                inv_log10 = self._const_f64(0.4342944819032518)
            else:
                inv_log10 = self._const_f32(0.4342944819032518)
            result = self.module.alloc_id()
            self._body += _make_instruction(OpFMul, self._get_type(result_key),
                                             result, log_x, inv_log10)
            self._id_types[result] = result_key
            return result
        return self._emit_glsl_ext(glsl_op, args, result_key)

    def _emit_glsl_ext(self, glsl_op: int, args: list[int], result_type_key: str) -> int:
        result_type = self._get_type(result_type_key)
        result = self.module.alloc_id()
        self._body += _make_instruction(
            OpExtInst, result_type, result, self._glsl_ext_id, glsl_op, *args)
        self._id_types[result] = result_type_key
        return result

    def _emit_cast(self, node: ir.IRCast) -> int:
        value = self._emit_expr(node.value)
        if node.dtype == "int":
            i32_type = self._get_type("i32")
            src_type = self._id_types.get(value, "f32")
            if src_type == "i32":
                return value  # already i32, no-op
            result = self.module.alloc_id()
            if src_type == "u32":
                self._body += _make_instruction(OpBitcast, i32_type, result, value)
            else:
                self._body += _make_instruction(OpConvertFToS, i32_type, result, value)
            self._id_types[result] = "i32"
            return result
        if node.dtype == "float":
            f32_type = self._get_type("f32")
            src_type = self._id_types.get(value, "f32")
            if src_type == "f32":
                return value  # already f32, no-op
            result = self.module.alloc_id()
            if src_type in ("u32", "i32"):
                op = OpConvertSToF if src_type == "i32" else OpConvertUToF
                self._body += _make_instruction(op, f32_type, result, value)
            else:
                self._body += _make_instruction(OpConvertSToF, f32_type, result, value)
            self._id_types[result] = "f32"
            return result
        raise NotImplementedError(f"SPIR-V cast to {node.dtype}")

    def _emit_ifexp(self, node: ir.IRIfExp) -> int:
        cond = self._emit_expr(node.condition)
        then_val = self._emit_expr(node.then_value)
        else_val = self._emit_expr(node.else_value)
        type_key = self._id_types.get(then_val, "f32")
        res_type = self._get_type(type_key)
        result = self.module.alloc_id()
        self._body += _make_instruction(OpSelect, res_type, result,
                                         cond, then_val, else_val)
        self._id_types[result] = type_key
        return result

    def _emit_attribute(self, node: ir.IRAttribute) -> int:
        """Handle attribute access — only shape/len resolved at compile time."""
        raise NotImplementedError(
            f"Attribute '{node.attr}' must be resolved before SPIR-V codegen")

    # --- Control flow ---

    def _emit_if(self, node: ir.IRIf):
        cond = self._emit_expr(node.condition)

        then_label = self.module.alloc_id()
        else_label = self.module.alloc_id() if node.else_body else None
        merge_label = self.module.alloc_id()

        self._body += _make_instruction(OpSelectionMerge, merge_label,
                                         SelectionControl_None)
        if else_label:
            self._body += _make_instruction(OpBranchConditional, cond,
                                             then_label, else_label)
        else:
            self._body += _make_instruction(OpBranchConditional, cond,
                                             then_label, merge_label)

        # Then
        self._body += _make_instruction(OpLabel, then_label)
        self._emit_body(node.then_body)
        if not self._last_is_terminator():
            self._body += _make_instruction(OpBranch, merge_label)

        # Else
        if else_label:
            self._body += _make_instruction(OpLabel, else_label)
            self._emit_body(node.else_body)
            if not self._last_is_terminator():
                self._body += _make_instruction(OpBranch, merge_label)

        self._body += _make_instruction(OpLabel, merge_label)

    def _emit_sequential_for(self, node: ir.IRSequentialFor):
        """Emit a sequential loop using OpLoopMerge.

        SPIR-V structured control flow layout:
            [pre-header]  store start → loop_var_ptr, branch → header
            [header]      OpLoopMerge merge, continue; branch → cond_block
            [cond_block]  load loop_var, compare, branch_conditional → body/merge
            [body]        ... user code ...  branch → continue
            [continue]    increment, store, branch → header
            [merge]       exit
        """
        u32_type = self._get_type("u32")
        start = self._to_u32(self._emit_expr(node.start))
        end = self._to_u32(self._emit_expr(node.end))

        header_label = self.module.alloc_id()
        cond_label = self.module.alloc_id()
        body_label = self.module.alloc_id()
        continue_label = self.module.alloc_id()
        merge_label = self.module.alloc_id()

        # Create loop variable (hoisted to entry block)
        ptr_key = "ptr_func_u32"
        ptr_type = self._get_or_create_type(ptr_key, lambda: self._make_pointer_type(
            StorageClass_Function, u32_type, ptr_key))
        loop_var_ptr = self.module.alloc_id()
        self._func_vars += _make_instruction(OpVariable, ptr_type, loop_var_ptr,
                                              StorageClass_Function)

        # Pre-header: initialize loop var and branch to header
        self._body += _make_instruction(OpStore, loop_var_ptr, start)
        self._body += _make_instruction(OpBranch, header_label)

        # Header: OpLoopMerge must be immediately followed by OpBranch
        self._body += _make_instruction(OpLabel, header_label)
        self._body += _make_instruction(OpLoopMerge, merge_label, continue_label,
                                         LoopControl_None)
        self._body += _make_instruction(OpBranch, cond_label)

        # Condition block: load, compare, conditional branch
        self._body += _make_instruction(OpLabel, cond_label)
        loop_val = self.module.alloc_id()
        self._body += _make_instruction(OpLoad, u32_type, loop_val, loop_var_ptr)
        self._id_types[loop_val] = "u32"

        bool_type = self._get_type("bool")
        cond = self.module.alloc_id()
        self._body += _make_instruction(OpULessThan, bool_type, cond, loop_val, end)
        self._id_types[cond] = "bool"
        self._body += _make_instruction(OpBranchConditional, cond, body_label, merge_label)

        # Body
        self._body += _make_instruction(OpLabel, body_label)
        old_var = self._vars.get(node.var)
        self._vars[node.var] = loop_val

        old_break = self._break_label
        old_continue = self._continue_label
        self._break_label = merge_label
        self._continue_label = continue_label

        self._emit_body(node.body)

        self._break_label = old_break
        self._continue_label = old_continue

        if not self._last_is_terminator():
            self._body += _make_instruction(OpBranch, continue_label)

        # Continue block: increment and branch back to header
        self._body += _make_instruction(OpLabel, continue_label)
        one = self._const_u32(1)
        next_val = self.module.alloc_id()
        self._body += _make_instruction(OpIAdd, u32_type, next_val, loop_val, one)
        self._id_types[next_val] = "u32"
        self._body += _make_instruction(OpStore, loop_var_ptr, next_val)
        self._body += _make_instruction(OpBranch, header_label)

        # Merge (loop exit)
        self._body += _make_instruction(OpLabel, merge_label)

        if old_var is not None:
            self._vars[node.var] = old_var
        else:
            self._vars.pop(node.var, None)

    def _emit_while(self, node: ir.IRWhile):
        header_label = self.module.alloc_id()
        body_label = self.module.alloc_id()
        continue_label = self.module.alloc_id()
        merge_label = self.module.alloc_id()

        self._body += _make_instruction(OpBranch, header_label)
        self._body += _make_instruction(OpLabel, header_label)
        self._body += _make_instruction(OpLoopMerge, merge_label, continue_label,
                                         LoopControl_None)

        cond = self._emit_expr(node.condition)
        self._body += _make_instruction(OpBranchConditional, cond, body_label, merge_label)

        self._body += _make_instruction(OpLabel, body_label)

        old_break = self._break_label
        old_continue = self._continue_label
        self._break_label = merge_label
        self._continue_label = continue_label

        self._emit_body(node.body)

        self._break_label = old_break
        self._continue_label = old_continue

        self._body += _make_instruction(OpBranch, continue_label)
        self._body += _make_instruction(OpLabel, continue_label)
        self._body += _make_instruction(OpBranch, header_label)

        self._body += _make_instruction(OpLabel, merge_label)

    def _emit_break(self):
        if self._break_label is None:
            raise RuntimeError("break outside of loop")
        self._body += _make_instruction(OpBranch, self._break_label)

    def _emit_continue(self):
        if self._continue_label is None:
            raise RuntimeError("continue outside of loop")
        self._body += _make_instruction(OpBranch, self._continue_label)

    # --- Type coercion ---

    def _to_u32(self, val_id: int) -> int:
        """Coerce to u32 for array indexing."""
        src_type = self._id_types.get(val_id, "u32")
        if src_type == "u32":
            return val_id
        if src_type == "f32":
            u32_type = self._get_type("u32")
            result = self.module.alloc_id()
            self._body += _make_instruction(OpConvertFToU, u32_type, result, val_id)
            self._id_types[result] = "u32"
            return result
        if src_type == "i32":
            u32_type = self._get_type("u32")
            result = self.module.alloc_id()
            self._body += _make_instruction(OpBitcast, u32_type, result, val_id)
            self._id_types[result] = "u32"
            return result
        return val_id

    def _to_i32(self, val_id: int) -> int:
        """Coerce to i32."""
        src_type = self._id_types.get(val_id, "i32")
        if src_type == "i32":
            return val_id
        i32_type = self._get_type("i32")
        result = self.module.alloc_id()
        if src_type == "f32":
            self._body += _make_instruction(OpConvertFToS, i32_type, result, val_id)
        elif src_type == "u32":
            self._body += _make_instruction(OpBitcast, i32_type, result, val_id)
        else:
            self._body += _make_instruction(OpConvertFToS, i32_type, result, val_id)
        self._id_types[result] = "i32"
        return result

    def _to_f32(self, val_id: int) -> int:
        """Convert a value to f32."""
        src_type = self._id_types.get(val_id, "f32")
        if src_type == "f32":
            return val_id
        f32_type = self._get_type("f32")
        result = self.module.alloc_id()
        if src_type == "u32" or src_type == "u64":
            self._body += _make_instruction(OpConvertUToF, f32_type, result, val_id)
        elif src_type == "f64":
            self._body += _make_instruction(OpFConvert, f32_type, result, val_id)
        else:
            self._body += _make_instruction(OpConvertSToF, f32_type, result, val_id)
        self._id_types[result] = "f32"
        return result

    def _to_f64(self, val_id: int) -> int:
        """Convert a value to f64."""
        src_type = self._id_types.get(val_id, "f32")
        if src_type == "f64":
            return val_id
        f64_type = self._get_type("f64")
        result = self.module.alloc_id()
        if src_type == "f32":
            self._body += _make_instruction(OpFConvert, f64_type, result, val_id)
        elif src_type in ("u32", "u64"):
            self._body += _make_instruction(OpConvertUToF, f64_type, result, val_id)
        else:
            self._body += _make_instruction(OpConvertSToF, f64_type, result, val_id)
        self._id_types[result] = "f64"
        return result

    def _coerce_pair(self, left: int, right: int) -> tuple[int, int, str]:
        """Coerce two values to a common type. Returns (left, right, type_key).

        Promotion: u32/i32 → f32 if either operand is f32.
        If both are integer, keep as u32.
        """
        left_type = self._id_types.get(left, "f32")
        right_type = self._id_types.get(right, "f32")

        # If both are the same, done
        if left_type == right_type:
            return left, right, left_type

        # If either is f64, promote both to f64
        if left_type == "f64" or right_type == "f64":
            return self._to_f64(left), self._to_f64(right), "f64"

        # If either is f32, promote both to f32
        if left_type == "f32" or right_type == "f32":
            return self._to_f32(left), self._to_f32(right), "f32"

        # Integer widening: if either is 64-bit, promote
        if left_type in ("i64", "u64") or right_type in ("i64", "u64"):
            return left, right, left_type  # keep as-is, same width

        # Both integer but different signedness — use u32
        return left, right, "u32"


def generate_spirv(ir_func: ir.IRFunction, workgroup_size: int = 256) -> bytes:
    """Generate a SPIR-V compute shader binary from a PGC IR function."""
    codegen = SPIRVCodeGen(ir_func, workgroup_size)
    return codegen.generate()

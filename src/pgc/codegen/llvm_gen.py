"""PGC LLVM IR code generation — transforms PGC IR to LLVM IR via llvmlite.

Generates a kernel function with signature:
    void kernel(ptr field0, ptr field1, ..., i64 scalar0, ..., i64 n)

Fields are passed as typed pointers. The outermost parallel for-loop becomes
a simple sequential loop in the LLVM IR — the CPU backend splits the range
across threads and calls the kernel with different (start, end) pairs.
"""

from llvmlite import ir as llvm_ir

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64


def _llvm_type(pgc_type: ScalarType) -> llvm_ir.Type:
    """Convert a PGC scalar type to an LLVM IR type."""
    if pgc_type is f32:
        return llvm_ir.FloatType()
    if pgc_type is f64:
        return llvm_ir.DoubleType()
    if pgc_type in (i32, u32):
        return llvm_ir.IntType(32)
    if pgc_type in (i64, u64):
        return llvm_ir.IntType(64)
    raise TypeError(f"No LLVM type mapping for {pgc_type}")


def _is_float_type(llvm_type: llvm_ir.Type) -> bool:
    return isinstance(llvm_type, (llvm_ir.FloatType, llvm_ir.DoubleType))


def _is_int_type(llvm_type: llvm_ir.Type) -> bool:
    return isinstance(llvm_type, llvm_ir.IntType)


class LLVMCodeGen:
    """Generates LLVM IR from PGC IR for a single kernel function.

    The generated function takes:
      - One pointer per Field parameter (element type based on field dtype)
      - One scalar per scalar parameter
      - Two i64 values: loop_start, loop_end (for the parallel for range)

    The parallel for-loop is emitted as a simple loop from loop_start to loop_end.
    The CPU runtime calls this function once per thread with a subrange.
    """

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = ir_func
        self.module = llvm_ir.Module(name=ir_func.name)
        self.builder: llvm_ir.IRBuilder | None = None

        # Maps PGC parameter names to LLVM values
        self._params: dict[str, llvm_ir.Value] = {}
        # Maps PGC parameter names to their types (for knowing element type of fields)
        self._param_types: dict[str, ScalarType] = {}
        # Which params are fields (pointers) vs scalars
        self._field_params: set[str] = set()
        # Local variable storage (name -> alloca)
        self._locals: dict[str, llvm_ir.Value] = {}

        # Break/continue targets for loops
        self._break_target: llvm_ir.Block | None = None
        self._continue_target: llvm_ir.Block | None = None

    def generate(self) -> llvm_ir.Module:
        """Generate LLVM IR for the kernel. Returns the LLVM module."""
        func = self.ir_func

        # Classify parameters: fields become pointers, scalars stay scalar
        llvm_param_types = []
        param_names = []
        for param in func.params:
            pgc_type = param.type_annotation
            if pgc_type is None:
                raise TypeError(f"Parameter '{param.name}' has no type annotation. "
                                "Run type inference first.")
            self._param_types[param.name] = pgc_type
            elem_llvm = _llvm_type(pgc_type)
            if hasattr(param, '_is_field') and param._is_field:
                llvm_param_types.append(elem_llvm.as_pointer())
                self._field_params.add(param.name)
            elif hasattr(param, '_is_field') and not param._is_field:
                # Scalar parameter — passed by value
                llvm_param_types.append(elem_llvm)
            else:
                # Default: treat as pointer (field) for backwards compatibility
                llvm_param_types.append(elem_llvm.as_pointer())
                self._field_params.add(param.name)
            param_names.append(param.name)

        # Add loop_start and loop_end parameters (i64)
        i64_type = llvm_ir.IntType(64)
        llvm_param_types.extend([i64_type, i64_type])
        param_names.extend(["__loop_start__", "__loop_end__"])

        fn_type = llvm_ir.FunctionType(llvm_ir.VoidType(), llvm_param_types)
        llvm_func = llvm_ir.Function(self.module, fn_type, name=func.name)

        # Name the arguments and mark pointer params as noalias
        for i, (arg, name) in enumerate(zip(llvm_func.args, param_names)):
            arg.name = name
            self._params[name] = arg
            if name in self._field_params:
                llvm_func.args[i].add_attribute("noalias")

        entry = llvm_func.append_basic_block("entry")
        self.builder = llvm_ir.IRBuilder(entry)
        self._func = llvm_func
        self._entry_block = entry

        self._emit_body(func.body)

        # Make sure the last block is terminated
        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        return self.module

    def _emit_body(self, stmts: list):
        """Emit a list of IR statements."""
        for stmt in stmts:
            self._emit_stmt(stmt)
            # Stop emitting after a terminator (break/continue/return)
            if self.builder.block.is_terminated:
                break

    def _emit_stmt(self, node: ir.IRNode):
        """Emit a single IR statement."""
        if isinstance(node, ir.IRParallelFor):
            self._emit_parallel_for(node)
        elif isinstance(node, ir.IRSequentialFor):
            self._emit_sequential_for(node)
        elif isinstance(node, ir.IRWhile):
            self._emit_while(node)
        elif isinstance(node, ir.IRIf):
            self._emit_if(node)
        elif isinstance(node, ir.IRFieldStore):
            self._emit_field_store(node)
        elif isinstance(node, ir.IRAssign):
            self._emit_assign(node)
        elif isinstance(node, ir.IRReturn):
            self._emit_return(node)
        elif isinstance(node, ir.IRBreak):
            self._emit_break()
        elif isinstance(node, ir.IRContinue):
            self._emit_continue()
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRPrint):
            self._emit_print(node)
        elif isinstance(node, ir.IRSharedAlloc):
            self._emit_shared_alloc(node)
        elif isinstance(node, ir.IRLocalAlloc):
            self._emit_shared_alloc(node)  # same as shared on CPU: stack alloca
        elif isinstance(node, ir.IRBarrier):
            pass  # No-op on CPU (single-threaded per chunk)
        elif isinstance(node, ir.IRCall):
            # Standalone function call (expression statement)
            self._emit_expr(node)
        else:
            raise NotImplementedError(f"Cannot emit statement: {type(node).__name__}")

    def _emit_expr(self, node: ir.IRNode) -> llvm_ir.Value:
        """Emit an expression and return its LLVM value."""
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
        if isinstance(node, ir.IRAttribute):
            return self._emit_attribute(node)
        if isinstance(node, ir.IRCall):
            return self._emit_call(node)
        if isinstance(node, ir.IRCast):
            return self._emit_cast(node)
        if isinstance(node, ir.IRIfExp):
            return self._emit_ifexp(node)
        if isinstance(node, ir.IRTextureSample):
            return self._emit_texture_sample(node)
        if isinstance(node, ir.IRThreadId):
            # On CPU, thread_id within a chunk is (loop_var - loop_start)
            # Return 0 as a safe default (CPU doesn't have workgroups)
            return llvm_ir.Constant(llvm_ir.IntType(64), 0)
        raise NotImplementedError(f"Cannot emit expression: {type(node).__name__}")

    # --- Loops ---

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        """Emit the parallel for-loop using __loop_start__ and __loop_end__."""
        i64_type = llvm_ir.IntType(64)
        start = self._params["__loop_start__"]
        end = self._params["__loop_end__"]

        header = self._func.append_basic_block(f"for.{node.var}.header")
        body = self._func.append_basic_block(f"for.{node.var}.body")
        exit_bb = self._func.append_basic_block(f"for.{node.var}.exit")

        entry_block = self.builder.block
        self.builder.branch(header)
        self.builder = llvm_ir.IRBuilder(header)

        phi = self.builder.phi(i64_type, name=node.var)
        phi.add_incoming(start, entry_block)

        cond = self.builder.icmp_signed("<", phi, end, name=f"{node.var}.cond")
        self.builder.cbranch(cond, body, exit_bb)

        self.builder = llvm_ir.IRBuilder(body)

        # Store loop var for access in body
        old_local = self._locals.get(node.var)
        self._locals[node.var] = phi

        old_break = self._break_target
        old_continue = self._continue_target
        self._break_target = exit_bb
        self._continue_target = header

        self._emit_body(node.body)

        self._break_target = old_break
        self._continue_target = old_continue

        # Increment and branch back
        if not self.builder.block.is_terminated:
            next_val = self.builder.add(phi, llvm_ir.Constant(i64_type, 1),
                                        name=f"{node.var}.next")
            phi.add_incoming(next_val, self.builder.block)
            self.builder.branch(header)

        if old_local is not None:
            self._locals[node.var] = old_local
        else:
            self._locals.pop(node.var, None)

        self.builder = llvm_ir.IRBuilder(exit_bb)

    def _emit_sequential_for(self, node: ir.IRSequentialFor):
        """Emit a sequential (nested) for-loop."""
        i64_type = llvm_ir.IntType(64)
        start = self._to_i64(self._emit_expr(node.start))
        end = self._to_i64(self._emit_expr(node.end))

        header = self._func.append_basic_block(f"for.{node.var}.header")
        body = self._func.append_basic_block(f"for.{node.var}.body")
        exit_bb = self._func.append_basic_block(f"for.{node.var}.exit")

        entry_block = self.builder.block
        self.builder.branch(header)
        self.builder = llvm_ir.IRBuilder(header)

        phi = self.builder.phi(i64_type, name=node.var)
        phi.add_incoming(start, entry_block)

        cond = self.builder.icmp_signed("<", phi, end, name=f"{node.var}.cond")
        self.builder.cbranch(cond, body, exit_bb)

        self.builder = llvm_ir.IRBuilder(body)

        old_local = self._locals.get(node.var)
        self._locals[node.var] = phi

        old_break = self._break_target
        old_continue = self._continue_target
        self._break_target = exit_bb
        self._continue_target = header

        self._emit_body(node.body)

        self._break_target = old_break
        self._continue_target = old_continue

        if not self.builder.block.is_terminated:
            if node.step is not None:
                step_val = self._to_i64(self._emit_expr(node.step))
                next_val = self.builder.add(phi, step_val,
                                            name=f"{node.var}.next")
            else:
                next_val = self.builder.add(phi, llvm_ir.Constant(i64_type, 1),
                                            name=f"{node.var}.next")
            phi.add_incoming(next_val, self.builder.block)
            self.builder.branch(header)

        if old_local is not None:
            self._locals[node.var] = old_local
        else:
            self._locals.pop(node.var, None)

        self.builder = llvm_ir.IRBuilder(exit_bb)

    def _emit_while(self, node: ir.IRWhile):
        """Emit a while-loop."""
        header = self._func.append_basic_block("while.header")
        body = self._func.append_basic_block("while.body")
        exit_bb = self._func.append_basic_block("while.exit")

        self.builder.branch(header)
        self.builder = llvm_ir.IRBuilder(header)

        cond = self._emit_expr(node.condition)
        cond = self._to_i1(cond)
        self.builder.cbranch(cond, body, exit_bb)

        self.builder = llvm_ir.IRBuilder(body)

        old_break = self._break_target
        old_continue = self._continue_target
        self._break_target = exit_bb
        self._continue_target = header

        self._emit_body(node.body)

        self._break_target = old_break
        self._continue_target = old_continue

        if not self.builder.block.is_terminated:
            self.builder.branch(header)

        self.builder = llvm_ir.IRBuilder(exit_bb)

    def _emit_break(self):
        if self._break_target is None:
            raise RuntimeError("break outside of loop")
        self.builder.branch(self._break_target)

    def _emit_continue(self):
        if self._continue_target is None:
            raise RuntimeError("continue outside of loop")
        self.builder.branch(self._continue_target)

    # --- Control flow ---

    def _emit_if(self, node: ir.IRIf):
        """Emit an if/else statement."""
        cond = self._emit_expr(node.condition)
        cond = self._to_i1(cond)

        then_bb = self._func.append_basic_block("if.then")
        else_bb = self._func.append_basic_block("if.else") if node.else_body else None
        merge_bb = self._func.append_basic_block("if.merge")

        if else_bb:
            self.builder.cbranch(cond, then_bb, else_bb)
        else:
            self.builder.cbranch(cond, then_bb, merge_bb)

        # Then
        self.builder = llvm_ir.IRBuilder(then_bb)
        self._emit_body(node.then_body)
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_bb)

        # Else
        if else_bb:
            self.builder = llvm_ir.IRBuilder(else_bb)
            self._emit_body(node.else_body)
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_bb)

        self.builder = llvm_ir.IRBuilder(merge_bb)

    def _emit_ifexp(self, node: ir.IRIfExp) -> llvm_ir.Value:
        """Emit a ternary expression using select."""
        cond = self._emit_expr(node.condition)
        cond = self._to_i1(cond)
        then_val = self._emit_expr(node.then_value)
        else_val = self._emit_expr(node.else_value)
        then_val, else_val = self._coerce_pair(then_val, else_val)
        return self.builder.select(cond, then_val, else_val, name="ifexp")

    # --- Assignments ---

    def _emit_field_store(self, node: ir.IRFieldStore):
        """Emit: field[index] = value."""
        base_ptr = self._emit_expr(node.field)
        index = self._to_i64(self._emit_expr(node.index))
        value = self._emit_expr(node.value)

        elem_ptr = self.builder.gep(base_ptr, [index], inbounds=True, name="store.ptr")
        # Cast value to the element type of the pointer
        elem_type = base_ptr.type.pointee
        value = self._coerce_to(value, elem_type)
        self.builder.store(value, elem_ptr, align=4)

    def _create_entry_alloca(self, typ, name):
        """Create an alloca in the function entry block (ensures domination)."""
        cur_block = self.builder.block
        # Position at the start of the entry block
        if self._entry_block.instructions:
            self.builder.position_before(self._entry_block.instructions[0])
        else:
            self.builder.position_at_end(self._entry_block)
        alloca = self.builder.alloca(typ, name=name)
        # Restore builder position
        self.builder.position_at_end(cur_block)
        return alloca

    def _emit_assign(self, node: ir.IRAssign):
        """Emit local variable assignment.

        Uses alloca for local variables so they can be reassigned.
        Allocas are placed in the entry block to ensure LLVM domination.
        """
        value = self._emit_expr(node.value)

        if node.target in self._locals:
            existing = self._locals[node.target]
            # If existing is an alloca, store into it
            if hasattr(existing, 'opname') and existing.opname == 'alloca':
                value = self._coerce_to(value, existing.type.pointee)
                self.builder.store(value, existing)
                return
            # If it's a phi or direct value, replace with alloca
            alloca = self._create_entry_alloca(value.type, node.target)
            self.builder.store(value, alloca)
            self._locals[node.target] = alloca
        else:
            alloca = self._create_entry_alloca(value.type, node.target)
            self.builder.store(value, alloca)
            self._locals[node.target] = alloca

    def _emit_return(self, node: ir.IRReturn):
        if node.value is not None:
            # Kernels return void; ignore return value for now
            self._emit_expr(node.value)
        self.builder.ret_void()

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        """Emit an atomic operation on a field element."""
        base_ptr = self._emit_expr(node.field)
        index = self._to_i64(self._emit_expr(node.index))
        value = self._emit_expr(node.value)

        elem_ptr = self.builder.gep(base_ptr, [index], inbounds=True, name="atomic.ptr")
        elem_type = base_ptr.type.pointee
        value = self._coerce_to(value, elem_type)

        if node.op == "add":
            if _is_float_type(elem_type):
                self.builder.atomic_rmw("fadd", elem_ptr, value, "monotonic")
            else:
                self.builder.atomic_rmw("add", elem_ptr, value, "monotonic")
        elif node.op == "min":
            if _is_float_type(elem_type):
                # Float atomic min via compare-and-swap loop
                self._emit_atomic_float_minmax(elem_ptr, value, "min")
            else:
                self.builder.atomic_rmw("min", elem_ptr, value, "monotonic")
        elif node.op == "max":
            if _is_float_type(elem_type):
                self._emit_atomic_float_minmax(elem_ptr, value, "max")
            else:
                self.builder.atomic_rmw("max", elem_ptr, value, "monotonic")
        else:
            raise NotImplementedError(f"Atomic op: {node.op}")

    def _emit_atomic_float_minmax(self, ptr, value, op: str):
        """Emit a float atomic min/max via compare-and-swap loop."""
        # For CPU, just do a non-atomic load-compare-store (single-threaded per element)
        old_val = self.builder.load(ptr, name="atomic.old")
        if op == "min":
            cond = self.builder.fcmp_ordered("<", value, old_val, name="atomic.cmp")
        else:
            cond = self.builder.fcmp_ordered(">", value, old_val, name="atomic.cmp")
        new_val = self.builder.select(cond, value, old_val, name="atomic.new")
        self.builder.store(new_val, ptr)

    def _emit_shared_alloc(self, node: ir.IRSharedAlloc):
        """Emit shared memory as a stack alloca (CPU has no shared memory)."""
        _dtype_llvm = {"float": llvm_ir.FloatType(), "double": llvm_ir.DoubleType(),
                       "int": llvm_ir.IntType(32), "long": llvm_ir.IntType(64),
                       "uint": llvm_ir.IntType(32), "ulong": llvm_ir.IntType(64)}
        elem_type = _dtype_llvm.get(node.dtype, llvm_ir.FloatType())
        # Use constant size for the alloca
        if isinstance(node.size, ir.IRConstant):
            arr_type = llvm_ir.ArrayType(elem_type, node.size.value)
            alloca = self._create_entry_alloca(arr_type, node.name)
            # Bitcast to pointer for array access
            ptr = self.builder.bitcast(alloca, elem_type.as_pointer())
            self._locals[node.name] = ptr
            self._field_params.add(node.name)  # treat as pointer for load/store
        else:
            # Dynamic size — use a fixed upper bound
            arr_type = llvm_ir.ArrayType(elem_type, 256)
            alloca = self._create_entry_alloca(arr_type, node.name)
            ptr = self.builder.bitcast(alloca, elem_type.as_pointer())
            self._locals[node.name] = ptr
            self._field_params.add(node.name)

    _printf_count = 0

    def _emit_print(self, node: ir.IRPrint):
        """Emit a printf call for kernel debugging."""
        fmt_parts = []
        call_args = []

        if node.format_parts:
            for i, (kind, val) in enumerate(node.format_parts):
                if i > 0:
                    fmt_parts.append(" ")
                if kind == "str":
                    fmt_parts.append(val.replace("%", "%%"))
                else:
                    arg_val = self._emit_expr(node.args[val])
                    if _is_float_type(arg_val.type):
                        arg_val = self.builder.fpext(arg_val, llvm_ir.DoubleType())
                        fmt_parts.append("%f")
                    else:
                        fmt_parts.append("%lld")
                    call_args.append(arg_val)
        else:
            for i, arg_node in enumerate(node.args):
                if i > 0:
                    fmt_parts.append(" ")
                arg_val = self._emit_expr(arg_node)
                if _is_float_type(arg_val.type):
                    arg_val = self.builder.fpext(arg_val, llvm_ir.DoubleType())
                    fmt_parts.append("%f")
                else:
                    fmt_parts.append("%lld")
                call_args.append(arg_val)

        fmt_str = "".join(fmt_parts) + "\n"

        # Create global string constant for format
        fmt_bytes = (fmt_str + "\0").encode("utf-8")
        fmt_type = llvm_ir.ArrayType(llvm_ir.IntType(8), len(fmt_bytes))
        LLVMCodeGen._printf_count += 1
        fmt_global = llvm_ir.GlobalVariable(self.module, fmt_type,
                                            name=f".printf_fmt.{LLVMCodeGen._printf_count}")
        fmt_global.global_constant = True
        fmt_global.linkage = "internal"
        fmt_global.initializer = llvm_ir.Constant(fmt_type,
                                                   bytearray(fmt_bytes))

        # Get or declare printf
        i8_ptr = llvm_ir.IntType(8).as_pointer()
        try:
            printf_fn = self.module.get_global("printf")
        except KeyError:
            printf_type = llvm_ir.FunctionType(llvm_ir.IntType(32), [i8_ptr], var_arg=True)
            printf_fn = llvm_ir.Function(self.module, printf_type, name="printf")

        # GEP to get pointer to first element of the format string
        zero = llvm_ir.Constant(llvm_ir.IntType(32), 0)
        fmt_ptr = self.builder.gep(fmt_global, [zero, zero], inbounds=True)

        self.builder.call(printf_fn, [fmt_ptr] + call_args)

    # --- Expressions ---

    def _emit_constant(self, node: ir.IRConstant) -> llvm_ir.Value:
        if isinstance(node.value, float):
            return llvm_ir.Constant(llvm_ir.FloatType(), node.value)
        if isinstance(node.value, int):
            return llvm_ir.Constant(llvm_ir.IntType(64), node.value)
        if isinstance(node.value, bool):
            return llvm_ir.Constant(llvm_ir.IntType(1), int(node.value))
        raise TypeError(f"Unsupported constant type: {type(node.value)}")

    def _emit_name(self, node: ir.IRName) -> llvm_ir.Value:
        # Check locals first (allocas need a load)
        if node.name in self._locals:
            val = self._locals[node.name]
            if hasattr(val, 'opname') and val.opname == 'alloca':
                return self.builder.load(val, name=f"{node.name}.load")
            return val
        # Then function parameters
        if node.name in self._params:
            return self._params[node.name]
        raise NameError(f"Undefined variable: {node.name}")

    def _emit_binop(self, node: ir.IRBinOp) -> llvm_ir.Value:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        left, right = self._coerce_pair(left, right)

        if _is_float_type(left.type):
            return self._emit_float_binop(node.op, left, right)
        if _is_int_type(left.type):
            return self._emit_int_binop(node.op, left, right)
        raise TypeError(f"Unsupported operand type for {node.op}: {left.type}")

    def _emit_float_binop(self, op: str, left, right) -> llvm_ir.Value:
        ops = {
            "+": self.builder.fadd,
            "-": self.builder.fsub,
            "*": self.builder.fmul,
            "/": self.builder.fdiv,
            "%": self.builder.frem,
        }
        if op in ops:
            return ops[op](left, right, name="binop")
        if op == "**":
            powf = self.module.declare_intrinsic('llvm.pow', [left.type])
            return self.builder.call(powf, [left, right], name="pow")
        if op == "//":
            # Floor division for floats: floor(a / b)
            div = self.builder.fdiv(left, right, name="div")
            floorf = self.module.declare_intrinsic('llvm.floor', [left.type])
            return self.builder.call(floorf, [div], name="floordiv")
        raise NotImplementedError(f"Float binary op: {op}")

    def _emit_int_binop(self, op: str, left, right) -> llvm_ir.Value:
        ops = {
            "+": self.builder.add,
            "-": self.builder.sub,
            "*": self.builder.mul,
            "//": self.builder.sdiv,
            "%": self.builder.srem,
            "<<": self.builder.shl,
            ">>": self.builder.ashr,
            "&": self.builder.and_,
            "|": self.builder.or_,
            "^": self.builder.xor,
        }
        if op in ops:
            return ops[op](left, right, name="binop")
        if op == "/":
            # Integer division → convert to float, divide, convert back
            f64_type = llvm_ir.DoubleType()
            fl = self.builder.sitofp(left, f64_type)
            fr = self.builder.sitofp(right, f64_type)
            return self.builder.fdiv(fl, fr, name="div")
        if op == "**":
            # Integer power: convert to float, use pow, convert back
            f64_type = llvm_ir.DoubleType()
            fl = self.builder.sitofp(left, f64_type)
            fr = self.builder.sitofp(right, f64_type)
            powf = self.module.declare_intrinsic('llvm.pow', [f64_type])
            result = self.builder.call(powf, [fl, fr], name="pow")
            return self.builder.fptosi(result, left.type, name="pow.int")
        raise NotImplementedError(f"Integer binary op: {op}")

    def _emit_unaryop(self, node: ir.IRUnaryOp) -> llvm_ir.Value:
        operand = self._emit_expr(node.operand)
        if node.op == "-":
            if _is_float_type(operand.type):
                return self.builder.fsub(
                    llvm_ir.Constant(operand.type, 0.0), operand, name="neg")
            return self.builder.neg(operand, name="neg")
        if node.op == "+":
            return operand
        if node.op == "not":
            operand = self._to_i1(operand)
            return self.builder.not_(operand, name="not")
        if node.op == "~":
            return self.builder.not_(operand, name="invert")
        raise NotImplementedError(f"Unary op: {node.op}")

    def _emit_compare(self, node: ir.IRCompare) -> llvm_ir.Value:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)
        left, right = self._coerce_pair(left, right)

        if _is_float_type(left.type):
            fcmp_ops = {
                "==": "==", "!=": "!=",
                "<": "<", "<=": "<=",
                ">": ">", ">=": ">=",
            }
            return self.builder.fcmp_ordered(fcmp_ops[node.op], left, right, name="cmp")

        if _is_int_type(left.type):
            icmp_ops = {
                "==": "==", "!=": "!=",
                "<": "<", "<=": "<=",
                ">": ">", ">=": ">=",
            }
            return self.builder.icmp_signed(icmp_ops[node.op], left, right, name="cmp")

        raise TypeError(f"Cannot compare type: {left.type}")

    def _emit_boolop(self, node: ir.IRBoolOp) -> llvm_ir.Value:
        """Emit boolean and/or (eager evaluation — not short-circuit)."""
        result = self._to_i1(self._emit_expr(node.values[0]))
        for val_node in node.values[1:]:
            val = self._to_i1(self._emit_expr(val_node))
            if node.op == "and":
                result = self.builder.and_(result, val, name="and")
            else:
                result = self.builder.or_(result, val, name="or")
        return result

    def _emit_field_load(self, node: ir.IRFieldLoad) -> llvm_ir.Value:
        base_ptr = self._emit_expr(node.field)
        index = self._to_i64(self._emit_expr(node.index))
        elem_ptr = self.builder.gep(base_ptr, [index], inbounds=True, name="load.ptr")
        return self.builder.load(elem_ptr, name="load.val", align=4)

    def _emit_texture_sample(self, node: ir.IRTextureSample) -> llvm_ir.Value:
        """Emit software trilinear interpolation for texture3d.sample()."""
        f32_type = llvm_ir.FloatType()
        i64_type = llvm_ir.IntType(64)
        W, H, D = node.shape

        # Get field pointer
        base_ptr = self._params[node.field_name]

        # Evaluate normalized [0,1] coordinates
        u = self._to_float(self._emit_expr(node.coords[0]))
        v = self._to_float(self._emit_expr(node.coords[1]))
        w = self._to_float(self._emit_expr(node.coords[2]))

        # Convert to texel coordinates: fx = u * (W - 1), etc.
        fx = self.builder.fmul(u, llvm_ir.Constant(f32_type, float(W - 1)), name="tex.fx")
        fy = self.builder.fmul(v, llvm_ir.Constant(f32_type, float(H - 1)), name="tex.fy")
        fz = self.builder.fmul(w, llvm_ir.Constant(f32_type, float(D - 1)), name="tex.fz")

        # Integer part (floor)
        floor_fn = self.module.declare_intrinsic("llvm.floor", [f32_type])
        ix_f = self.builder.call(floor_fn, [fx], name="tex.ix_f")
        iy_f = self.builder.call(floor_fn, [fy], name="tex.iy_f")
        iz_f = self.builder.call(floor_fn, [fz], name="tex.iz_f")

        # Fractional part
        dx = self.builder.fsub(fx, ix_f, name="tex.dx")
        dy = self.builder.fsub(fy, iy_f, name="tex.dy")
        dz = self.builder.fsub(fz, iz_f, name="tex.dz")

        # Convert to integer indices
        ix = self.builder.fptosi(ix_f, i64_type, name="tex.ix")
        iy = self.builder.fptosi(iy_f, i64_type, name="tex.iy")
        iz = self.builder.fptosi(iz_f, i64_type, name="tex.iz")

        # Clamp helper
        def clamp_idx(val, max_val, name):
            zero = llvm_ir.Constant(i64_type, 0)
            mx = llvm_ir.Constant(i64_type, max_val)
            c = self.builder.icmp_signed("<", val, zero)
            val = self.builder.select(c, zero, val, name=f"{name}.lo")
            c = self.builder.icmp_signed(">", val, mx)
            return self.builder.select(c, mx, val, name=f"{name}.hi")

        one = llvm_ir.Constant(i64_type, 1)
        ix0 = clamp_idx(ix, W - 1, "ix0")
        ix1 = clamp_idx(self.builder.add(ix, one), W - 1, "ix1")
        iy0 = clamp_idx(iy, H - 1, "iy0")
        iy1 = clamp_idx(self.builder.add(iy, one), H - 1, "iy1")
        iz0 = clamp_idx(iz, D - 1, "iz0")
        iz1 = clamp_idx(self.builder.add(iz, one), D - 1, "iz1")

        # Linear index: iz * (W * H) + iy * W + ix
        wh = llvm_ir.Constant(i64_type, W * H)
        w_const = llvm_ir.Constant(i64_type, W)

        def linear(xi, yi, zi):
            return self.builder.add(
                self.builder.add(self.builder.mul(zi, wh), self.builder.mul(yi, w_const)),
                xi)

        def load_at(xi, yi, zi, name):
            idx = linear(xi, yi, zi)
            ptr = self.builder.gep(base_ptr, [idx], inbounds=True)
            return self.builder.load(ptr, name=name, align=4)

        # 8 corner values
        c000 = load_at(ix0, iy0, iz0, "c000")
        c100 = load_at(ix1, iy0, iz0, "c100")
        c010 = load_at(ix0, iy1, iz0, "c010")
        c110 = load_at(ix1, iy1, iz0, "c110")
        c001 = load_at(ix0, iy0, iz1, "c001")
        c101 = load_at(ix1, iy0, iz1, "c101")
        c011 = load_at(ix0, iy1, iz1, "c011")
        c111 = load_at(ix1, iy1, iz1, "c111")

        # Trilinear interpolation
        one_f = llvm_ir.Constant(f32_type, 1.0)
        inv_dx = self.builder.fsub(one_f, dx)
        inv_dy = self.builder.fsub(one_f, dy)
        inv_dz = self.builder.fsub(one_f, dz)

        def lerp(a, b, t, inv_t, name):
            return self.builder.fadd(
                self.builder.fmul(a, inv_t),
                self.builder.fmul(b, t), name=name)

        c00 = lerp(c000, c100, dx, inv_dx, "c00")
        c10 = lerp(c010, c110, dx, inv_dx, "c10")
        c01 = lerp(c001, c101, dx, inv_dx, "c01")
        c11 = lerp(c011, c111, dx, inv_dx, "c11")
        c0 = lerp(c00, c10, dy, inv_dy, "c0")
        c1 = lerp(c01, c11, dy, inv_dy, "c1")
        return lerp(c0, c1, dz, inv_dz, "tex.result")

    def _emit_attribute(self, node: ir.IRAttribute) -> llvm_ir.Value:
        # x.shape[0] is resolved at compile time via type info
        # __len__ is also resolved at compile time
        # For now, these are errors — they should be resolved before codegen
        raise NotImplementedError(
            f"Attribute access '{node.attr}' should be resolved before LLVM codegen. "
            "Use constant folding or resolve at call time."
        )

    def _emit_call(self, node: ir.IRCall) -> llvm_ir.Value:
        """Emit a math builtin call."""
        args = [self._emit_expr(a) for a in node.args]

        # min/max with two args
        if node.func_name in ("min", "max") and len(args) == 2:
            return self._emit_minmax(node.func_name, args[0], args[1])

        # abs
        if node.func_name == "abs" and len(args) == 1:
            return self._emit_abs(args[0])

        # LLVM intrinsics (single-arg math functions)
        intrinsic_map = {
            "sqrt": "llvm.sqrt",
            "sin": "llvm.sin",
            "cos": "llvm.cos",
            "exp": "llvm.exp",
            "exp2": "llvm.exp2",
            "log": "llvm.log",
            "log2": "llvm.log2",
            "log10": "llvm.log10",
            "floor": "llvm.floor",
            "ceil": "llvm.ceil",
            "fabs": "llvm.fabs",
        }

        if node.func_name in intrinsic_map:
            arg = args[0]
            arg = self._to_float(arg)
            intrinsic = self.module.declare_intrinsic(
                intrinsic_map[node.func_name], [arg.type])
            return self.builder.call(intrinsic, [arg], name=node.func_name)

        # pow(x, y)
        if node.func_name == "pow" and len(args) == 2:
            a, b = args
            a = self._to_float(a)
            b = self._coerce_to(b, a.type)
            powf = self.module.declare_intrinsic('llvm.pow', [a.type])
            return self.builder.call(powf, [a, b], name="pow")

        # Trig functions not in llvm intrinsics — use libm
        libm_funcs = {"tan", "asin", "acos", "atan", "atan2"}
        if node.func_name in libm_funcs:
            return self._emit_libm_call(node.func_name, args)

        raise NotImplementedError(f"Builtin function: {node.func_name}")

    def _emit_minmax(self, func_name: str, a: llvm_ir.Value, b: llvm_ir.Value) -> llvm_ir.Value:
        a, b = self._coerce_pair(a, b)
        if _is_float_type(a.type):
            intrinsic_name = "llvm.minnum" if func_name == "min" else "llvm.maxnum"
            # declare_intrinsic generates wrong signature for minnum/maxnum,
            # so declare manually with two parameters
            type_suffix = "f32" if isinstance(a.type, llvm_ir.FloatType) else "f64"
            full_name = f"{intrinsic_name}.{type_suffix}"
            try:
                fn = self.module.get_global(full_name)
            except KeyError:
                fn_type = llvm_ir.FunctionType(a.type, [a.type, a.type])
                fn = llvm_ir.Function(self.module, fn_type, name=full_name)
            return self.builder.call(fn, [a, b], name=func_name)
        # Integer min/max via comparison
        if func_name == "min":
            cond = self.builder.icmp_signed("<", a, b, name="min.cond")
        else:
            cond = self.builder.icmp_signed(">", a, b, name="max.cond")
        return self.builder.select(cond, a, b, name=func_name)

    def _emit_abs(self, val: llvm_ir.Value) -> llvm_ir.Value:
        if _is_float_type(val.type):
            fabs = self.module.declare_intrinsic('llvm.fabs', [val.type])
            return self.builder.call(fabs, [val], name="abs")
        # Integer abs: (x ^ (x >> 31)) - (x >> 31) for i32
        bits = val.type.width
        shift = llvm_ir.Constant(val.type, bits - 1)
        sign = self.builder.ashr(val, shift, name="sign")
        xored = self.builder.xor(val, sign, name="xored")
        return self.builder.sub(xored, sign, name="abs")

    def _emit_libm_call(self, name: str, args: list[llvm_ir.Value]) -> llvm_ir.Value:
        """Emit a call to a libm function (linked at runtime)."""
        args = [self._to_float(a) for a in args]
        # Use f64 for libm
        f64_type = llvm_ir.DoubleType()
        args = [self._coerce_to(a, f64_type) for a in args]
        fn_type = llvm_ir.FunctionType(f64_type, [f64_type] * len(args))
        fn = self.module.declare_intrinsic(f'llvm.{name}' if False else '', [])
        # Actually, use a regular external function declaration for libm
        fn_name = name
        try:
            fn = self.module.get_global(fn_name)
        except KeyError:
            fn = llvm_ir.Function(self.module, fn_type, name=fn_name)
        return self.builder.call(fn, args, name=f"{name}.result")

    def _emit_cast(self, node: ir.IRCast) -> llvm_ir.Value:
        val = self._emit_expr(node.value)
        if node.dtype == "int":
            if _is_float_type(val.type):
                return self.builder.fptosi(val, llvm_ir.IntType(64), name="toint")
            return val
        if node.dtype == "float":
            if _is_int_type(val.type):
                return self.builder.sitofp(val, llvm_ir.FloatType(), name="tofloat")
            return val
        raise NotImplementedError(f"Cast to {node.dtype}")

    # --- Type coercion helpers ---

    def _to_i64(self, val: llvm_ir.Value) -> llvm_ir.Value:
        """Convert a value to i64 (for indexing)."""
        i64_type = llvm_ir.IntType(64)
        if val.type == i64_type:
            return val
        if _is_int_type(val.type):
            if val.type.width < 64:
                return self.builder.sext(val, i64_type, name="to.i64")
            return self.builder.trunc(val, i64_type, name="to.i64")
        if _is_float_type(val.type):
            return self.builder.fptosi(val, i64_type, name="to.i64")
        raise TypeError(f"Cannot convert {val.type} to i64")

    def _to_i1(self, val: llvm_ir.Value) -> llvm_ir.Value:
        """Convert a value to i1 (boolean) for branching."""
        i1_type = llvm_ir.IntType(1)
        if val.type == i1_type:
            return val
        if _is_int_type(val.type):
            return self.builder.icmp_signed("!=", val,
                                            llvm_ir.Constant(val.type, 0), name="to.bool")
        if _is_float_type(val.type):
            return self.builder.fcmp_ordered("!=", val,
                                             llvm_ir.Constant(val.type, 0.0), name="to.bool")
        raise TypeError(f"Cannot convert {val.type} to i1")

    def _to_float(self, val: llvm_ir.Value) -> llvm_ir.Value:
        """Ensure value is a float type."""
        if _is_float_type(val.type):
            return val
        if _is_int_type(val.type):
            return self.builder.sitofp(val, llvm_ir.FloatType(), name="to.float")
        raise TypeError(f"Cannot convert {val.type} to float")

    def _coerce_to(self, val: llvm_ir.Value, target: llvm_ir.Type) -> llvm_ir.Value:
        """Coerce a value to a target LLVM type."""
        if val.type == target:
            return val

        # float -> float (widen/narrow)
        if _is_float_type(val.type) and _is_float_type(target):
            if isinstance(target, llvm_ir.DoubleType):
                return self.builder.fpext(val, target, name="fpext")
            return self.builder.fptrunc(val, target, name="fptrunc")

        # int -> float
        if _is_int_type(val.type) and _is_float_type(target):
            return self.builder.sitofp(val, target, name="sitofp")

        # float -> int
        if _is_float_type(val.type) and _is_int_type(target):
            return self.builder.fptosi(val, target, name="fptosi")

        # int -> int (widen/narrow)
        if _is_int_type(val.type) and _is_int_type(target):
            if val.type.width < target.width:
                return self.builder.sext(val, target, name="sext")
            return self.builder.trunc(val, target, name="trunc")

        raise TypeError(f"Cannot coerce {val.type} to {target}")

    def _coerce_pair(self, a: llvm_ir.Value, b: llvm_ir.Value):
        """Coerce two values to a common type (type promotion)."""
        if a.type == b.type:
            return a, b

        # Float wins over int
        if _is_float_type(a.type) and _is_int_type(b.type):
            return a, self._coerce_to(b, a.type)
        if _is_int_type(a.type) and _is_float_type(b.type):
            return self._coerce_to(a, b.type), b

        # Wider float wins
        if _is_float_type(a.type) and _is_float_type(b.type):
            if isinstance(a.type, llvm_ir.DoubleType):
                return a, self._coerce_to(b, a.type)
            return self._coerce_to(a, b.type), b

        # Wider int wins
        if _is_int_type(a.type) and _is_int_type(b.type):
            if a.type.width > b.type.width:
                return a, self._coerce_to(b, a.type)
            return self._coerce_to(a, b.type), b

        raise TypeError(f"Cannot coerce pair: {a.type}, {b.type}")


def generate_llvm_ir(ir_func: ir.IRFunction) -> llvm_ir.Module:
    """Generate LLVM IR for a single kernel function."""
    codegen = LLVMCodeGen(ir_func)
    return codegen.generate()

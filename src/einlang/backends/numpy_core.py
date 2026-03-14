"""NumPy backend core: execute, env scope stack only (no global table)."""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Union

import numpy as np

from ..backends.base import Backend
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR, BindingIR,
    LiteralIR, FunctionCallIR, IRVisitor,
    BlockExpressionIR, RectangularAccessIR, LoweredReductionIR, LoweredComprehensionIR,
    IfExpressionIR, BinaryOpIR, UnaryOpIR,
    is_einstein_binding, is_function_binding,
)
from ..shared.defid import DefId, Resolver, FIXED_BUILTIN_ORDER, _BUILTIN_CRATE
from ..runtime.environment import ExecutionEnvironment, FunctionValue
from ..runtime.runtime import ExecutionResult
from .numpy_helpers import (
    _reject_non_lowered,
    builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
    builtin_shape, builtin_sum, builtin_max, builtin_min,
)


def _register_fixed_builtins(env: ExecutionEnvironment) -> None:
    fns = (
        builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
        builtin_shape, builtin_sum, builtin_max, builtin_min,
    )
    for i, fn in enumerate(fns):
        if i < len(FIXED_BUILTIN_ORDER):
            env.set_value(DefId(krate=_BUILTIN_CRATE, index=i), fn)


def _single_array_param(func_def: Any) -> bool:
    """True if function has exactly one parameter (typical for array-in, scalar/array-out)."""
    params = func_def.parameters
    return params is not None and len(params) == 1


class _ContainsReductionWithOpVisitor(IRVisitor[bool]):
    """True if expression contains a LoweredReductionIR with the given operation."""

    def __init__(self, operation: str) -> None:
        self._op = operation

    def _default(self) -> bool:
        return False

    def visit_lowered_reduction(self, node: LoweredReductionIR) -> bool:
        return node.operation == self._op

    def visit_binary_op(self, node: BinaryOpIR) -> bool:
        return (node.left and node.left.accept(self)) or (node.right and node.right.accept(self))

    def visit_unary_op(self, node: UnaryOpIR) -> bool:
        return node.operand is not None and node.operand.accept(self)

    def visit_block_expression(self, node: BlockExpressionIR) -> bool:
        for stmt in (node.statements or []):
            if stmt.accept(self):
                return True
        return (node.final_expr is not None and node.final_expr.accept(self))

    def visit_if_expression(self, node: IfExpressionIR) -> bool:
        return (
            (node.condition and node.condition.accept(self))
            or (node.then_expr and node.then_expr.accept(self))
            or (node.else_expr and node.else_expr.accept(self))
        )

    def visit_binding(self, node: BindingIR) -> bool:
        expr = node.expr
        return expr is not None and expr.accept(self)

    def visit_literal(self, node: LiteralIR) -> bool:
        return self._default()

    def visit_identifier(self, node: Any) -> bool:
        return self._default()

    def visit_rectangular_access(self, node: RectangularAccessIR) -> bool:
        return self._default()

    def visit_function_call(self, node: FunctionCallIR) -> bool:
        return self._default()

    def visit_jagged_access(self, node: Any) -> bool:
        return self._default()

    def visit_lambda(self, node: Any) -> bool:
        return self._default()

    def visit_range(self, node: Any) -> bool:
        return self._default()

    def visit_array_comprehension(self, node: Any) -> bool:
        return self._default()

    def visit_module(self, node: Any) -> bool:
        return self._default()

    def visit_array_literal(self, node: Any) -> bool:
        return self._default()

    def visit_tuple_expression(self, node: Any) -> bool:
        return self._default()

    def visit_tuple_access(self, node: Any) -> bool:
        return self._default()

    def visit_interpolated_string(self, node: Any) -> bool:
        return self._default()

    def visit_cast_expression(self, node: Any) -> bool:
        return self._default()

    def visit_member_access(self, node: Any) -> bool:
        return self._default()

    def visit_try_expression(self, node: Any) -> bool:
        return self._default()

    def visit_match_expression(self, node: Any) -> bool:
        return self._default()

    def visit_reduction_expression(self, node: Any) -> bool:
        return self._default()

    def visit_where_expression(self, node: Any) -> bool:
        return self._default()

    def visit_pipeline_expression(self, node: Any) -> bool:
        return self._default()

    def visit_builtin_call(self, node: Any) -> bool:
        return self._default()

    def visit_literal_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_identifier_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_tuple_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_array_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_rest_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_guard_pattern(self, node: Any) -> bool:
        return self._default()

    def visit_program(self, node: Any) -> bool:
        return self._default()


def _body_has_reduction(body: Any, operation: str) -> bool:
    """True if body (block or binding list) contains a LoweredReductionIR with the given operation."""
    statements = (body.statements or []) if hasattr(body, "statements") else []
    visitor = _ContainsReductionWithOpVisitor(operation)
    for stmt in statements:
        if not isinstance(stmt, BindingIR):
            continue
        expr = stmt.expr
        if expr is None:
            continue
        try:
            if expr.accept(visitor):
                return True
        except NotImplementedError:
            pass
    return False


def _check_block_is_index_of_extremum(body: BlockExpressionIR) -> Optional[str]:
    """If this block implements index-of-max or index-of-min, return 'argmax' or 'argmin'; else None."""
    if body.final_expr is None:
        return None
    statements = body.statements or []
    final_expr = body.final_expr
    has_max = _body_has_reduction(body, "max")
    has_min = _body_has_reduction(body, "min")
    has_comprehension = any(
        isinstance(s.expr, LoweredComprehensionIR)
        for s in statements if isinstance(s, BindingIR)
    )
    first_elem_ok = False
    if isinstance(final_expr, RectangularAccessIR):
        indices = final_expr.indices or []
        if len(indices) == 1:
            idx = indices[0]
            if isinstance(idx, LiteralIR):
                try:
                    if int(idx.value) == 0:
                        first_elem_ok = True
                except (TypeError, ValueError):
                    pass
    final_is_min_red = isinstance(final_expr, LoweredReductionIR) and final_expr.operation == "min"
    final_is_max_red = isinstance(final_expr, LoweredReductionIR) and final_expr.operation == "max"
    if (has_max and not has_min) and (first_elem_ok and has_comprehension or final_is_min_red):
        return "argmax"
    if (has_min and not has_max) and (first_elem_ok and has_comprehension or final_is_max_red):
        return "argmin"
    return None


def _detect_index_of_extremum(func_def: Any) -> Optional[str]:
    """
    General pattern: function returns index (or indices) of the maximum or minimum of its
    single array argument. Matches multiple implementations and shapes:
    - Block: comprehension + first element, or sentinel + min/max reduction.
    - If/else body (e.g. 1D vs 2D): check both branches so either path can be optimized.
    Returns "argmax", "argmin", or None.
    """
    if not _single_array_param(func_def):
        return None
    body = func_def.body
    if body is None:
        return None
    candidates = []
    if isinstance(body, BlockExpressionIR):
        candidates.append(body)
    elif isinstance(body, IfExpressionIR):
        if body.then_expr is not None:
            candidates.append(body.then_expr)
        if body.else_expr is not None:
            candidates.append(body.else_expr)
    for block in candidates:
        if isinstance(block, BlockExpressionIR):
            out = _check_block_is_index_of_extremum(block)
            if out is not None:
                return out
    return None


def _numpy_optimized_dispatch(func_def: Any, args: List[Any]) -> Optional[Any]:
    """
    If func_def matches a known pattern that NumPy can implement in one pass,
    return the result; otherwise return None (caller runs the body).
    Extensible: add more (detector, handler) pairs for sum, mean, etc.
    """
    if len(args) != 1 or not isinstance(args[0], np.ndarray):
        return None
    a = np.asarray(args[0])
    # Index-of-extremum pattern (argmax/argmin in any form)
    key = _detect_index_of_extremum(func_def)
    if key == "argmax":
        if a.ndim == 1:
            return int(np.argmax(a))
        return np.argmax(a, axis=-1)
    if key == "argmin":
        if a.ndim == 1:
            return int(np.argmin(a))
        return np.argmin(a, axis=-1)
    return None


class CoreExecutionMixin:
    """Execute, env scope stack only. No def_table, no builtin_defids; all in env."""

    def __init__(self):
        self.env: ExecutionEnvironment = ExecutionEnvironment()
        _register_fixed_builtins(self.env)
        self.resolver: Optional[Resolver] = None

    def execute(
        self,
        program: ProgramIR,
        inputs: Optional[Dict[str, Any]] = None,
        resolver: Optional[Resolver] = None,
        tcx: Optional[Any] = None,
        *,
        input_by_defid: Optional[Dict[DefId, Any]] = None,
        main_defid: Optional[DefId] = None,
        entry_source_file: Optional[str] = None,
    ) -> Any:
        from ..shared.defid import DefType
        from ..runtime import set_entry_file
        # Python: set __file__ (entry file path). Rust: set env var (like CARGO_MANIFEST_DIR).
        if entry_source_file and entry_source_file not in ("<inline>", "<stdin>"):
            set_entry_file(entry_source_file)
            try:
                os.environ["EINLANG_SCRIPT_DIR"] = os.path.dirname(
                    os.path.abspath(entry_source_file)
                )
            except Exception:
                os.environ["EINLANG_SCRIPT_DIR"] = os.getcwd()
        else:
            set_entry_file(None)
            os.environ["EINLANG_SCRIPT_DIR"] = os.getcwd()
        self.resolver = resolver
        self._tcx = tcx
        self.env = ExecutionEnvironment()
        _register_fixed_builtins(self.env)
        for func in program.functions:
            if func.defid:
                self.env.set_value(func.defid, func, name=func.name)
        for mod in (program.modules or []):
            for func in self._collect_module_functions(mod):
                if func.defid:
                    self.env.set_value(func.defid, func, name=func.name)
        if tcx:
            function_ir_map = getattr(tcx, "function_ir_map", None)
            if function_ir_map:
                for func in function_ir_map.values():
                    if func is not None and is_function_binding(func) and func.defid:
                        self.env.set_value(func.defid, func, name=func.name)
        if resolver:
            for defid, (def_type, definition) in resolver._def_registry.items():
                if def_type == DefType.BUILTIN:
                    key = DefId(krate=defid.krate, index=defid.index)
                    self.env.set_value(key, definition)
        if input_by_defid:
            for defid, value in input_by_defid.items():
                self.env.set_value(defid, value)
        bucket_size = int(os.environ.get("EINLANG_PROFILE_LINES", "0") or "0")
        self._profile_bucket_size = bucket_size
        self._profile_buckets = {} if bucket_size > 0 else None
        self._einstein_vectorized = 0
        self._einstein_scalar = 0
        self._einstein_hybrid = 0
        self._einstein_call_scalar = 0
        profile_statements = bool(os.environ.get("EINLANG_PROFILE_STATEMENTS", ""))
        self._profile_statements = profile_statements
        profile_blocks = bool(os.environ.get("EINLANG_PROFILE_BLOCKS", ""))
        self._profile_blocks = profile_blocks
        profile_functions = bool(os.environ.get("EINLANG_PROFILE_FUNCTIONS", ""))
        self._profile_functions = profile_functions
        self._profile_fn_times: Dict[str, float] = {} if profile_functions else {}
        try:
            if main_defid:
                main_func = self.env.get_value(main_defid)
                if main_func is not None:
                    result_value = self._call_function(main_func, [])
                    if self._profile_fn_times:
                        print("[profile] === per-function total (s) ===", flush=True)
                        for name, total in sorted(self._profile_fn_times.items(), key=lambda x: -x[1]):
                            if total > 0.01:
                                print(f"  {name}: {total:.2f}", flush=True)
                    if os.environ.get("EINLANG_DEBUG_VECTORIZE", "").strip().lower() in ("1", "true", "yes"):
                        v = getattr(self, "_einstein_vectorized", 0)
                        s = getattr(self, "_einstein_scalar", 0)
                        h = getattr(self, "_einstein_hybrid", 0)
                        c = getattr(self, "_einstein_call_scalar", 0)
                        total = v + s + h + c
                        sys.stderr.write(
                            f"[vectorize] Einstein clauses: {v} vectorized, {s} scalar, {h} hybrid, {c} call-scalar (total {total})\n"
                        )
                    return ExecutionResult(value=result_value)
            outputs = {}
            if program.statements:
                with self.env.scope():
                    for stmt_index, stmt in enumerate(program.statements):
                        if stmt is None:
                            raise ValueError("IR statement is None")
                        if profile_statements:
                            if self._profile_buckets is not None:
                                self._profile_buckets = {}
                            self._stmt_t0 = time.perf_counter()
                        result_value = stmt.accept(self)
                        variable_defid = None
                        if isinstance(stmt, BindingIR) and not is_function_binding(stmt):
                            variable_defid = stmt.defid
                        if variable_defid is None:
                            binding = getattr(stmt, "_binding", None)
                            if binding is not None and isinstance(binding, BindingIR):
                                variable_defid = binding.defid
                        if variable_defid is not None:
                            var_name = stmt.name if isinstance(stmt, BindingIR) else (getattr(stmt, "_binding", None).name if getattr(stmt, "_binding", None) else None)
                            self.env.set_value(variable_defid, result_value, name=var_name)
                            outputs[variable_defid] = result_value
                        if profile_statements:
                            elapsed = time.perf_counter() - self._stmt_t0
                            line = (stmt.location.line if stmt.location else None) or "?"
                            name = stmt.name if isinstance(stmt, BindingIR) else ""
                            print(f"[profile] stmt {stmt_index} (L{line}) {name}: {elapsed:.2f}s", flush=True)
                            if self._profile_buckets is not None and self._profile_buckets:
                                size = self._profile_bucket_size
                                for lo in sorted(self._profile_buckets.keys()):
                                    print(f"  L{lo}-L{lo + size}: {self._profile_buckets[lo]:.2f}s", flush=True)
                                self._profile_buckets = {}
                    for defid, value in self.env.get_current_scope().items():
                        if defid not in outputs:
                            outputs[defid] = value
            if self._profile_fn_times:
                print("[profile] === per-function total (s) ===", flush=True)
                for name, total in sorted(self._profile_fn_times.items(), key=lambda x: -x[1]):
                    if total > 0.001:
                        print(f"  {name}: {total:.2f}", flush=True)
            if self._profile_buckets is not None and self._profile_buckets and not profile_statements:
                size = self._profile_bucket_size
                for lo in sorted(self._profile_buckets.keys()):
                    print(f"[profile] L{lo}-L{lo + size}: {self._profile_buckets[lo]:.2f}s", flush=True)
            if os.environ.get("EINLANG_DEBUG_VECTORIZE", "").strip().lower() in ("1", "true", "yes"):
                v = getattr(self, "_einstein_vectorized", 0)
                s = getattr(self, "_einstein_scalar", 0)
                h = getattr(self, "_einstein_hybrid", 0)
                c = getattr(self, "_einstein_call_scalar", 0)
                total = v + s + h + c
                sys.stderr.write(
                    f"[vectorize] Einstein clauses: {v} vectorized, {s} scalar, {h} hybrid, {c} call-scalar (total {total})\n"
                )
            return ExecutionResult(outputs=outputs)
        except Exception as e:
            from ..shared.errors import EinlangSourceError
            if isinstance(e, EinlangSourceError):
                return ExecutionResult(error=e)
            return ExecutionResult(error=RuntimeError(str(e)))

    def execute_expression(self, expr: ExpressionIR, env: Dict[DefId, Any]) -> Any:
        with self.env.scope():
            for defid, value in env.items():
                self.env.set_value(defid, value)
            return expr.accept(self)

    def _collect_module_functions(self, mod: Any) -> List[FunctionDefIR]:
        from ..ir.nodes import ModuleIR
        if not isinstance(mod, ModuleIR):
            return []
        result = list(mod.functions or [])
        for sub in (mod.submodules or []):
            result.extend(self._collect_module_functions(sub))
        return result

    def _call_function(self, func_def: Union[FunctionDefIR, Any], args: List[Any]) -> Any:
        # func_def may be BindingIR (named function) or FunctionValueIR (lambda)
        params = func_def.parameters
        body = func_def.body
        name = (func_def.name if hasattr(func_def, "name") else None) or "<lambda>"
        expected = len(params)
        actual = len(args)
        if actual != expected:
            raise RuntimeError(f"Function '{name}' expects {expected} argument(s), got {actual}")
        # General NumPy fast path: dispatch by body pattern (index-of-extremum, etc.)
        result = _numpy_optimized_dispatch(func_def, args)
        if result is not None:
            return result
        with self.env.scope():
            for param, arg_value in zip(params, args):
                if param.defid is None:
                    raise RuntimeError(f"Parameter has no defid; cannot bind. Name: {param.name}")
                self.env.set_value(param.defid, arg_value, name=param.name)
            if getattr(self, "_profile_functions", False):
                t0 = time.perf_counter()
                result = body.accept(self)
                elapsed = time.perf_counter() - t0
                self._profile_fn_times[name] = self._profile_fn_times.get(name, 0.0) + elapsed
                if elapsed > 0.01:
                    print(f"[profile] fn {name}: {elapsed:.2f}s", flush=True)
                return result
            return body.accept(self)

    def codegen(self, program: ProgramIR) -> str:
        return "# NumPy code generation not yet implemented"

    def visit_program(self, node: ProgramIR) -> Any:
        results = []
        for stmt in node.statements:
            results.append(stmt.accept(self))
        return results[-1] if results else None

    def visit_module(self, node: Any) -> Any:
        raise NotImplementedError("Module execution not yet implemented")

    def visit_binding(self, node: Any) -> Any:
        if is_function_binding(node):
            if node.defid:
                self.env.set_value(node.defid, node, name=node.name)
            return None
        if is_einstein_binding(node):
            from ..ir.nodes import LoweredEinsteinIR, LoweredRecurrenceIR
            expr = node.expr
            if not isinstance(expr, (LoweredEinsteinIR, LoweredRecurrenceIR)):
                raise RuntimeError(
                    f"Non-lowered EinsteinDeclaration reached backend. "
                    f"EinsteinLoweringPass must run before codegen. (node type: {type(node).__name__})"
                )
        from ..ir.nodes import LoweredEinsteinIR, LoweredRecurrenceIR
        expr = node.expr
        if isinstance(expr, (LoweredEinsteinIR, LoweredRecurrenceIR)):
            if expr is None:
                return None
            stack = getattr(self, "_variable_decl_stack", None)
            if stack is None:
                self._variable_decl_stack = []
                stack = self._variable_decl_stack
            stack.append(node)
            try:
                result = node.expr.accept(self)
            finally:
                stack.pop()
            if node.defid is not None:
                self.env.set_value(node.defid, result, name=node.name)
            return result
        value = node.expr.accept(self)
        if node.defid is not None:
            self.env.set_value(node.defid, value, name=node.name)
        return value

    def visit_literal_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_identifier_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_wildcard_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_tuple_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_array_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_rest_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_guard_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_or_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_constructor_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_binding_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_range_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")

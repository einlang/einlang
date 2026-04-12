"""NumPy backend core: execute, env scope stack only (no global table)."""

from contextlib import contextmanager
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, BindingIR,
    LiteralIR, FunctionCallIR,
    BlockExpressionIR, RectangularAccessIR,
    IdentifierIR,
    LoweredReductionIR, LoweredSelectAtArgmaxIR,
    is_einstein_binding, is_function_binding,
)
from ..shared.defid import DefId, Resolver, FIXED_BUILTIN_ORDER, _BUILTIN_CRATE
from ..shared.debug_trace import emit_debug_log
from ..runtime.environment import ExecutionEnvironment
from ..runtime.runtime import ExecutionResult
from .numpy_helpers import (
    NumPyVectorizationState,
    builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
    builtin_shape, builtin_sum, builtin_max, builtin_min,
)


def _debug_value_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"shape": None, "value": None}
    try:
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=object)
        else:
            arr = np.asarray(value)
        return {"shape": list(arr.shape), "value": arr.tolist()}
    except Exception:
        return {"shape": None, "value": repr(value)}


def _call_arguments(call: FunctionCallIR) -> List[Any]:
    return list(getattr(call, "arguments", []) or [])


def _int_list_from_expr(expr: Any, backend: Any) -> List[int]:
    return [int(v) for v in np.asarray(expr.accept(backend)).reshape(-1)]


def _register_fixed_builtins(env: ExecutionEnvironment) -> None:
    fns = (
        builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
        builtin_shape, builtin_sum, builtin_max, builtin_min,
    )
    for i, fn in enumerate(fns):
        if i < len(FIXED_BUILTIN_ORDER):
            env.set_value(DefId(krate=_BUILTIN_CRATE, index=i), fn)


def _resolved_call_name(
    call: FunctionCallIR,
    function_ir_map: Dict[Any, Any],
    mono: Optional[Any],
) -> Optional[str]:
    """Resolve a call name through monomorphization when available."""
    function_defid = getattr(call, "function_defid", None)
    callee = function_ir_map.get(function_defid)
    generic_callee = None
    if mono is not None and function_defid is not None:
        generic_defid = mono.get_generic_defid_for_specialized(function_defid)
        if generic_defid is not None:
            generic_callee = function_ir_map.get(generic_defid)
    name = getattr(generic_callee, "name", None) or getattr(callee, "name", None)
    return name if isinstance(name, str) else None


class CoreExecutionMixin:
    """Execute, env scope stack only. No def_table, no builtin_defids; all in env."""

    def __init__(self):
        self.env: ExecutionEnvironment = ExecutionEnvironment()
        _register_fixed_builtins(self.env)
        self.resolver: Optional[Resolver] = None
        self._vectorization_state = NumPyVectorizationState()

    def _get_vectorization_state(self) -> NumPyVectorizationState:
        state = getattr(self, "_vectorization_state", None)
        if state is None:
            state = NumPyVectorizationState()
            self._vectorization_state = state
        return state

    @contextmanager
    def _vectorization_scope(self, **updates: Any):
        state = self._get_vectorization_state()
        saved = {key: getattr(state, key) for key in updates}
        for key, value in updates.items():
            setattr(state, key, value)
        try:
            yield state
        finally:
            for key, value in saved.items():
                setattr(state, key, value)

    def _vectorization_parallel_shape(self) -> Optional[Tuple[int, ...]]:
        return self._get_vectorization_state().parallel_shape

    def _vectorization_parallel_defids_order(self) -> Optional[Tuple[Any, ...]]:
        return self._get_vectorization_state().parallel_defids_order

    def _vectorization_safe_oob_enabled(self) -> bool:
        return self._get_vectorization_state().safe_oob

    def _in_recurrence_vectorization_clause(self) -> bool:
        return self._get_vectorization_state().recurrence_clause

    def _vectorize_debug_enabled(self) -> bool:
        value = os.environ.get("EINLANG_DEBUG_VECTORIZE", "").strip().lower()
        return value in ("1", "true", "yes", "2", "verbose", "all", "detail")

    def _print_vectorize_summary(self) -> None:
        v = getattr(self, "_einstein_vectorized", 0)
        s = getattr(self, "_einstein_scalar", 0)
        h = getattr(self, "_einstein_hybrid", 0)
        c = getattr(self, "_einstein_call_scalar", 0)
        total = v + s + h + c
        print(
            f"[vectorize] Einstein clauses: {v} vectorized, {s} scalar, {h} hybrid, {c} call-scalar (total {total})",
            flush=True,
        )

    def _statement_profile_label(self, stmt_index: int, stmt: Any) -> str:
        line = (stmt.location.line if stmt.location else None) or "?"
        name = stmt.name if isinstance(stmt, BindingIR) else ""
        return f"[profile] stmt {stmt_index} (L{line}) {name}:"

    def _print_statement_profile(self, stmt_index: int, stmt: Any, elapsed: float) -> None:
        print(f"{self._statement_profile_label(stmt_index, stmt)} {elapsed:.2f}s", flush=True)

    def _store_output_value(
        self,
        outputs: Dict[DefId, Any],
        defid: Optional[DefId],
        value: Any,
        *,
        name: Optional[str] = None,
    ) -> None:
        if defid is None:
            return
        if getattr(value, "is_circular_recurrence_buffer", False):
            value = value.materialize()
        self.env.set_value(defid, value, name=name)
        outputs[defid] = value

    def _print_function_profile_report(self, threshold: float) -> None:
        if not self._profile_fn_times:
            return
        print("[profile] === per-function total (s) ===", flush=True)
        for name, total in sorted(self._profile_fn_times.items(), key=lambda x: -x[1]):
            if total > threshold:
                print(f"  {name}: {total:.2f}", flush=True)

    def _initialize_execution_state(self) -> None:
        bucket_size = int(os.environ.get("EINLANG_PROFILE_LINES", "0") or "0")
        self._profile_bucket_size = bucket_size
        self._profile_buckets = {} if bucket_size > 0 else None
        self._einstein_vectorized = 0
        self._einstein_scalar = 0
        self._einstein_hybrid = 0
        self._einstein_call_scalar = 0
        self._profile_statements = bool(os.environ.get("EINLANG_PROFILE_STATEMENTS", ""))
        self._profile_blocks = bool(os.environ.get("EINLANG_PROFILE_BLOCKS", ""))
        profile_functions = bool(os.environ.get("EINLANG_PROFILE_FUNCTIONS", ""))
        self._profile_functions = profile_functions
        self._profile_fn_times = {} if profile_functions else {}
        self._differential_buffers = {}
        self._vectorization_state = NumPyVectorizationState()

    def _register_program_functions(self, program: ProgramIR, tcx: Optional[Any]) -> None:
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

    def _register_builtin_definitions(self, resolver: Optional[Resolver]) -> None:
        from ..shared.defid import DefType

        if resolver is None:
            return
        for defid, (def_type, definition) in resolver._def_registry.items():
            if def_type == DefType.BUILTIN:
                key = DefId(krate=defid.krate, index=defid.index)
                self.env.set_value(key, definition)

    def _load_inputs_by_defid(self, input_by_defid: Optional[Dict[DefId, Any]]) -> None:
        if not input_by_defid:
            return
        for defid, value in input_by_defid.items():
            self.env.set_value(defid, value)

    def _binding_output_defid(self, stmt: Any) -> tuple:
        binding = None
        variable_defid = None
        if isinstance(stmt, BindingIR) and not is_function_binding(stmt):
            variable_defid = stmt.defid
        if variable_defid is None:
            binding = getattr(stmt, "_binding", None)
            if binding is not None and isinstance(binding, BindingIR):
                variable_defid = binding.defid
        return variable_defid, binding

    def _execute_lowered_binding_expr(self, node: Any, expr: Any) -> Any:
        if expr is None:
            return None
        stack = getattr(self, "_variable_decl_stack", None)
        if stack is None:
            self._variable_decl_stack = []
            stack = self._variable_decl_stack
        nested_under_vectorized_binding = bool(stack) and self._vectorization_parallel_shape() is not None
        stack.append(node)
        try:
            if nested_under_vectorized_binding and isinstance(
                expr,
                (LoweredReductionIR, LoweredSelectAtArgmaxIR),
            ):
                # Ordinary scalar/tensor expressions inside a vectorized binding should
                # keep NumPy broadcasting. Routing them through per-outer-slot scalar
                # evaluation silently turns vectorized code (for example GELU's local
                # `inner` binding) into Python loops over every outer index.
                slot_eval = self._evaluate_lowered_per_outer_slot(expr)
                if slot_eval is not None:
                    return slot_eval
            return expr.accept(self)
        finally:
            stack.pop()

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
        self._register_program_functions(program, tcx)
        self._register_builtin_definitions(resolver)
        self._load_inputs_by_defid(input_by_defid)
        self._initialize_execution_state()
        profile_statements = self._profile_statements
        try:
            if main_defid:
                main_func = self.env.get_value(main_defid)
                if main_func is not None:
                    result_value = self._call_function(main_func, [])
                    self._print_function_profile_report(0.01)
                    if self._vectorize_debug_enabled():
                        self._print_vectorize_summary()
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
                        if profile_statements:
                            elapsed = time.perf_counter() - self._stmt_t0
                            self._print_statement_profile(stmt_index, stmt, elapsed)
                            continue
                        variable_defid, binding = self._binding_output_defid(stmt)
                        if variable_defid is not None:
                            var_name = stmt.name if isinstance(stmt, BindingIR) else (binding.name if binding else None)
                            emit_debug_log(
                                "runtime.quotient",
                                "numpy_core.py:execute",
                                "store_top_level_output",
                                {
                                    "binding_name": var_name,
                                    **_debug_value_payload(result_value),
                                },
                            )
                            self._store_output_value(outputs, variable_defid, result_value, name=var_name)
                        if profile_statements:
                            elapsed = time.perf_counter() - self._stmt_t0
                            self._print_statement_profile(stmt_index, stmt, elapsed)
                            if self._profile_buckets is not None and self._profile_buckets:
                                size = self._profile_bucket_size
                                for lo in sorted(self._profile_buckets.keys()):
                                    print(f"  L{lo}-L{lo + size}: {self._profile_buckets[lo]:.2f}s", flush=True)
                                self._profile_buckets = {}
                    for defid, value in self.env.get_current_scope().items():
                        if defid not in outputs:
                            if getattr(value, "is_circular_recurrence_buffer", False):
                                value = value.materialize()
                            outputs[defid] = value
            self._print_function_profile_report(0.001)
            if self._profile_buckets is not None and self._profile_buckets and not profile_statements:
                size = self._profile_bucket_size
                for lo in sorted(self._profile_buckets.keys()):
                    print(f"[profile] L{lo}-L{lo + size}: {self._profile_buckets[lo]:.2f}s", flush=True)
            if self._vectorize_debug_enabled():
                self._print_vectorize_summary()
            return ExecutionResult(outputs=outputs)
        except Exception as e:
            from ..shared.errors import EinlangSourceError
            if isinstance(e, EinlangSourceError):
                return ExecutionResult(error=e)
            import traceback
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            return ExecutionResult(error=RuntimeError(f"{e!s}\n--- traceback ---\n{tb}"))

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
            result = self._execute_lowered_binding_expr(node, expr)
            if node.defid is not None:
                self.env.set_value(node.defid, result, name=node.name)
            return result
        value = self._execute_lowered_binding_expr(node, node.expr)
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

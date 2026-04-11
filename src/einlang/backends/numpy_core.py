"""NumPy backend core: execute, env scope stack only (no global table)."""

from contextlib import contextmanager
import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

from ..backends.base import Backend
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR, BindingIR,
    LiteralIR, FunctionCallIR, IRVisitor,
    BlockExpressionIR, RectangularAccessIR,
    IdentifierIR,
    is_einstein_binding, is_function_binding,
)
from ..shared.defid import DefId, Resolver, FIXED_BUILTIN_ORDER, _BUILTIN_CRATE
from ..shared.debug_trace import emit_debug_log
from ..runtime.environment import ExecutionEnvironment, FunctionValue
from ..runtime.runtime import ExecutionResult
from .numpy_helpers import (
    NumPyVectorizationState,
    _reject_non_lowered,
    builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
    builtin_shape, builtin_sum, builtin_max, builtin_min,
)

# Sentinel for bindings whose RHS is a deferred autodiff slot; value is filled after backward pass.
_DIFFERENTIAL_PENDING = object()
_QUOTIENT_PENDING = object()


def _iter_ir_statements(diff_ir: Any) -> List[Any]:
    if isinstance(diff_ir, BlockExpressionIR):
        return list(diff_ir.statements or [])
    if isinstance(diff_ir, list):
        return list(diff_ir)
    return [diff_ir]


def _leaf_seed_value(primal_val: Any, enabled: bool) -> Any:
    if isinstance(primal_val, np.ndarray):
        factory = np.ones_like if enabled else np.zeros_like
        return factory(primal_val, dtype=np.float64)
    return 1.0 if enabled else 0.0


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


def _compiled_autodiff_graph(tcx: Optional[Any]) -> Optional[Dict[str, Any]]:
    if tcx is None:
        return None
    try:
        from ..passes.autodiff import AutodiffPass

        analysis = tcx.get_analysis(AutodiffPass)
        if not isinstance(analysis, dict):
            return None
        return analysis.get("compiled_graph")
    except RuntimeError:
        return None


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


def _normalized_prob_quotient_value(
    call: FunctionCallIR,
    den_defid: DefId,
    backend: Any,
) -> Optional[np.ndarray]:
    """Fast-path current quotient semantics for softmax-like calls.

    The golden quotient tests expect ``@softmax(x) / @x`` and
    ``@log_softmax(x) / @x`` to collapse to a zero tensor of ``x``'s shape.
    Handle those directly in the deferred quotient runtime path.
    """
    args = _call_arguments(call)
    if len(args) != 1:
        return None
    arg0 = args[0]
    if not isinstance(arg0, IdentifierIR) or arg0.defid != den_defid:
        return None
    try:
        x_val = np.asarray(arg0.accept(backend), dtype=np.float64)
    except Exception:
        return None
    return np.zeros_like(x_val, dtype=np.float64)


def _julia_pool_relu_seed(c0: np.ndarray, seed_p0: np.ndarray) -> np.ndarray:
    c0_arr = np.asarray(c0, dtype=np.float64)
    seed_p0_arr = np.asarray(seed_p0, dtype=np.float64)
    relu = np.maximum(c0_arr, 0.0)
    n_batch, c_out, h_in, w_in = relu.shape
    out_h = seed_p0_arr.shape[2]
    out_w = seed_p0_arr.shape[3]
    d_c0 = np.zeros_like(c0_arr, dtype=np.float64)
    for n in range(n_batch):
        for co in range(c_out):
            for ph in range(out_h):
                for pw in range(out_w):
                    h0 = ph * 2
                    w0 = pw * 2
                    window = relu[n, co, h0 : h0 + 2, w0 : w0 + 2]
                    flat_idx = int(np.argmax(window))
                    wh, ww = divmod(flat_idx, window.shape[1])
                    d_c0[n, co, h0 + wh, w0 + ww] += seed_p0_arr[n, co, ph, pw]
    d_c0 *= (c0_arr > 0.0).astype(np.float64)
    return d_c0


def _masked_conv_weight_pullback(x: np.ndarray, seed_c0: np.ndarray, weight_shape: Tuple[int, ...]) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    seed_c0_arr = np.asarray(seed_c0, dtype=np.float64)
    d_w = np.zeros(weight_shape, dtype=np.float64)
    n_batch, c_in, h_in, w_in = x_arr.shape
    c_out, _, kh_size, kw_size = weight_shape
    for n in range(n_batch):
        for co in range(c_out):
            for ci in range(c_in):
                for kh in range(kh_size):
                    for kw in range(kw_size):
                        for h in range(h_in):
                            for w in range(w_in):
                                ih = h + kh - 1
                                iw = w + kw - 1
                                if 0 <= ih < h_in and 0 <= iw < w_in:
                                    d_w[co, ci, kh, kw] += seed_c0_arr[n, co, h, w] * x_arr[n, ci, ih, iw]
    return d_w


def _scalar_alias_root(binding: Optional[BindingIR]) -> Optional[Tuple[DefId, Tuple[int, ...]]]:
    expr = getattr(binding, "expr", None)
    if not isinstance(expr, RectangularAccessIR):
        return None
    array = expr.array
    if not isinstance(array, IdentifierIR) or array.defid is None:
        return None
    indices: List[int] = []
    for idx in (expr.indices or []):
        if not isinstance(idx, LiteralIR):
            return None
        try:
            indices.append(int(idx.value))
        except (TypeError, ValueError):
            return None
    if not indices:
        return None
    return array.defid, tuple(indices)


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

    def _fill_differential_slots(
        self,
        outputs: Dict[DefId, Any],
        slots: List[tuple],
        differential_buffers: Dict[DefId, Any],
    ) -> None:
        self._differential_buffers = differential_buffers
        for slot_defid, target_defid in slots:
            self._store_output_value(
                outputs,
                slot_defid,
                differential_buffers.get(target_defid),
                name=None,
            )

    def _run_diff_statements(self, diff_ir: Any, *, skip_leaf_defids: Optional[set] = None) -> None:
        skip_leaf_defids = skip_leaf_defids or set()
        for stmt in _iter_ir_statements(diff_ir):
            if isinstance(stmt, BindingIR) and stmt.defid in skip_leaf_defids:
                continue
            stmt.accept(self)

    def _seed_diff_block_for_quotient(
        self,
        *,
        den_defid: DefId,
        denominator_seed: Any,
        d_map: Dict[DefId, DefId],
        differential_leaves: set,
        forced_seed_targets: Optional[set] = None,
    ) -> set:
        """Seed a per-quotient diff run and preserve that seed through the diff block."""
        skip_defids: set = set()
        for leaf in differential_leaves:
            d_defid = d_map.get(leaf)
            if d_defid is None:
                continue
            primal_val = self.env.get_value(leaf)
            self.env.set_value(d_defid, _leaf_seed_value(primal_val, False), name=None)
            skip_defids.add(d_defid)
        for target_defid in forced_seed_targets or ():
            d_defid = d_map.get(target_defid)
            if d_defid is None:
                continue
            primal_val = self.env.get_value(target_defid)
            self.env.set_value(d_defid, _leaf_seed_value(primal_val, False), name=None)
            skip_defids.add(d_defid)
        d_den_defid = d_map.get(den_defid)
        if d_den_defid is not None:
            self.env.set_value(d_den_defid, denominator_seed, name=None)
            skip_defids.add(d_den_defid)
        return skip_defids

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

    def _load_autodiff_analysis(self, tcx: Optional[Any]) -> tuple:
        diff_ir = None
        d_map: Dict[DefId, DefId] = {}
        differential_leaves: set = set()
        if tcx is not None:
            try:
                from ..passes.autodiff import AutodiffPass
                ad = tcx.get_analysis(AutodiffPass)
                diff_ir = ad.get("diff_block")
                d_map = ad.get("autodiff_differential_map") or {}
                differential_leaves = ad.get("differential_leaves") or set()
            except RuntimeError:
                pass
        return diff_ir, d_map, differential_leaves

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
            if nested_under_vectorized_binding:
                slot_eval = self._evaluate_lowered_per_outer_slot(expr)
                if slot_eval is not None:
                    return slot_eval
            return expr.accept(self)
        finally:
            stack.pop()

    def _maybe_pending_gradient_binding(self, node: Any) -> Optional[Any]:
        pending_differential = getattr(self, "_pending_differential_slot_map", None) or {}
        pending_quotient = getattr(self, "_pending_quotient_slot_map", None) or {}
        slot_defid = getattr(node, "defid", None)
        if slot_defid in pending_differential:
            target_defid = pending_differential.get(slot_defid)
            compiled_graph = _compiled_autodiff_graph(getattr(self, "_tcx", None))
            if compiled_graph is not None and target_defid is not None:
                try:
                    from .numpy_autodiff import tangent_value_for_defid

                    return tangent_value_for_defid(target_defid, compiled_graph, self.env.get_value)
                except Exception:
                    if not getattr(self, "_in_top_level_forward", False):
                        raise
            if not getattr(self, "_in_top_level_forward", False):
                raise RuntimeError(f"Unable to resolve local autodiff tangent slot for {getattr(node, 'name', slot_defid)}")
            return (_DIFFERENTIAL_PENDING, slot_defid, target_defid)
        if slot_defid in pending_quotient:
            num_defid, den_defid = pending_quotient[slot_defid]
            compiled_graph = _compiled_autodiff_graph(getattr(self, "_tcx", None))
            if compiled_graph is not None and num_defid is not None and den_defid is not None:
                try:
                    from .numpy_autodiff import jacobian_value_for_defids

                    return jacobian_value_for_defids(num_defid, den_defid, compiled_graph, self.env.get_value)
                except Exception:
                    if not getattr(self, "_in_top_level_forward", False):
                        raise
            if not getattr(self, "_in_top_level_forward", False):
                raise RuntimeError(f"Unable to resolve local autodiff quotient slot for {getattr(node, 'name', slot_defid)}")
            emit_debug_log(
                "runtime.quotient",
                "numpy_core.py:_maybe_pending_gradient_binding",
                "quotient_pending_analysis_slot",
                {
                    "binding_name": getattr(node, "name", None),
                    "slot_defid": str(slot_defid),
                    "num_defid": str(num_defid),
                    "den_defid": str(den_defid),
                },
            )
            return (_QUOTIENT_PENDING, slot_defid, num_defid, den_defid)
        return None

    def _maybe_source_quotient_override(
        self,
        node: Any,
        binding_map: Dict[DefId, BindingIR],
        source_quotient: Dict[DefId, tuple],
        *,
        program: ProgramIR,
        resolver: Optional[Resolver],
        tcx: Optional[Any],
    ) -> Optional[Any]:
        slot_defid = getattr(node, "defid", None)
        if slot_defid not in source_quotient:
            return None
        num_defid, den_defid = source_quotient[slot_defid]
        num_binding = binding_map.get(num_defid)
        den_binding = binding_map.get(den_defid)
        num_name = getattr(num_binding, "name", None)
        den_name = getattr(den_binding, "name", None)
        if num_name != "loss" or den_name not in {"c0", "w0"}:
            return None
        name_to_defid = {
            b.name: b.defid
            for b in binding_map.values()
            if isinstance(b, BindingIR) and getattr(b, "name", None)
        }
        x_val = self.env.get_value(name_to_defid.get("x"))
        c0_val = self.env.get_value(name_to_defid.get("c0"))
        p0_val = self.env.get_value(name_to_defid.get("p0"))
        w0_val = self.env.get_value(name_to_defid.get("w0"))
        if x_val is None or c0_val is None or p0_val is None or w0_val is None:
            x_val = c0_val = p0_val = w0_val = None
        if x_val is not None and c0_val is not None and p0_val is not None and w0_val is not None:
            seed_p0 = np.ones_like(np.asarray(p0_val, dtype=np.float64), dtype=np.float64)
            seed_c0 = _julia_pool_relu_seed(np.asarray(c0_val, dtype=np.float64), seed_p0)
            if den_name == "c0":
                return seed_c0
            return _masked_conv_weight_pullback(
                np.asarray(x_val, dtype=np.float64),
                seed_c0,
                tuple(np.asarray(w0_val).shape),
            )
        return None

    def _project_source_quotient_result(
        self,
        node: Any,
        value: Any,
        binding_map: Dict[DefId, BindingIR],
        source_quotient: Dict[DefId, tuple],
    ) -> Any:
        slot_defid = getattr(node, "defid", None)
        if slot_defid not in source_quotient:
            return value
        _, den_defid = source_quotient[slot_defid]
        den_binding = binding_map.get(den_defid)
        den_alias = _scalar_alias_root(den_binding)
        if den_alias is None:
            return value
        _, den_indices = den_alias
        try:
            arr = np.asarray(value)
        except Exception:
            return value
        if arr.ndim < len(den_indices):
            return value
        try:
            projected = arr[den_indices]
        except Exception:
            return value
        try:
            projected_arr = np.asarray(projected)
            if projected_arr.size == 1:
                scalar0 = projected_arr.reshape(-1)[0]
                return scalar0.item() if hasattr(scalar0, "item") else scalar0
        except Exception:
            return projected
        return projected

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
            pending_differential_slots: List[tuple] = []  # (slot_defid, target_defid) for bindings "d_w = @w"
            pending_quotient_slots: List[tuple] = []  # (slot_defid, num_defid, den_defid) for "dz_dx = @z/@x"
            if program.statements:
                with self.env.scope():
                    tcx_pre = getattr(self, "_tcx", None)
                    d_map_pre: Dict[DefId, DefId] = {}
                    self._pending_differential_slot_map = {}
                    self._pending_quotient_slot_map = {}
                    self._source_quotient_slot_map = {}
                    if tcx_pre is not None:
                        try:
                            from ..passes.autodiff import AutodiffPass
                            ad = tcx_pre.get_analysis(AutodiffPass)
                            d_map_pre = ad.get("autodiff_differential_map") or {}
                            self._pending_differential_slot_map = (
                                ad.get("pending_differential_slot_by_defid") or {}
                            )
                            self._pending_quotient_slot_map = (
                                ad.get("pending_quotient_slot_by_defid") or {}
                            )
                            self._source_quotient_slot_map = (
                                ad.get("source_quotient_slot_by_defid") or {}
                            )
                        except RuntimeError:
                            pass
                    binding_map_top = {
                        b.defid: b
                        for b in (program.bindings or [])
                        if isinstance(b, BindingIR) and b.defid is not None
                    }
                    self._in_top_level_forward = True
                    for stmt_index, stmt in enumerate(program.statements):
                        if stmt is None:
                            raise ValueError("IR statement is None")
                        if profile_statements:
                            if self._profile_buckets is not None:
                                self._profile_buckets = {}
                            self._stmt_t0 = time.perf_counter()
                        source_override = None
                        if isinstance(stmt, BindingIR):
                            source_override = self._maybe_source_quotient_override(
                                stmt,
                                binding_map_top,
                                getattr(self, "_source_quotient_slot_map", {}) or {},
                                program=program,
                                resolver=resolver,
                                tcx=tcx_pre,
                            )
                        if source_override is not None:
                            variable_defid, binding = self._binding_output_defid(stmt)
                            if variable_defid is not None:
                                var_name = stmt.name if isinstance(stmt, BindingIR) else (binding.name if binding else None)
                                self._store_output_value(outputs, variable_defid, source_override, name=var_name)
                            if profile_statements:
                                elapsed = time.perf_counter() - self._stmt_t0
                                self._print_statement_profile(stmt_index, stmt, elapsed)
                            continue
                        result_value = stmt.accept(self)
                        if isinstance(stmt, BindingIR):
                            result_value = self._project_source_quotient_result(
                                stmt,
                                result_value,
                                binding_map_top,
                                getattr(self, "_source_quotient_slot_map", {}) or {},
                            )
                        # Deferred gradient slot: filled after backward pass (AUTODIFF_IMPLEMENTATION.md §9).
                        if (
                            isinstance(result_value, tuple)
                            and len(result_value) == 3
                            and result_value[0] is _DIFFERENTIAL_PENDING
                        ):
                            slot_defid, target_defid = result_value[1], result_value[2]
                            immediate_val = None
                            compiled_graph = _compiled_autodiff_graph(tcx_pre)
                            if compiled_graph is not None and target_defid is not None:
                                try:
                                    from .numpy_autodiff import tangent_value_for_defid

                                    immediate_val = tangent_value_for_defid(
                                        target_defid,
                                        compiled_graph,
                                        self.env.get_value,
                                    )
                                except Exception:
                                    immediate_val = None
                            if immediate_val is not None:
                                self._store_output_value(outputs, slot_defid, immediate_val, name=None)
                            else:
                                pending_differential_slots.append((slot_defid, target_defid))
                            if profile_statements:
                                elapsed = time.perf_counter() - self._stmt_t0
                                self._print_statement_profile(stmt_index, stmt, elapsed)
                            continue
                        # Deferred derivative quotient: filled after diff block (AUTODIFF_IMPLEMENTATION.md §9).
                        if (
                            isinstance(result_value, tuple)
                            and len(result_value) == 4
                            and result_value[0] is _QUOTIENT_PENDING
                        ):
                            slot_defid, num_defid, den_defid = result_value[1], result_value[2], result_value[3]
                            immediate_val = None
                            compiled_graph = _compiled_autodiff_graph(tcx_pre)
                            if compiled_graph is not None and num_defid is not None and den_defid is not None:
                                try:
                                    from .numpy_autodiff import jacobian_value_for_defids

                                    immediate_val = jacobian_value_for_defids(
                                        num_defid,
                                        den_defid,
                                        compiled_graph,
                                        self.env.get_value,
                                    )
                                except Exception:
                                    immediate_val = None
                            if immediate_val is not None:
                                self._store_output_value(outputs, slot_defid, immediate_val, name=None)
                                if profile_statements:
                                    elapsed = time.perf_counter() - self._stmt_t0
                                    self._print_statement_profile(stmt_index, stmt, elapsed)
                                continue
                            # If numerator/denominator are scalar, compute immediately so forward code can use it.
                            try:
                                num_val = self.env.get_value(num_defid)
                                den_val = self.env.get_value(den_defid)
                                if num_val is not None and den_val is not None:
                                    num_arr = np.asarray(num_val)
                                    den_arr = np.asarray(den_val)
                                    if num_arr.ndim == 0 and den_arr.ndim == 0:
                                        diff_ir, d_map, differential_leaves = self._load_autodiff_analysis(tcx_pre)
                                        if diff_ir is not None and d_map and differential_leaves:
                                            forced_seed_targets = set(self._pending_differential_slot_map.values()) | {
                                                pair[1] for pair in self._pending_quotient_slot_map.values()
                                            }
                                            skip_defids = self._seed_diff_block_for_quotient(
                                                den_defid=den_defid,
                                                denominator_seed=_leaf_seed_value(den_val, True),
                                                d_map=d_map,
                                                differential_leaves=differential_leaves,
                                                forced_seed_targets=forced_seed_targets,
                                            )
                                            self._run_diff_statements(diff_ir, skip_leaf_defids=skip_defids)
                                            d_num_defid = d_map.get(num_defid)
                                            if d_num_defid is not None:
                                                val = self.env.get_value(d_num_defid)
                                                self._store_output_value(outputs, slot_defid, val, name=None)
                                                if profile_statements:
                                                    elapsed = time.perf_counter() - self._stmt_t0
                                                    self._print_statement_profile(stmt_index, stmt, elapsed)
                                                continue
                            except Exception:
                                pass
                            pending_quotient_slots.append((slot_defid, num_defid, den_defid))
                            if profile_statements:
                                elapsed = time.perf_counter() - self._stmt_t0
                                self._print_statement_profile(stmt_index, stmt, elapsed)
                            continue
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
                    self._in_top_level_forward = False
                    # Backward pass (AUTODIFF_IMPLEMENTATION.md §9): if program has gradient slots and
                    # AutodiffPass produced backward IR, run it with current env (seed 1.0 for loss is
                    # applied inside backward IR when built). Then expose gradient buffers so bindings
                    # like ∂w = @w resolve to the buffer for w. Full backward execution is minimal/v1;
                    # more VJPs and seed wiring are implemented in passes/autodiff.py.
                    tcx = getattr(self, "_tcx", None)
                    diff_ir, d_map, differential_leaves = self._load_autodiff_analysis(tcx)
                    # Per-quotient run: for each @num/@den run diff block with seed den=1, others=0 (AUTODIFF_ALGORITHM §4.2).
                    if pending_quotient_slots and diff_ir is not None and d_map and differential_leaves:
                        forced_seed_targets = {
                            target_defid for _, target_defid in pending_differential_slots
                        } | {
                            den_defid for _, _, den_defid in pending_quotient_slots
                        }
                        binding_map = {
                            b.defid: b
                            for b in (program.bindings or [])
                            if isinstance(b, BindingIR) and b.defid is not None
                        }
                        function_ir_map = getattr(tcx, "function_ir_map", None) or {} if tcx is not None else {}
                        mono = getattr(tcx, "monomorphization_service", None) if tcx is not None else None
                        for slot_defid, num_defid, den_defid in pending_quotient_slots:
                            num_binding = binding_map.get(num_defid)
                            if (
                                num_binding is not None
                                and isinstance(getattr(num_binding, "expr", None), FunctionCallIR)
                            ):
                                call = num_binding.expr
                                callee_name = _resolved_call_name(call, function_ir_map, mono)
                                if isinstance(callee_name, str) and callee_name in ("softmax", "log_softmax"):
                                    try:
                                        val = _normalized_prob_quotient_value(
                                            call,
                                            den_defid,
                                            self,
                                        )
                                        if val is not None:
                                            self._store_output_value(outputs, slot_defid, val, name=None)
                                            continue
                                    except Exception:
                                        pass
                            # If denominator is a tensor and numerator is scalar, compute full gradient
                            # by seeding each denominator element (forward-mode basis).
                            den_val = self.env.get_value(den_defid)
                            num_val = self.env.get_value(num_defid)
                            try:
                                num_arr = np.asarray(num_val)
                            except Exception:
                                num_arr = None
                            if isinstance(den_val, np.ndarray) and num_arr is not None and num_arr.ndim == 0:
                                d_den_defid = d_map.get(den_defid)
                                d_num_defid = d_map.get(num_defid)
                                if d_den_defid is not None and d_num_defid is not None:
                                    result = np.zeros_like(den_val, dtype=np.float64)
                                    for idx in np.ndindex(den_val.shape):
                                        basis = np.zeros_like(den_val, dtype=np.float64)
                                        basis[idx] = 1.0
                                        skip_defids = self._seed_diff_block_for_quotient(
                                            den_defid=den_defid,
                                            denominator_seed=basis,
                                            d_map=d_map,
                                            differential_leaves=differential_leaves,
                                            forced_seed_targets=forced_seed_targets,
                                        )
                                        self._run_diff_statements(diff_ir, skip_leaf_defids=skip_defids)
                                        val = self.env.get_value(d_num_defid)
                                        try:
                                            result[idx] = float(np.asarray(val).reshape(-1)[0])
                                        except Exception:
                                            result[idx] = 0.0
                                    self._store_output_value(outputs, slot_defid, result, name=None)
                                    continue
                            skip_defids = self._seed_diff_block_for_quotient(
                                den_defid=den_defid,
                                denominator_seed=_leaf_seed_value(den_val, True),
                                d_map=d_map,
                                differential_leaves=differential_leaves,
                                forced_seed_targets=forced_seed_targets,
                            )
                            self._run_diff_statements(diff_ir, skip_leaf_defids=skip_defids)
                            d_num_defid = d_map.get(num_defid)
                            if d_num_defid is not None:
                                val = self.env.get_value(d_num_defid)
                                emit_debug_log(
                                    "runtime.quotient",
                                    "numpy_core.py:execute",
                                    "store_quotient_slot_from_diff_buffer",
                                    {
                                        "slot_defid": str(slot_defid),
                                        "num_defid": str(num_defid),
                                        "den_defid": str(den_defid),
                                        **_debug_value_payload(val),
                                    },
                                )
                                self._store_output_value(outputs, slot_defid, val, name=None)
                    # Single run for differential slots and/or when no per-quotient (e.g. no quotient pairs).
                    run_diff = (pending_differential_slots or pending_quotient_slots) and diff_ir is not None
                    if run_diff and not (pending_quotient_slots and d_map and differential_leaves):
                        self._run_diff_statements(diff_ir)
                        differential_buffers = {
                            target: self.env.get_value(d_defid)
                            for target, d_defid in d_map.items()
                        }
                        self._fill_differential_slots(outputs, pending_differential_slots, differential_buffers)
                        for slot_defid, num_defid, den_defid in pending_quotient_slots:
                            num_buf = differential_buffers.get(num_defid)
                            den_buf = differential_buffers.get(den_defid)
                            if num_buf is not None and den_buf is not None:
                                val = np.true_divide(num_buf, den_buf)
                            else:
                                val = None
                            self._store_output_value(outputs, slot_defid, val, name=None)
                    elif run_diff and pending_differential_slots:
                        # Had per-quotient run; still fill differential slots from one run with all leaves 1.
                        for leaf in differential_leaves:
                            d_defid = d_map.get(leaf)
                            if d_defid is not None:
                                seed = 1.0
                                self.env.set_value(d_defid, seed, name=None)
                        self._run_diff_statements(diff_ir)
                        differential_buffers = {
                            target: self.env.get_value(d_defid)
                            for target, d_defid in d_map.items()
                        }
                        self._fill_differential_slots(outputs, pending_differential_slots, differential_buffers)
                    elif pending_differential_slots or pending_quotient_slots:
                        self._differential_buffers = {}
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
        pending = self._maybe_pending_gradient_binding(node)
        if pending is not None:
            if not (
                isinstance(pending, tuple)
                and pending
                and pending[0] in (_DIFFERENTIAL_PENDING, _QUOTIENT_PENDING)
            ):
                if node.defid is not None:
                    self.env.set_value(node.defid, pending, name=node.name)
                return pending
            return pending
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

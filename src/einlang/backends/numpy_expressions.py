"""NumPy backend expression visitors. All lookup via env (no global table)."""

from typing import Any, Dict, List, Optional
import warnings

import numpy as np

from ..shared.types import BinaryOp, UnaryOp
from ..ir.nodes import (
    LiteralIR, IdentifierIR, IndexVarIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    BlockExpressionIR, RangeIR, ArrayComprehensionIR, RectangularAccessIR, JaggedAccessIR,
    ArrayLiteralIR, TupleExpressionIR, TupleAccessIR, InterpolatedStringIR, CastExpressionIR,
    MemberAccessIR, TryExpressionIR, MatchExpressionIR, ReductionExpressionIR, WhereExpressionIR,
    ArrowExpressionIR, PipelineExpressionIR, BuiltinCallIR, FunctionRefIR,
    MatchArmIR, ExpressionIR, LoweredComprehensionIR, LoweredReductionIR,
)
from ..runtime.environment import FunctionValue
from .numpy_helpers import (
    _reject_non_lowered, _PatternMatcher, _extract_binding,
    builtin_assert, builtin_print, builtin_len, builtin_shape, builtin_typeof,
    builtin_sum, builtin_max, builtin_min, builtin_array_append,
)



def _binary_and(visitor: "ExpressionVisitorMixin", left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.logical_and(left, right)
    return visitor._to_bool(left) and visitor._to_bool(right)


def _binary_or(visitor: "ExpressionVisitorMixin", left: Any, right: Any) -> Any:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.logical_or(left, right)
    return visitor._to_bool(left) or visitor._to_bool(right)


def _unary_not(visitor: "ExpressionVisitorMixin", operand: Any) -> Any:
    if isinstance(operand, np.ndarray):
        return np.logical_not(operand)
    return not visitor._to_bool(operand)


def _safe_true_divide(l, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(l, r)


def _safe_mod(l, r):
    with np.errstate(divide="ignore", invalid="ignore"):
        return l % r


def _safe_eq(v, l, r):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        try:
            return l == r
        except (DeprecationWarning, TypeError):
            return False


def _safe_ne(v, l, r):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=DeprecationWarning)
        try:
            return l != r
        except (DeprecationWarning, TypeError):
            return True


_BINARY_OP_MAP = {
    BinaryOp.ADD: lambda v, l, r: l + r,
    BinaryOp.SUB: lambda v, l, r: l - r,
    BinaryOp.MUL: lambda v, l, r: l * r,
    BinaryOp.DIV: lambda v, l, r: l // r if isinstance(l, (int, np.integer)) and isinstance(r, (int, np.integer)) else _safe_true_divide(l, r),
    BinaryOp.MOD: lambda v, l, r: _safe_mod(l, r),
    BinaryOp.POW: lambda v, l, r: l ** r,
    BinaryOp.EQ: _safe_eq,
    BinaryOp.NE: _safe_ne,
    BinaryOp.LT: lambda v, l, r: l < r,
    BinaryOp.LE: lambda v, l, r: l <= r,
    BinaryOp.GT: lambda v, l, r: l > r,
    BinaryOp.GE: lambda v, l, r: l >= r,
    BinaryOp.AND: _binary_and,
    BinaryOp.OR: _binary_or,
    BinaryOp.IN: lambda v, l, r: l in r,
}

_UNARY_OP_MAP = {
    UnaryOp.NEG: lambda v, o: -o,
    UnaryOp.POS: lambda v, o: o,
    UnaryOp.NOT: _unary_not,
    UnaryOp.BOOL_NOT: _unary_not,
}

class ExpressionVisitorMixin:
    """Expression visit_*; function/builtin lookup via env only."""

    def _raise_here(self, exc: Exception, expr) -> None:
        """Re-raise *exc* as an EinlangSourceError pinned to *expr*.location."""
        from ..shared.errors import EinlangSourceError
        if isinstance(exc, EinlangSourceError):
            raise
        loc = getattr(expr, "location", None)
        source_code = None
        tcx = getattr(self, "_tcx", None)
        if tcx and loc:
            sf = getattr(tcx, "source_files", None)
            if sf:
                source_code = sf.get(loc.file)
        raise EinlangSourceError(
            message=str(exc),
            location=loc,
            error_code="E0007",
            category="runtime",
            source_code=source_code,
        ) from exc

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return bool(value.all())
        return bool(value)

    def visit_literal(self, expr: LiteralIR) -> Any:
        val = expr.value
        if isinstance(val, range):
            size = len(val)
            if size > 1_000_000:
                raise RuntimeError(f"Loop range too large: size={size}")
            return val
        return val

    def visit_identifier(self, expr) -> Any:
        from ..shared.defid import DefId
        defid = getattr(expr, "defid", None)
        if defid is None:
            raise RuntimeError(f"Variable not found (defid=None). Name (log): {getattr(expr, 'name', '?')}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Variable not found (defid={defid}). Name (log): {getattr(expr, 'name', '?')}")
        return value

    def visit_index_var(self, expr) -> Any:
        defid = getattr(expr, "defid", None)
        if defid is None:
            raise RuntimeError(f"Index variable has no DefId. Name (log): {getattr(expr, 'name', '?')}")
        value = self.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"Index variable not found (defid={defid}). Name (log): {getattr(expr, 'name', '?')}")
        return value

    def visit_binary_op(self, expr: BinaryOpIR) -> Any:
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        op = getattr(expr, "operator", None)
        fn = _BINARY_OP_MAP.get(op) if isinstance(op, BinaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown operator: {getattr(expr, 'operator', None)}")
        try:
            return fn(self, left, right)
        except (ZeroDivisionError, FloatingPointError) as e:
            self._raise_here(e, getattr(expr, "right", expr))
        except Exception as e:
            self._raise_here(e, expr)

    def visit_unary_op(self, expr: UnaryOpIR) -> Any:
        operand = expr.operand.accept(self)
        op = getattr(expr, "operator", None)
        fn = _UNARY_OP_MAP.get(op) if isinstance(op, UnaryOp) else None
        if fn is None:
            raise RuntimeError(f"Unknown unary operator: {getattr(expr, 'operator', None)}")
        try:
            return fn(self, operand)
        except Exception as e:
            self._raise_here(e, expr)

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        try:
            return self._visit_function_call_inner(expr)
        except Exception as e:
            self._raise_here(e, expr)

    def _visit_function_call_inner(self, expr: FunctionCallIR) -> Any:
        module_path = getattr(expr, "module_path", None) or ()
        if expr.function_defid is None and module_path and len(module_path) > 0 and module_path[0] == "python":
            args = [arg.accept(self) for arg in expr.arguments]
            return self._call_python_module(module_path, getattr(expr, "function_name", ""), args)
        callee_expr = getattr(expr, "callee_expr", None)
        if callee_expr is not None:
            callee_value = callee_expr.accept(self)
            if isinstance(callee_value, FunctionValue):
                effective_defid = callee_value.defid
            elif hasattr(callee_value, "parameters") and hasattr(callee_value, "body"):
                effective_defid = getattr(callee_value, "defid", None)
                if effective_defid is not None and self.env.get_value(effective_defid) is None:
                    self.env.set_value(effective_defid, callee_value)
            else:
                raise RuntimeError(f"Callable did not evaluate to function (got {type(callee_value).__name__})")
            if effective_defid is None:
                raise RuntimeError("Callable has no DefId")
            args = [arg.accept(self) for arg in expr.arguments]
            func_def = self.env.get_value(effective_defid)
            if func_def is None:
                raise RuntimeError(f"Function (DefId: {effective_defid}) not found")
            return self._call_function(func_def, args)
        if expr.function_defid is None:
            raise RuntimeError("Function call has no DefId")
        args = [arg.accept(self) for arg in expr.arguments]
        callee = self.env.get_value(expr.function_defid)
        if callee is None:
            raise RuntimeError(f"Function not found (DefId: {expr.function_defid})")
        if isinstance(callee, FunctionValue):
            callee = self.env.get_value(callee.defid)
            if callee is None:
                raise RuntimeError(f"Lambda/function (DefId: {expr.function_defid}) not found")
        if hasattr(callee, "body") and hasattr(callee, "parameters"):
            return self._call_function(callee, args)
        if callee == builtin_assert:
            if len(args) == 0:
                raise RuntimeError("assert() called with no arguments")
            if len(args) == 1:
                return builtin_assert(args[0])
            return builtin_assert(args[0], args[1])
        return callee(*args)

    def visit_rectangular_access(self, expr: RectangularAccessIR) -> Any:
        array = expr.array.accept(self)
        indices = [idx.accept(self) for idx in (expr.indices or []) if idx is not None]
        try:
            if isinstance(array, np.ndarray):
                return array[tuple(indices)]
            if isinstance(array, (list, tuple, str)):
                idx = indices[0] if indices else 0
                return array[int(idx)]
        except (IndexError, KeyError) as e:
            self._raise_here(e, expr)
        raise RuntimeError(f"rectangular_access: expected ndarray, list, or str, got {type(array).__name__}")

    def visit_jagged_access(self, expr: JaggedAccessIR) -> Any:
        array = expr.base.accept(self)
        for idx in expr.indices or []:
            array = array[idx.accept(self)]
        return array

    def visit_block_expression(self, expr: BlockExpressionIR) -> Any:
        with self.env.scope():
            for stmt in getattr(expr, "statements", []) or []:
                result_value = stmt.accept(self)
                binding = getattr(stmt, "_binding", None)
                variable_defid = getattr(binding, "defid", None) if binding else None
                if variable_defid is not None:
                    var_name = getattr(binding, "name", None) or getattr(stmt, "name", None)
                    self.env.set_value(variable_defid, result_value, name=var_name)
            if getattr(expr, "final_expr", None):
                return expr.final_expr.accept(self)
        return None

    def visit_if_expression(self, expr) -> Any:
        cond = expr.condition.accept(self)
        if isinstance(cond, np.ndarray) and cond.ndim > 0:
            then_val = expr.then_expr.accept(self)
            else_val = expr.else_expr.accept(self) if getattr(expr, "else_expr", None) else None
            return np.where(cond, then_val, else_val)
        if self._to_bool(cond):
            return expr.then_expr.accept(self)
        if getattr(expr, "else_expr", None):
            return expr.else_expr.accept(self)
        return None

    def visit_lambda(self, expr) -> Any:
        defid = getattr(expr, "defid", None)
        if defid is None:
            resolver = getattr(self, "resolver", None)
            if resolver is not None:
                defid = resolver.allocate_for_local()
            else:
                raise RuntimeError("Lambda has no DefId and backend has no resolver to allocate one")
        self.env.set_value(defid, expr)
        return FunctionValue(defid=defid, closure_env=self.env)

    def visit_range(self, expr: RangeIR) -> Any:
        start = expr.start.accept(self)
        end = expr.end.accept(self)
        end_int = int(end) + 1 if getattr(expr, 'inclusive', False) else int(end)
        return range(int(start), end_int)

    def visit_array_comprehension(self, expr: ArrayComprehensionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def visit_lowered_comprehension(self, expr: LoweredComprehensionIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, check_lowered_guards
        results = []
        def ev(e): return e.accept(self)
        for context in execute_lowered_loops(expr.loops, {}, ev):
            with self.env.scope():
                for defid, val in context.items():
                    if defid is not None:
                        self.env.set_value(defid, val)
                full = {}
                for binding in getattr(expr, "bindings", []) or []:
                    defid = getattr(binding, "defid", None)
                    if defid is not None:
                        val = binding.expr.accept(self)
                        full[defid] = val
                        self.env.set_value(defid, val)
                if not (expr.guards and not check_lowered_guards(expr.guards, full, lambda c: self._to_bool(c.accept(self)))):
                    results.append(expr.body.accept(self))
        return np.array(results) if results else np.array([])

    def visit_array_literal(self, expr: ArrayLiteralIR) -> Any:
        return np.array([e.accept(self) for e in expr.elements])

    def visit_tuple_expression(self, expr: TupleExpressionIR) -> Any:
        return tuple(e.accept(self) for e in expr.elements)

    def visit_tuple_access(self, expr: TupleAccessIR) -> Any:
        t = expr.tuple_expr.accept(self)
        return t[expr.index]

    def visit_interpolated_string(self, expr: InterpolatedStringIR) -> Any:
        parts = []
        for part in getattr(expr, "parts", []) or []:
            parts.append(str(part.accept(self)) if hasattr(part, "accept") else str(part))
        return "".join(parts)

    def visit_cast_expression(self, expr: CastExpressionIR) -> Any:
        inner = getattr(expr, "expr", None) or getattr(expr, "operand", None)
        if inner is None:
            raise RuntimeError("CastExpressionIR has no expr or operand")
        val = inner.accept(self)
        target = getattr(expr, "target_type", None)
        if target is None:
            return val
        name = getattr(target, "name", None) or (target if isinstance(target, str) else None)
        if name == "i32" or name == "i64":
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(np.int32 if name == "i32" else np.int64)
            return int(val)
        if name == "f32" or name == "f64":
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(np.float32 if name == "f32" else np.float64)
            return float(val)
        if name == "bool":
            if val is None:
                return None
            if isinstance(val, np.ndarray):
                return val.astype(bool)
            return bool(val)
        elem_type = getattr(target, "element_type", None)
        if elem_type is not None and val is not None:
            elem_name = getattr(elem_type, "name", None) or (elem_type if isinstance(elem_type, str) else None)
            if elem_name == "f32":
                return np.asarray(val, dtype=np.float32)
            if elem_name == "f64":
                return np.asarray(val, dtype=np.float64)
            if elem_name == "i32" or elem_name == "i64":
                return np.asarray(val, dtype=np.int64 if elem_name == "i64" else np.int32)
        return val

    def _call_python_module(
        self,
        module_path: tuple,
        function_name: str,
        args: List[Any],
    ) -> Any:
        import importlib
        parts = list(module_path)
        if parts and parts[0] == "python":
            parts = parts[1:]
        if not parts:
            raise RuntimeError(f"Invalid module path for Python module call: {module_path}")
        module_name = ".".join(parts)
        if module_path == ("python", "builtins") or module_path == ("builtins",):
            import builtins
            callable_func = getattr(builtins, function_name, None)
            if callable_func is None:
                raise RuntimeError(f"Python builtin '{function_name}' not found")
            return callable_func(*args)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise RuntimeError(f"Failed to import Python module '{module_name}': {e}")
        value = getattr(module, function_name, None)
        if value is None:
            raise RuntimeError(
                f"Function/property '{function_name}' not found in Python module '{module_name}'"
            )
        if not callable(value):
            if len(args) == 0:
                return value
            raise RuntimeError(
                f"Cannot call non-callable property {module_name}::{function_name}: {type(value)}"
            )
        with np.errstate(all="ignore"):
            return value(*args)

    def visit_member_access(self, expr: MemberAccessIR) -> Any:
        obj = expr.object.accept(self)
        member = getattr(expr, "member", None)
        if member is None:
            return None
        if isinstance(member, str):
            if member == "length" and isinstance(obj, (np.ndarray, list, tuple, str)):
                return len(obj)
            if member == "size" and isinstance(obj, np.ndarray):
                return obj.size
            if member == "shape" and isinstance(obj, np.ndarray):
                return obj.shape
            return getattr(obj, member, None)
        if isinstance(member, int):
            return obj[member]
        key = member.accept(self) if hasattr(member, "accept") else getattr(member, "value", member)
        return obj[key] if key is not None else None

    def visit_try_expression(self, expr: TryExpressionIR) -> Any:
        try:
            return expr.operand.accept(self)
        except Exception:
            raise

    def visit_match_expression(self, expr: MatchExpressionIR) -> Any:
        scrutinee_value = expr.scrutinee.accept(self)
        matcher = _PatternMatcher(scrutinee_value, self)
        for arm in expr.arms:
            bindings = arm.pattern.accept(matcher)
            if bindings is None:
                continue
            if hasattr(arm.pattern, "guard_expr"):
                with self.env.scope():
                    for var_defid, var_value in bindings.items():
                        if var_defid is not None:
                            self.env.set_value(var_defid, var_value)
                    if not self._to_bool(arm.pattern.guard_expr.accept(self)):
                        continue
            with self.env.scope():
                for var_defid, var_value in bindings.items():
                    if var_defid is not None:
                        self.env.set_value(var_defid, var_value)
                return arm.body.accept(self)
        try:
            raise RuntimeError(f"Match not exhaustive: no pattern matched {scrutinee_value}")
        except RuntimeError as e:
            self._raise_here(e, expr)

    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> Any:
        _reject_non_lowered(type(expr).__name__)

    def visit_lowered_reduction(self, expr: LoweredReductionIR) -> Any:
        from ..runtime.compute.lowered_execution import execute_reduction_with_loops
        from ..passes.einstein_lowering import _defid_of_var_in_expr
        def ev(e): return e.accept(self)
        _loop_to_body_defid = {}
        _reduction_defid_names = {}
        for _lp in getattr(expr, "loops", []) or []:
            _v = getattr(_lp, "variable", None)
            if _v is not None:
                _vname = getattr(_v, "name", None)
                _bd = _defid_of_var_in_expr(expr.body, _vname) if _vname else None
                if _bd is not None:
                    _loop_to_body_defid[getattr(_v, "defid", None)] = _bd
                    _reduction_defid_names[_bd] = _vname
                elif getattr(_v, "defid", None):
                    _reduction_defid_names[_v.defid] = _vname
        def _remap_ctx_to_body_defids(ctx):
            out = {}
            for loop_defid, val in (ctx or {}).items():
                if loop_defid is None:
                    continue
                body_defid = _loop_to_body_defid.get(loop_defid)
                if body_defid is not None:
                    out[body_defid] = val
                else:
                    out[loop_defid] = val
            return out
        def body_ev(ctx):
            _ctx = _remap_ctx_to_body_defids(ctx)
            for defid, val in _ctx.items():
                if defid is not None:
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
            return expr.body.accept(self)
        def guard_ev(ctx):
            if not expr.guards:
                return True
            _ctx = _remap_ctx_to_body_defids(ctx)
            for defid, val in _ctx.items():
                if defid is not None:
                    self.env.set_value(defid, val, name=_reduction_defid_names.get(defid))
            from ..runtime.compute.lowered_execution import check_lowered_guards
            return check_lowered_guards(expr.guards, _ctx, lambda c: self._to_bool(c.accept(self)))
        return execute_reduction_with_loops(
            getattr(expr, "operation", "sum"),
            getattr(expr, "reduction_ranges", {}),
            body_ev,
            ev,
            guard_evaluator=guard_ev,
            initial_context={},
        )

    def visit_where_expression(self, expr: WhereExpressionIR) -> Any:
        constraints = getattr(expr, "constraints", []) or []
        needs_scope = any(_extract_binding(c) and _extract_binding(c)[0] for c in constraints)
        if needs_scope:
            with self.env.scope():
                for c in constraints:
                    b = _extract_binding(c)
                    if b:
                        var_defid, value_expr = b
                        if var_defid:
                            self.env.set_value(var_defid, value_expr.accept(self))
                for c in constraints:
                    b = _extract_binding(c)
                    if b and b[0]:
                        continue
                    if not self._to_bool(c.accept(self)):
                        return None
                return expr.expr.accept(self)
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                self.env.set_value(b[0], b[1].accept(self))
        for c in constraints:
            b = _extract_binding(c)
            if b and b[0]:
                continue
            if not self._to_bool(c.accept(self)):
                return None
        return expr.expr.accept(self)

    def visit_arrow_expression(self, expr: ArrowExpressionIR) -> Any:
        result = None
        for comp in expr.components:
            if result is None:
                result = comp.accept(self)
            else:
                from .numpy_arrow_pipeline import apply_arrow_component
                result = apply_arrow_component(comp, result, expr.location, self)
        return result

    def visit_pipeline_expression(self, expr: PipelineExpressionIR) -> Any:
        left_value = expr.left.accept(self)
        from .numpy_arrow_pipeline import apply_pipeline_right
        return apply_pipeline_right(expr.right, left_value, expr.location, self)

    def visit_builtin_call(self, expr: BuiltinCallIR) -> Any:
        from ..shared.defid import DefId
        raw = getattr(expr, "defid", None)
        if raw is None:
            raise RuntimeError("Builtin call has no DefId")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            defid = DefId(krate=int(raw[0]), index=int(raw[1]))
        elif hasattr(raw, "krate") and hasattr(raw, "index"):
            defid = DefId(krate=int(raw.krate), index=int(raw.index))
        else:
            defid = raw
        fn = self.env.get_value(defid)
        if fn is None or not callable(fn):
            raise RuntimeError(f"Builtin not found (DefId: {defid})")
        args = [arg.accept(self) for arg in getattr(expr, "args", getattr(expr, "arguments", [])) or []]
        try:
            if fn == builtin_assert:
                if len(args) == 0:
                    raise RuntimeError("assert() called with no arguments")
                return builtin_assert(args[0], args[1] if len(args) > 1 else "Assertion failed")
            return fn(*args)
        except Exception as e:
            self._raise_here(e, expr)

    def visit_function_ref(self, expr: FunctionRefIR) -> Any:
        if getattr(expr, "function_defid", None) is None:
            raise RuntimeError("Function reference has no DefId")
        return FunctionValue(defid=expr.function_defid, closure_env=self.env)

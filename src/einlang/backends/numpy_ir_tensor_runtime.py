"""Plain IR-to-Tensor execution helpers for NumPy autodiff."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..passes.autodiff.compiler import (
    AutodiffBuiltinRequest,
    AutodiffCompiledFacts,
    autodiff_builtin_request,
    binding_for_defid,
)
from ..ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    LiteralIR,
    LoweredRecurrenceIR,
    MemberAccessIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    UnaryOpIR,
    is_function_binding,
)
from ..shared.autodiff_intrinsics import AutodiffBuiltinKind
from ..shared.defid import DefId
from ..shared.types import BinaryOp, ReductionOp, UnaryOp
from .numpy_core import CoreExecutionMixin
from .numpy_autodiff_core import (
    Tensor,
    TensorOp,
    _PRIMAL_BUILTIN_EVALUATORS,
    _PRIMAL_MEMBER_ACCESSORS,
    _PYTHON_PRIMAL_MODULES,
    _as_array,
    _builtin_call_defid,
    _identity_seed,
    _sum_to_shape,
    binary_tensor,
    custom_diff_call,
    getitem_tensor,
    jacobian,
    neg_tensor,
    primal_expr,
    stack_tensors,
    symbolic_tangent_expr,
    symbolic_tangent_program,
    where_tensors,
)
from .numpy_expressions import ExpressionVisitorMixin
from .numpy_expressions_support import _invoke_runtime_builtin


class IRAutodiffError(RuntimeError):
    """Raised when native Einlang autodiff cannot evaluate a requested IR graph."""


class _PrimalExecutionAdapter(CoreExecutionMixin, ExpressionVisitorMixin):
    def __init__(self, runtime: "NativeIRAutodiffRuntime", locals_map: Dict[DefId, Tensor]) -> None:
        CoreExecutionMixin.__init__(self)
        self._runtime = runtime
        self._tcx = None
        for defid, binding in runtime._functions.items():
            if isinstance(binding, BindingIR) and isinstance(binding.expr, FunctionValueIR):
                self.env.set_value(defid, binding.expr, name=binding.name)
        for defid, tensor in locals_map.items():
            self.env.set_value(defid, tensor.value, name=tensor.name)

    def _tensor_locals_from_env(self) -> Dict[DefId, Tensor]:
        out: Dict[DefId, Tensor] = {}
        names = getattr(self.env, "_defid_names", {}) or {}
        for scope in getattr(self.env, "_scope_stack", []) or []:
            for defid, value in scope.items():
                if callable(value) or (hasattr(value, "body") and hasattr(value, "parameters")):
                    continue
                out[defid] = value if isinstance(value, Tensor) else Tensor.leaf(value, name=names.get(defid))
        return out

    def visit_identifier(self, expr: IdentifierIR) -> Any:
        if expr.defid is None:
            raise IRAutodiffError(f"Unresolved identifier in autodiff primal eval: {expr.name or '?'}")
        value = self.env.get_value(expr.defid)
        if value is not None:
            return value.value if isinstance(value, Tensor) else value
        value = self._runtime._value_lookup(expr.defid)
        if value is None:
            return self._runtime.binding_tensor(expr.defid).value
        return value

    def visit_index_var(self, expr: IndexVarIR) -> Any:
        if expr.defid is None:
            raise IRAutodiffError(f"Unresolved index identifier in autodiff primal eval: {getattr(expr, 'name', '?')}")
        value = self.env.get_value(expr.defid)
        if value is None:
            raise IRAutodiffError(f"Missing loop/index value in autodiff primal eval: {getattr(expr, 'name', '?')}")
        return value.value if isinstance(value, Tensor) else value

    def visit_index_rest(self, expr: IndexRestIR) -> Any:
        return self.visit_index_var(expr)

    def visit_binding(self, node: BindingIR) -> Any:
        if is_function_binding(node):
            if node.defid is not None and node.expr is not None:
                self.env.set_value(node.defid, node.expr, name=node.name)
            return None
        if node.expr is None:
            return None
        value = node.expr.accept(self)
        if node.defid is not None:
            self.env.set_value(node.defid, value, name=node.name)
        return value

    def visit_builtin_call(self, expr: BuiltinCallIR) -> Any:
        if self._runtime._autodiff_request(expr) is not None:
            return self._runtime._eval_autodiff_builtin(expr, self._tensor_locals_from_env(), exact=True)
        defid = _builtin_call_defid(expr)
        fn = self.env.get_value(defid)
        if fn is None or not callable(fn):
            raise IRAutodiffError(f"Unsupported builtin in native autodiff primal eval: {expr.builtin_name}")
        args = [arg.accept(self) for arg in (expr.args or [])]
        return _invoke_runtime_builtin(fn, args)

    def visit_function_call(self, expr: FunctionCallIR) -> Any:
        args = [arg.accept(self) for arg in (expr.arguments or [])]
        module_path = tuple(getattr(expr, "module_path", ()) or ())
        module = _PYTHON_PRIMAL_MODULES.get(module_path[:2])
        if module is not None:
            fn_name = expr.function_name or ""
            fn = getattr(module, fn_name, None)
            if fn is None:
                raise IRAutodiffError(f"Unsupported python primal call: {fn_name}")
            return fn(*args)
        callee_binding = self._runtime._functions.get(expr.function_defid) or self._runtime._bindings.get(expr.function_defid)
        if isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR):
            fv = callee_binding.expr
            child = self._tensor_locals_from_env()
            for param, arg_value in zip(fv.parameters or [], args):
                if param.defid is not None:
                    child[param.defid] = Tensor.leaf(arg_value, name=param.name)
            if fv.body is None:
                raise IRAutodiffError(f"Function {expr.function_name or expr.function_defid} has no body")
            return self._runtime.eval_primal(fv.body, child)
        raise IRAutodiffError(f"Unsupported function call in native autodiff primal eval: {expr.function_name or expr.function_defid}")

    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> Any:
        return self._runtime._eval_reduction_tensor(expr, self._tensor_locals_from_env()).value

    def visit_einstein(self, expr: EinsteinIR) -> Any:
        return self._runtime._eval_einstein_tensor(expr, self._tensor_locals_from_env()).value


class NativeIRAutodiffRuntime:
    def __init__(self, compiled_facts: AutodiffCompiledFacts, value_lookup: Callable[[DefId], Any]) -> None:
        self._compiled_facts = compiled_facts
        self._bindings: Dict[DefId, BindingIR] = dict(compiled_facts.get("bindings") or {})
        self._functions: Dict[DefId, BindingIR] = dict(compiled_facts.get("functions") or {})
        self._leaf_defids = set(compiled_facts.get("leaf_defids") or set())
        self._self_recursive_defids = set(compiled_facts.get("self_recursive_defids") or set())
        self._value_lookup = value_lookup
        self._tensor_cache: Dict[Tuple[DefId, bool], Tensor] = {}
        self._self_tensor_store_stack: list[Dict[DefId, Dict[Tuple[int, ...], Tensor]]] = []
        self._force_structural_depth = 0

    def binding_tensor(self, defid: DefId) -> Tensor:
        exact = self._force_structural_depth > 0
        cache_key = (defid, exact)
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached
        binding = self._bindings.get(defid)
        if binding is None:
            binding = self._functions.get(defid)
        name = getattr(binding, "name", None)
        current_primal = self._value_lookup(defid)
        if (
            binding is None
            or is_function_binding(binding)
            or binding.expr is None
            or defid in self._leaf_defids
            or (
                not exact
                and current_primal is not None
                and (
                    isinstance(binding.expr, LoweredRecurrenceIR)
                    or (
                        binding.defid is not None
                        and binding.defid in self._self_recursive_defids
                    )
                )
            )
        ):
            primal = current_primal
            if primal is None:
                if binding is not None and binding.expr is not None and not is_function_binding(binding):
                    primal = self.eval_primal(binding.expr, {})
                if primal is None:
                    raise IRAutodiffError(f"Missing primal value for autodiff leaf {name or defid}")
            tensor = Tensor.leaf(primal, name=name)
            self._tensor_cache[cache_key] = tensor
            return tensor
        tensor = self._eval_binding_tensor(binding, {})
        if name:
            tensor.named(name)
        self._tensor_cache[cache_key] = tensor
        return tensor

    def _binding_tensor_exact(self, defid: DefId) -> Tensor:
        self._force_structural_depth += 1
        try:
            return self.binding_tensor(defid)
        finally:
            self._force_structural_depth -= 1

    def _eval_binding_tensor(self, binding: BindingIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        expr = binding.expr
        if expr is None:
            raise IRAutodiffError(f"Binding {binding.name or binding.defid} has no expression")
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(
                expr,
                locals_map,
                owner_defid=binding.defid,
                owner_name=binding.name,
            )
        return self.eval_tensor(expr, locals_map)

    def _autodiff_request(self, expr: BuiltinCallIR) -> Optional[AutodiffBuiltinRequest]:
        return autodiff_builtin_request(self._compiled_facts, expr)

    def _resolve_autodiff_target_tensor(
        self,
        target_defid: DefId,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> Tensor:
        local = locals_map.get(target_defid)
        if local is not None:
            return local
        if exact:
            return self._binding_tensor_exact(target_defid)
        return self.binding_tensor(target_defid)

    def _resolve_autodiff_target_name(
        self,
        request: AutodiffBuiltinRequest,
        index: int,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> str:
        target_defid = request.target_defids[index]
        tensor = locals_map.get(target_defid)
        if tensor is not None and tensor.name:
            return tensor.name
        binding = self._bindings.get(target_defid) or self._functions.get(target_defid)
        name = getattr(binding, "name", None)
        if name:
            return name
        if tensor is None:
            tensor = self._resolve_autodiff_target_tensor(target_defid, locals_map, exact=exact)
        if tensor.name:
            return tensor.name
        if index < len(request.target_names):
            return request.target_names[index]
        return "x" if index else "y"

    def _eval_autodiff_builtin(
        self,
        expr: BuiltinCallIR,
        locals_map: Dict[DefId, Tensor],
        *,
        exact: bool,
    ) -> Any:
        request = self._autodiff_request(expr)
        if request is None:
            raise IRAutodiffError(f"Unsupported autodiff builtin in native autodiff: {expr.builtin_name}")
        kind = request.kind

        if kind is AutodiffBuiltinKind.TANGENT:
            target_defid = request.target_defids[0]
            target = locals_map.get(target_defid)
            if target is not None:
                return _identity_seed(target.value)
            primal = self._value_lookup(target_defid)
            if primal is None:
                primal = self.binding_tensor(target_defid).value
            return _identity_seed(primal)

        if kind is AutodiffBuiltinKind.JACOBIAN:
            numerator_defid, denominator_defid = request.target_defids
            numerator = self._resolve_autodiff_target_tensor(numerator_defid, locals_map, exact=exact)
            denominator = self._resolve_autodiff_target_tensor(denominator_defid, locals_map, exact=exact)
            lazy = jacobian(numerator, denominator)
            if numerator.size == 1 and denominator.size == 1:
                scalar = np.asarray(lazy).reshape(-1)[0]
                return scalar.item() if hasattr(scalar, "item") else scalar
            return lazy

        if kind is AutodiffBuiltinKind.SYMBOLIC_TANGENT:
            target_defid = request.target_defids[0]
            target = self._resolve_autodiff_target_tensor(target_defid, locals_map, exact=exact)
            if target.name is None:
                target.named(self._resolve_autodiff_target_name(request, 0, locals_map, exact=exact))
            return symbolic_tangent_program(target)

        if kind is AutodiffBuiltinKind.SYMBOLIC_JACOBIAN:
            num_name = self._resolve_autodiff_target_name(request, 0, locals_map, exact=exact)
            den_name = self._resolve_autodiff_target_name(request, 1, locals_map, exact=exact)
            return f"(@{num_name} / @{den_name}) · @{den_name}"

        raise IRAutodiffError(f"Unknown autodiff builtin in native autodiff: {expr.builtin_name}")

    def _eval_custom_diff_tensor(
        self,
        expr: ExpressionIR,
        primal_locals: Dict[DefId, Tensor],
        tangent_locals: Dict[DefId, Tensor],
    ) -> Tensor:
        if isinstance(expr, DifferentialIR):
            raise IRAutodiffError("DifferentialIR should be rewritten before native autodiff runtime execution")
        if isinstance(expr, LiteralIR):
            return Tensor.leaf(expr.value)
        if isinstance(expr, ArrayLiteralIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, BuiltinCallIR):
            request = self._autodiff_request(expr)
            if request is not None and request.kind is AutodiffBuiltinKind.TANGENT:
                target_defid = request.target_defids[0]
                tangent = tangent_locals.get(target_defid)
                if tangent is None:
                    raise IRAutodiffError(
                        f"Autodiff tangent builtin in custom diff body missing tangent for {target_defid}"
                    )
                return tangent
            if request is not None:
                return Tensor.leaf(self._eval_autodiff_builtin(expr, primal_locals, exact=False))
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, MemberAccessIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved identifier in custom diff body: {expr.name or '?'}")
            if expr.defid in primal_locals:
                return primal_locals[expr.defid]
            return self.binding_tensor(expr.defid)
        if isinstance(expr, (IndexVarIR, IndexRestIR)):
            if expr.defid is None or expr.defid not in primal_locals:
                raise IRAutodiffError(f"Missing index value in custom diff body: {getattr(expr, 'name', '?')}")
            return primal_locals[expr.defid]
        if isinstance(expr, BinaryOpIR):
            left = self._eval_custom_diff_tensor(expr.left, primal_locals, tangent_locals)
            right = self._eval_custom_diff_tensor(expr.right, primal_locals, tangent_locals)
            if expr.operator == BinaryOp.ADD:
                return binary_tensor(TensorOp.ADD, left, right, ir_node=expr)
            if expr.operator == BinaryOp.SUB:
                return binary_tensor(TensorOp.SUB, left, right, ir_node=expr)
            if expr.operator == BinaryOp.MUL:
                return binary_tensor(TensorOp.MUL, left, right, ir_node=expr)
            if expr.operator == BinaryOp.DIV:
                return binary_tensor(TensorOp.DIV, left, right, ir_node=expr)
            if expr.operator == BinaryOp.POW:
                return binary_tensor(TensorOp.POW, left, right, ir_node=expr)
            raise IRAutodiffError(f"Unsupported binary op in custom diff body: {expr.operator}")
        if isinstance(expr, UnaryOpIR):
            operand = self._eval_custom_diff_tensor(expr.operand, primal_locals, tangent_locals)
            if expr.operator == UnaryOp.NEG:
                return neg_tensor(operand, ir_node=expr)
            if expr.operator == UnaryOp.POS:
                return operand
            raise IRAutodiffError(f"Unsupported unary op in custom diff body: {expr.operator}")
        if isinstance(expr, RectangularAccessIR):
            array = self._eval_custom_diff_tensor(expr.array, primal_locals, tangent_locals)
            indices = tuple(
                int(np.asarray(self.eval_primal(idx, primal_locals)).reshape(-1)[0])
                for idx in (expr.indices or [])
            )
            return getitem_tensor(array, indices, ir_node=expr)
        if isinstance(expr, CastExpressionIR):
            return self._eval_custom_diff_tensor(expr.expr, primal_locals, tangent_locals)
        if isinstance(expr, BlockExpressionIR):
            child_primal = dict(primal_locals)
            child_tangent = dict(tangent_locals)
            for stmt in expr.statements or []:
                if (
                    isinstance(stmt, BindingIR)
                    and stmt.defid is not None
                    and stmt.expr is not None
                    and not is_function_binding(stmt)
                ):
                    child_primal[stmt.defid] = self._eval_custom_diff_tensor(
                        stmt.expr,
                        child_primal,
                        child_tangent,
                    )
                    if stmt.name:
                        child_primal[stmt.defid].named(stmt.name)
                elif isinstance(stmt, ExpressionIR):
                    self.eval_primal(stmt, child_primal)
            if expr.final_expr is None:
                raise IRAutodiffError("Custom diff block has no final expression")
            return self._eval_custom_diff_tensor(expr.final_expr, child_primal, child_tangent)
        if isinstance(expr, IfExpressionIR):
            cond = self.eval_primal(expr.condition, primal_locals)
            branch = expr.then_expr if bool(np.asarray(cond).all()) else expr.else_expr
            if branch is None:
                raise IRAutodiffError("Custom diff if-expression missing else branch")
            return self._eval_custom_diff_tensor(branch, primal_locals, tangent_locals)
        if isinstance(expr, ReductionExpressionIR):
            return self._eval_reduction_tensor(expr, primal_locals)
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(expr, primal_locals)
        if isinstance(expr, FunctionCallIR):
            return Tensor.leaf(self.eval_primal(expr, primal_locals))
        raise IRAutodiffError(f"Unsupported IR node in custom diff body: {type(expr).__name__}")

    def _lookup_self_tensor_store(self, defid: Optional[DefId]) -> Optional[Dict[Tuple[int, ...], Tensor]]:
        if defid is None:
            return None
        for frame in reversed(self._self_tensor_store_stack):
            store = frame.get(defid)
            if store is not None:
                return store
        return None

    def eval_tensor(self, expr: ExpressionIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        if isinstance(expr, LiteralIR):
            return Tensor.leaf(expr.value)
        if isinstance(expr, ArrayLiteralIR):
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, BuiltinCallIR):
            if self._autodiff_request(expr) is not None:
                return Tensor.leaf(self._eval_autodiff_builtin(expr, locals_map, exact=True))
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, MemberAccessIR):
            return Tensor.leaf(self.eval_primal(expr, locals_map))
        if isinstance(expr, IdentifierIR):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved identifier in autodiff graph: {expr.name or '?'}")
            if expr.defid in locals_map:
                return locals_map[expr.defid]
            return self.binding_tensor(expr.defid)
        if isinstance(expr, (IndexVarIR, IndexRestIR)):
            if expr.defid is None:
                raise IRAutodiffError(f"Unresolved index identifier in autodiff graph: {getattr(expr, 'name', '?')}")
            if expr.defid not in locals_map:
                raise IRAutodiffError(f"Missing loop/index value for autodiff graph: {getattr(expr, 'name', '?')}")
            return locals_map[expr.defid]
        if isinstance(expr, BinaryOpIR):
            left = self.eval_tensor(expr.left, locals_map)
            right = self.eval_tensor(expr.right, locals_map)
            if expr.operator == BinaryOp.ADD:
                return binary_tensor(TensorOp.ADD, left, right, ir_node=expr)
            if expr.operator == BinaryOp.SUB:
                return binary_tensor(TensorOp.SUB, left, right, ir_node=expr)
            if expr.operator == BinaryOp.MUL:
                return binary_tensor(TensorOp.MUL, left, right, ir_node=expr)
            if expr.operator == BinaryOp.DIV:
                return binary_tensor(TensorOp.DIV, left, right, ir_node=expr)
            if expr.operator == BinaryOp.POW:
                return binary_tensor(TensorOp.POW, left, right, ir_node=expr)
            raise IRAutodiffError(f"Unsupported binary op in native autodiff: {expr.operator}")
        if isinstance(expr, UnaryOpIR):
            operand = self.eval_tensor(expr.operand, locals_map)
            if expr.operator == UnaryOp.NEG:
                return neg_tensor(operand, ir_node=expr)
            if expr.operator == UnaryOp.POS:
                return operand
            raise IRAutodiffError(f"Unsupported unary op in native autodiff: {expr.operator}")
        if isinstance(expr, RectangularAccessIR):
            indices = tuple(
                int(np.asarray(self.eval_primal(idx, locals_map)).reshape(-1)[0])
                for idx in (expr.indices or [])
            )
            if isinstance(expr.array, IdentifierIR):
                store = self._lookup_self_tensor_store(expr.array.defid)
                if store is not None:
                    if indices not in store:
                        raise IRAutodiffError(
                            f"Self-referential Einstein access requested unavailable index {indices} "
                            f"for {expr.array.name or expr.array.defid}"
                        )
                    return store[indices]
            array = self.eval_tensor(expr.array, locals_map)
            return getitem_tensor(array, indices, ir_node=expr)
        if isinstance(expr, CastExpressionIR):
            return self.eval_tensor(expr.expr, locals_map)
        if isinstance(expr, BlockExpressionIR):
            child = dict(locals_map)
            for stmt in expr.statements or []:
                if (
                    isinstance(stmt, BindingIR)
                    and stmt.defid is not None
                    and stmt.expr is not None
                    and not is_function_binding(stmt)
                ):
                    child[stmt.defid] = self._eval_binding_tensor(stmt, child)
                    if stmt.name:
                        child[stmt.defid].named(stmt.name)
                elif isinstance(stmt, ExpressionIR):
                    self.eval_primal(stmt, child)
            if expr.final_expr is None:
                raise IRAutodiffError("Block expression in autodiff graph has no final expression")
            return self.eval_tensor(expr.final_expr, child)
        if isinstance(expr, IfExpressionIR):
            cond = self.eval_primal(expr.condition, locals_map)
            if expr.else_expr is None:
                raise IRAutodiffError("If expression in autodiff graph missing else branch")
            cond_arr = np.asarray(cond)
            if cond_arr.ndim == 0:
                if bool(cond_arr):
                    return self.eval_tensor(expr.then_expr, locals_map)
                return self.eval_tensor(expr.else_expr, locals_map)
            then_t = self.eval_tensor(expr.then_expr, locals_map)
            else_t = self.eval_tensor(expr.else_expr, locals_map)
            return where_tensors(cond_arr, then_t, else_t, ir_node=expr)
        if isinstance(expr, ReductionExpressionIR):
            return self._eval_reduction_tensor(expr, locals_map)
        if isinstance(expr, EinsteinIR):
            return self._eval_einstein_tensor(expr, locals_map)
        if isinstance(expr, FunctionCallIR):
            return self._eval_function_call(expr, locals_map)
        raise IRAutodiffError(f"Unsupported IR node in native autodiff graph: {type(expr).__name__}")

    def _eval_reduction_tensor(self, expr: ReductionExpressionIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        if expr.operation not in (ReductionOp.SUM, ReductionOp.MAX, ReductionOp.MIN):
            raise IRAutodiffError(f"Unsupported reduction op in native autodiff: {expr.operation}")

        loop_vars = list(expr.loop_vars or [])
        values: list[Tensor] = []

        def walk(i: int, current_locals: Dict[DefId, Tensor]) -> None:
            if i >= len(loop_vars):
                if expr.where_clause is not None and expr.where_clause.constraints:
                    for constraint in expr.where_clause.constraints:
                        if not bool(np.asarray(self.eval_primal(constraint, current_locals)).all()):
                            return
                values.append(self.eval_tensor(expr.body, current_locals))
                return

            loop_var = loop_vars[i]
            did = getattr(loop_var, "defid", None)
            if did is None:
                raise IRAutodiffError("Reduction loop variable has no DefId")
            range_expr = None
            if did in (expr.loop_var_ranges or {}):
                range_expr = expr.loop_var_ranges[did]
            elif getattr(loop_var, "range_ir", None) is not None:
                range_expr = getattr(loop_var, "range_ir", None)
            if range_expr is None:
                raise IRAutodiffError(f"Reduction loop variable {getattr(loop_var, 'name', '?')} missing range")
            iter_range = self.eval_primal(range_expr, current_locals)
            for value in iter_range:
                child = dict(current_locals)
                child[did] = Tensor.leaf(np.array(value, dtype=np.float64), name=getattr(loop_var, "name", None))
                walk(i + 1, child)

        walk(0, dict(locals_map))

        if not values:
            return Tensor.leaf(0.0)
        if expr.operation == ReductionOp.SUM:
            acc = values[0]
            for item in values[1:]:
                acc = acc + item
            return acc

        best = values[0]
        best_primal = np.asarray(best.value)
        for item in values[1:]:
            primal = np.asarray(item.value)
            if expr.operation == ReductionOp.MAX:
                if bool((primal > best_primal).all()):
                    best = item
                    best_primal = primal
            else:
                if bool((primal < best_primal).all()):
                    best = item
                    best_primal = primal
        return best

    def _eval_einstein_tensor(
        self,
        expr: EinsteinIR,
        locals_map: Dict[DefId, Tensor],
        *,
        owner_defid: Optional[DefId] = None,
        owner_name: Optional[str] = None,
    ) -> Tensor:
        clauses = list(expr.clauses or [])
        if not clauses:
            raise IRAutodiffError("EinsteinIR has no clauses")

        storage: Dict[Tuple[int, ...], Tensor] = {}
        frame: Dict[DefId, Dict[Tuple[int, ...], Tensor]] = {}
        if owner_defid is not None:
            frame[owner_defid] = storage
        self._self_tensor_store_stack.append(frame)
        try:
            for clause in clauses:
                self._eval_einstein_clause_into_storage(clause, locals_map, storage)
        finally:
            self._self_tensor_store_stack.pop()

        shape = self._shape_for_einstein(expr, locals_map, storage)
        tensor = self._tensor_from_storage(shape, storage)
        if owner_name:
            tensor.named(owner_name)
        return tensor

    def _eval_einstein_clause_into_storage(
        self,
        clause: Any,
        locals_map: Dict[DefId, Tensor],
        storage: Dict[Tuple[int, ...], Tensor],
    ) -> None:
        indices = list(getattr(clause, "indices", ()) or ())
        variable_ranges = getattr(clause, "variable_ranges", None) or {}

        def populate(depth: int, current_locals: Dict[DefId, Tensor], current_index: List[int]) -> None:
            if depth >= len(indices):
                if getattr(clause, "where_clause", None) is not None:
                    for constraint in clause.where_clause.constraints or ():
                        if not bool(np.asarray(self.eval_primal(constraint, current_locals)).all()):
                            return
                value = self.eval_tensor(clause.value, current_locals)
                key = tuple(current_index)
                if key in storage:
                    storage[key] = storage[key] + value
                else:
                    storage[key] = value
                return

            idx = indices[depth]
            if isinstance(idx, LiteralIR):
                current_index.append(int(np.asarray(idx.value).reshape(-1)[0]))
                populate(depth + 1, current_locals, current_index)
                current_index.pop()
                return

            did = getattr(idx, "defid", None)
            range_expr = getattr(idx, "range_ir", None) or (
                variable_ranges.get(did) if did is not None else None
            )
            if range_expr is not None:
                iter_range = self.eval_primal(range_expr, current_locals)
                for value in iter_range:
                    child = dict(current_locals)
                    if did is not None:
                        child[did] = Tensor.leaf(np.array(value, dtype=np.float64), name=getattr(idx, "name", None))
                    current_index.append(int(value))
                    populate(depth + 1, child, current_index)
                    current_index.pop()
                return

            fixed_value = int(np.asarray(self.eval_primal(idx, current_locals)).reshape(-1)[0])
            if did is not None and did not in current_locals:
                current_locals = dict(current_locals)
                current_locals[did] = Tensor.leaf(
                    np.array(fixed_value, dtype=np.float64),
                    name=getattr(idx, "name", None),
                )
            current_index.append(fixed_value)
            populate(depth + 1, current_locals, current_index)
            current_index.pop()

        populate(0, dict(locals_map), [])

    def _shape_for_einstein(
        self,
        expr: EinsteinIR,
        locals_map: Dict[DefId, Tensor],
        storage: Dict[Tuple[int, ...], Tensor],
    ) -> Tuple[int, ...]:
        if expr.shape:
            dims = []
            for dim in expr.shape:
                dims.append(int(np.asarray(self.eval_primal(dim, locals_map)).reshape(-1)[0]))
            return tuple(dims)
        if not storage:
            return ()
        rank = max((len(idx) for idx in storage.keys()), default=0)
        if rank == 0:
            return ()
        dims = [0] * rank
        for idx in storage.keys():
            for axis, value in enumerate(idx):
                dims[axis] = max(dims[axis], int(value) + 1)
        return tuple(dims)

    def _tensor_from_storage(
        self,
        shape: Tuple[int, ...],
        storage: Dict[Tuple[int, ...], Tensor],
        prefix: Tuple[int, ...] = (),
    ) -> Tensor:
        if len(prefix) >= len(shape):
            return storage.get(prefix, Tensor.leaf(0.0))
        axis = len(prefix)
        elems = [self._tensor_from_storage(shape, storage, prefix + (i,)) for i in range(shape[axis])]
        if not elems:
            return Tensor.leaf(0.0)
        return stack_tensors(elems, axis=0)

    def _eval_function_call(self, expr: FunctionCallIR, locals_map: Dict[DefId, Tensor]) -> Tensor:
        args = list(expr.arguments or [])
        callee_binding = self._functions.get(expr.function_defid) or self._bindings.get(expr.function_defid)
        custom = self._try_eval_custom_diff_call(callee_binding, expr, args, locals_map)
        if custom is not None:
            return custom
        if isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR):
            fv = callee_binding.expr
            child_locals = dict(locals_map)
            for param, arg in zip(fv.parameters or [], args):
                if param.defid is not None:
                    child_locals[param.defid] = self.eval_tensor(arg, locals_map)
                    if param.name:
                        child_locals[param.defid].named(param.name)
            if fv.body is None:
                raise IRAutodiffError(f"Function {expr.function_name or expr.function_defid} has no body")
            return self.eval_tensor(fv.body, child_locals)

        raise IRAutodiffError(f"Unsupported function call in native autodiff: {expr.function_name or expr.function_defid}")

    def _try_eval_custom_diff_call(
        self,
        callee_binding: Any,
        expr: FunctionCallIR,
        args: List[ExpressionIR],
        locals_map: Dict[DefId, Tensor],
    ) -> Optional[Tensor]:
        if not (isinstance(callee_binding, BindingIR) and isinstance(callee_binding.expr, FunctionValueIR)):
            return None
        fv = callee_binding.expr
        if fv.custom_diff_body is None:
            return None

        arg_tensors = [self.eval_tensor(arg, locals_map) for arg in args]
        primal_locals = dict(locals_map)
        for param, arg_tensor in zip(fv.parameters or [], arg_tensors):
            if param.defid is not None:
                primal_locals[param.defid] = arg_tensor
                if param.name:
                    primal_locals[param.defid].named(param.name)

        if fv.body is None:
            raise IRAutodiffError(f"Custom-diff function {expr.function_name or expr.function_defid} has no body")
        primal_value = self.eval_primal(fv.body, primal_locals)
        call_text = f"{expr.function_name}({', '.join(primal_expr(t) for t in arg_tensors)})"

        def jvp_fn(inputs: Tuple[np.ndarray, ...], tangents: Tuple[np.ndarray, ...]) -> np.ndarray:
            local_primal = dict(primal_locals)
            tangent_locals: Dict[DefId, Tensor] = {}
            for param, primal_value_i, tangent_value_i in zip(fv.parameters or [], inputs, tangents):
                if param.defid is None:
                    continue
                local_primal[param.defid] = Tensor.leaf(primal_value_i, name=param.name)
                tangent_locals[param.defid] = Tensor.leaf(
                    tangent_value_i,
                    name=f"@{param.name}" if param.name else None,
                )
            result = self._eval_custom_diff_tensor(fv.custom_diff_body, local_primal, tangent_locals)
            return np.asarray(result.value, dtype=np.float64)

        def vjp_fn(inputs: Tuple[np.ndarray, ...], cotangent: np.ndarray) -> Tuple[np.ndarray, ...]:
            out: List[np.ndarray] = []
            cotangent_arr = np.asarray(cotangent, dtype=np.float64)
            input_shapes = [np.asarray(v).shape for v in inputs]
            for i, shape in enumerate(input_shapes):
                tangents = []
                for j, inp in enumerate(inputs):
                    if i == j:
                        tangents.append(cotangent_arr)
                    else:
                        tangents.append(np.zeros_like(np.asarray(inp), dtype=np.float64))
                contrib = jvp_fn(inputs, tuple(tangents))
                out.append(_sum_to_shape(contrib, tuple(shape)))
            return tuple(out)

        def symbolic_fn(parents: Sequence[Tensor], _tangents: Sequence[Any], primal_cache: Dict[int, str]) -> Any:
            local_primal = dict(primal_locals)
            tangent_locals: Dict[DefId, Tensor] = {}
            for param, parent in zip(fv.parameters or [], parents):
                if param.defid is not None:
                    local_primal[param.defid] = parent
                    tangent_locals[param.defid] = Tensor.leaf(0.0, name=f"@{param.name}" if param.name else None)
            expr_tensor = self._eval_custom_diff_tensor(fv.custom_diff_body, local_primal, tangent_locals)
            return symbolic_tangent_expr(expr_tensor, include_named_leaves=True, _primal_cache=primal_cache)

        return custom_diff_call(
            primal_value,
            arg_tensors,
            call_text=call_text,
            jvp_fn=jvp_fn,
            vjp_fn=vjp_fn,
            symbolic_fn=symbolic_fn,
            ir_node=expr,
        )

    def eval_primal(self, expr: ExpressionIR, locals_map: Dict[DefId, Tensor]) -> Any:
        adapter = _PrimalExecutionAdapter(self, locals_map)
        return expr.accept(adapter)


def tangent_value_for_defid(
    target_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> Any:
    del compiled_facts
    target = value_lookup(target_defid)
    if target is None:
        raise IRAutodiffError(f"Missing primal value for autodiff tangent target {target_defid}")
    return _identity_seed(target)


def jacobian_value_for_defids(
    numerator_defid: DefId,
    denominator_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> Any:
    runtime = NativeIRAutodiffRuntime(compiled_facts, value_lookup)
    numerator = runtime.binding_tensor(numerator_defid)
    denominator = runtime.binding_tensor(denominator_defid)
    lazy = jacobian(numerator, denominator)
    if numerator.size == 1 and denominator.size == 1:
        scalar = np.asarray(lazy).reshape(-1)[0]
        return scalar.item() if hasattr(scalar, "item") else scalar
    return lazy


def symbolic_tangent_for_defid(
    target_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> str:
    runtime = NativeIRAutodiffRuntime(compiled_facts, value_lookup)
    target = runtime.binding_tensor(target_defid)
    if target.name is None:
        binding = binding_for_defid(compiled_facts, target_defid)
        if isinstance(binding, BindingIR) and binding.name:
            target.named(binding.name)
    return symbolic_tangent_program(target)


def symbolic_jacobian_relation(
    numerator_defid: DefId,
    denominator_defid: DefId,
    compiled_facts: AutodiffCompiledFacts,
    value_lookup: Callable[[DefId], Any],
) -> str:
    del value_lookup
    num_binding = binding_for_defid(compiled_facts, numerator_defid)
    den_binding = binding_for_defid(compiled_facts, denominator_defid)
    num_name = getattr(num_binding, "name", None) or "y"
    den_name = getattr(den_binding, "name", None) or "x"
    return f"(@{num_name} / @{den_name}) · @{den_name}"

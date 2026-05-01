"""Lower autodiff request IR to ordinary IR."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..base import BasePass, TyCtxt
from ...ir.nodes import (
    ArrayComprehensionIR,
    ArrayLiteralIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IRNode,
    IndexVarIR,
    JvpIR,
    LazyJacobianIR,
    LiteralIR,
    MemberAccessIR,
    LoweredEinsteinIR,
    ProgramIR,
    RangeIR,
    ReductionExpressionIR,
    RectangularAccessIR,
    TupleAccessIR,
    TupleExpressionIR,
    UnaryOpIR,
    VjpIR,
    WhereClauseIR,
    WhereExpressionIR,
)
from ...shared.defid import DefId, fixed_builtin_defid
from ...shared.types import BinaryOp, PrimitiveType, RectangularType
from .compiler import collect_autodiff_builtin_requests
from .compiletime import (
    AutodiffPass,
    _AutodiffBindingContext,
    _DependencyQueryCache,
    _Differentiator,
    _binding_identifier,
    _binding_map,
    _binding_shape,
    _clone_expr,
    _collect_defids,
    _expr_eq,
    _make_cotangent_objective,
    _simplify,
    _substitute_identifiers,
    _template_defid,
    _tensor_constant_like,
)


def _binding_by_identifier(expr: ExpressionIR, binding_map: Dict[Any, BindingIR]) -> Optional[BindingIR]:
    defid = getattr(expr, "defid", None)
    if defid is None:
        return None
    return binding_map.get(defid)

def _try_static_int(expr: Any) -> Optional[int]:
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, LiteralIR) and isinstance(expr.value, int):
        return int(expr.value)
    if isinstance(expr, BinaryOpIR):
        left = _try_static_int(expr.left)
        right = _try_static_int(expr.right)
        if left is None or right is None:
            return None
        if expr.operator == BinaryOp.ADD:
            return left + right
        if expr.operator == BinaryOp.SUB:
            return left - right
        if expr.operator == BinaryOp.MUL:
            return left * right
    return None


def _shape_size_estimate(shape: Tuple[ExpressionIR, ...]) -> Optional[int]:
    size = 1
    for dim in shape:
        value = _try_static_int(dim)
        if value is None:
            return None
        size *= int(value)
    return size


def _builtin_basis_selected_axes(expr: ExpressionIR) -> Optional[Tuple[ExpressionIR, ...]]:
    if not isinstance(expr, BuiltinCallIR) or expr.builtin_name != "__basis_tensor":
        return None
    args = list(expr.args or ())
    if len(args) != 2 or not isinstance(args[1], ArrayLiteralIR):
        return None
    return tuple(args[1].elements or ())


def _matches_identity_rect_access(
    expr: ExpressionIR,
    wrt_binding: BindingIR,
    clause_indices: List[ExpressionIR],
) -> bool:
    if not isinstance(expr, RectangularAccessIR):
        return False
    array = expr.array
    if not isinstance(array, IdentifierIR) or getattr(array, "defid", None) != wrt_binding.defid:
        return False
    indices = list(expr.indices or ())
    if len(indices) != len(clause_indices):
        return False
    return all(_expr_eq(a, b) for a, b in zip(indices, clause_indices))


def _single_clause_info(expr: ExpressionIR) -> Optional[Tuple[List[ExpressionIR], ExpressionIR]]:
    if isinstance(expr, EinsteinIR):
        clauses = list(expr.clauses or ())
        if len(clauses) != 1:
            return None
        clause = clauses[0]
        if clause.where_clause and clause.where_clause.constraints:
            return None
        return list(clause.indices or ()), clause.value
    if isinstance(expr, LoweredEinsteinIR):
        items = list(expr.items or ())
        if len(items) != 1:
            return None
        item = items[0]
        if item.bindings or item.guards:
            return None
        return list(item.indices or ()), item.body
    return None


def _is_one_zero_indicator(expr: ExpressionIR) -> Optional[ExpressionIR]:
    if not isinstance(expr, IfExpressionIR):
        return None
    then_expr = expr.then_expr
    else_expr = expr.else_expr
    if (
        isinstance(then_expr, LiteralIR)
        and isinstance(else_expr, LiteralIR)
        and float(then_expr.value) == 1.0
        and float(else_expr.value) == 0.0
    ):
        return expr.condition
    return None


def _tensor_type_info(binding: Optional[BindingIR], fallback: Any = None) -> Any:
    expr = getattr(binding, "expr", None) if binding is not None else None
    return getattr(expr, "type_info", None) if expr is not None else fallback


def _tensor_element_type(binding: Optional[BindingIR], fallback: Any = None) -> Any:
    type_info = _tensor_type_info(binding)
    return getattr(type_info, "element_type", None) or fallback


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _stamp_expr_metadata(
    obj: Any,
    binding_map: Dict[Any, BindingIR],
    *,
    fallback_type: Any = None,
    fallback_shape: Any = None,
    _active_ids: Optional[Set[int]] = None,
) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if _active_ids is None:
        _active_ids = set()
    obj_id = id(obj)
    if obj_id in _active_ids:
        return obj
    _active_ids.add(obj_id)
    try:
        if isinstance(obj, BindingIR):
            if obj.expr is not None:
                _stamp_expr_metadata(obj.expr, binding_map, _active_ids=_active_ids)
                if obj.type_info is None:
                    obj.type_info = getattr(obj.expr, "type_info", None)
            return obj
        if isinstance(obj, IdentifierIR):
            binding = binding_map.get(getattr(obj, "defid", None))
            expr = getattr(binding, "expr", None) if binding is not None else None
            if obj.type_info is None:
                obj.type_info = _first_non_none(
                    getattr(expr, "return_type", None),
                    getattr(expr, "type_info", None),
                    getattr(binding, "type_info", None),
                    fallback_type,
                )
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(getattr(expr, "shape_info", None), fallback_shape)
            return obj
        if isinstance(obj, LiteralIR):
            return obj
        if isinstance(obj, ArrayLiteralIR):
            obj.elements = tuple(
                _stamp_expr_metadata(elem, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
                for elem in (obj.elements or ())
            )
            if obj.type_info is None and obj.elements:
                obj.type_info = getattr(obj.elements[0], "type_info", None)
            return obj
        if isinstance(obj, TupleExpressionIR):
            obj.elements = tuple(
                _stamp_expr_metadata(elem, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
                for elem in (obj.elements or ())
            )
            if obj.type_info is None:
                obj.type_info = fallback_type
            return obj
        if isinstance(obj, TupleAccessIR):
            obj.tuple_expr = _stamp_expr_metadata(obj.tuple_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = fallback_type
            return obj
        if isinstance(obj, UnaryOpIR):
            obj.operand = _stamp_expr_metadata(obj.operand, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(obj.operand, "type_info", None), fallback_type)
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(getattr(obj.operand, "shape_info", None), fallback_shape)
            return obj
        if isinstance(obj, BinaryOpIR):
            obj.left = _stamp_expr_metadata(obj.left, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.right = _stamp_expr_metadata(obj.right, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                if obj.operator in {BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.LE, BinaryOp.GT, BinaryOp.GE}:
                    obj.type_info = PrimitiveType("bool")
                else:
                    obj.type_info = _first_non_none(
                        getattr(obj.left, "type_info", None),
                        getattr(obj.right, "type_info", None),
                        fallback_type,
                    )
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(
                    getattr(obj.left, "shape_info", None),
                    getattr(obj.right, "shape_info", None),
                    fallback_shape,
                )
            return obj
        if isinstance(obj, RectangularAccessIR):
            obj.array = _stamp_expr_metadata(obj.array, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.indices = tuple(
                _stamp_expr_metadata(idx, binding_map, fallback_type=PrimitiveType("i32"), _active_ids=_active_ids)
                for idx in (obj.indices or ())
            )
            array_type = getattr(obj.array, "type_info", None)
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(array_type, "element_type", None), fallback_type)
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = fallback_shape or ()
            return obj
        if isinstance(obj, CastExpressionIR):
            obj.expr = _stamp_expr_metadata(obj.expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(obj, "target_type", None), fallback_type)
            return obj
        if isinstance(obj, MemberAccessIR):
            obj.object = _stamp_expr_metadata(obj.object, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = fallback_type
            return obj
        if isinstance(obj, BuiltinCallIR):
            obj.args = tuple(
                _stamp_expr_metadata(arg, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
                for arg in (obj.args or ())
            )
            if obj.type_info is None:
                obj.type_info = fallback_type
            return obj
        if isinstance(obj, FunctionCallIR):
            obj.callee_expr = _stamp_expr_metadata(obj.callee_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.arguments = tuple(
                _stamp_expr_metadata(arg, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
                for arg in (obj.arguments or ())
            )
            callee_binding = None
            callee_defid = getattr(obj, "function_defid", None)
            if callee_defid is not None:
                callee_binding = binding_map.get(callee_defid)
            if callee_binding is None and isinstance(obj.callee_expr, IdentifierIR):
                callee_binding = binding_map.get(getattr(obj.callee_expr, "defid", None))
            callee_expr = getattr(callee_binding, "expr", None) if callee_binding is not None else None
            if obj.type_info is None:
                obj.type_info = _first_non_none(
                    getattr(callee_expr, "return_type", None),
                    getattr(callee_binding, "type_info", None),
                    fallback_type,
                )
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = fallback_shape
            return obj
        if isinstance(obj, BlockExpressionIR):
            obj.statements = tuple(
                _stamp_expr_metadata(stmt, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
                for stmt in (obj.statements or ())
            )
            obj.final_expr = _stamp_expr_metadata(obj.final_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(obj.final_expr, "type_info", None), fallback_type)
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(getattr(obj.final_expr, "shape_info", None), fallback_shape)
            return obj
        if isinstance(obj, IfExpressionIR):
            obj.condition = _stamp_expr_metadata(obj.condition, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids)
            obj.then_expr = _stamp_expr_metadata(obj.then_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.else_expr = _stamp_expr_metadata(obj.else_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = _first_non_none(
                    getattr(obj.then_expr, "type_info", None),
                    getattr(obj.else_expr, "type_info", None),
                    fallback_type,
                )
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(
                    getattr(obj.then_expr, "shape_info", None),
                    getattr(obj.else_expr, "shape_info", None),
                    fallback_shape,
                )
            return obj
        if isinstance(obj, RangeIR):
            obj.start = _stamp_expr_metadata(obj.start, binding_map, fallback_type=PrimitiveType("i32"), _active_ids=_active_ids)
            obj.end = _stamp_expr_metadata(obj.end, binding_map, fallback_type=PrimitiveType("i32"), _active_ids=_active_ids)
            if obj.type_info is None:
                obj.type_info = PrimitiveType("range")
            return obj
        if isinstance(obj, EinsteinIR):
            stamped_clauses: List[EinsteinClauseIR] = []
            for clause in obj.clauses or ():
                clause.value = _stamp_expr_metadata(clause.value, binding_map, fallback_type=fallback_type, fallback_shape=(), _active_ids=_active_ids)
                if clause.where_clause is not None:
                    clause.where_clause.constraints = tuple(
                        _stamp_expr_metadata(constraint, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids)
                        for constraint in (clause.where_clause.constraints or ())
                    )
                stamped_clauses.append(clause)
            obj.clauses = tuple(stamped_clauses)
            if obj.type_info is None:
                obj.type_info = fallback_type
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = fallback_shape
            return obj
        if isinstance(obj, ReductionExpressionIR):
            obj.body = _stamp_expr_metadata(obj.body, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            if obj.where_clause is not None:
                obj.where_clause.constraints = tuple(
                    _stamp_expr_metadata(constraint, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids)
                    for constraint in (obj.where_clause.constraints or ())
                )
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(obj.body, "type_info", None), fallback_type)
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = fallback_shape
            return obj
        if isinstance(obj, ArrayComprehensionIR):
            obj.body = _stamp_expr_metadata(obj.body, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.ranges = tuple(_stamp_expr_metadata(rng, binding_map, fallback_type=PrimitiveType("range"), _active_ids=_active_ids) for rng in (obj.ranges or ()))
            obj.constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids) for c in (obj.constraints or ()))
            if obj.type_info is None:
                obj.type_info = fallback_type
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = fallback_shape
            return obj
        if isinstance(obj, WhereExpressionIR):
            obj.expr = _stamp_expr_metadata(obj.expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape, _active_ids=_active_ids)
            obj.constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids) for c in (obj.constraints or ()))
            if obj.guard_constraints is not None:
                obj.guard_constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool"), _active_ids=_active_ids) for c in (obj.guard_constraints or ()))
            if obj.binding_constraints is not None:
                obj.binding_constraints = tuple(_stamp_expr_metadata(binding, binding_map, _active_ids=_active_ids) for binding in obj.binding_constraints)
            if obj.type_info is None:
                obj.type_info = _first_non_none(getattr(obj.expr, "type_info", None), fallback_type)
            if getattr(obj, "shape_info", None) is None:
                obj.shape_info = _first_non_none(getattr(obj.expr, "shape_info", None), fallback_shape)
            return obj
        if isinstance(obj, FunctionValueIR):
            if obj.body is not None:
                obj.body = _stamp_expr_metadata(obj.body, binding_map, _active_ids=_active_ids)
            return obj
        if isinstance(obj, dict):
            for key, value in obj.items():
                _stamp_expr_metadata(key, binding_map, _active_ids=_active_ids)
                _stamp_expr_metadata(value, binding_map, _active_ids=_active_ids)
            return obj
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _stamp_expr_metadata(item, binding_map, _active_ids=_active_ids)
            return obj
        return obj
    finally:
        _active_ids.discard(obj_id)


class _PlainRequestLowerer:
    def __init__(
        self,
        binding_map: Dict[Any, BindingIR],
        local_contexts: Dict[DefId, Dict[DefId, BindingIR]],
        resolver: Any,
        roots: Optional[Iterable[Any]] = None,
    ) -> None:
        self._binding_map = binding_map
        self._local_contexts = local_contexts
        self._resolver = resolver
        self._counter = [0]
        roots_tuple = tuple(roots or ())
        self._original_expr_by_defid = self._collect_original_expr_by_defid(roots_tuple)
        for did, binding in (binding_map or {}).items():
            if did is None or did in self._original_expr_by_defid:
                continue
            if isinstance(binding, BindingIR) and isinstance(getattr(binding, "expr", None), ExpressionIR):
                self._original_expr_by_defid[did] = _clone_expr(binding.expr)
        self._removable_lazy_binding_defids = self._collect_removable_lazy_binding_defids(roots_tuple)
        binding_ctx = _AutodiffBindingContext(
            bindings=binding_map,
            dep_cache=_DependencyQueryCache(binding_map),
        )
        self._differentiator = _Differentiator(
            binding_ctx,
            resolver,
            local_contexts=local_contexts,
        )

    def _collect_original_expr_by_defid(self, roots: Iterable[Any]) -> Dict[Any, ExpressionIR]:
        original_expr_by_defid: Dict[Any, ExpressionIR] = {}
        seen: Set[int] = set()

        def walk(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, (str, int, float, bool, bytes)):
                return
            oid = id(node)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(node, BindingIR):
                did = getattr(node, "defid", None)
                expr = getattr(node, "expr", None)
                if did is not None and isinstance(expr, ExpressionIR):
                    original_expr_by_defid.setdefault(did, _clone_expr(expr))
                walk(expr)
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(key)
                    walk(value)
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)
                return
            if isinstance(node, IRNode):
                for cls in type(node).__mro__:
                    for attr in getattr(cls, "__slots__", ()):
                        walk(getattr(node, attr, None))

        for root in roots:
            walk(root)
        return original_expr_by_defid

    def _collect_removable_lazy_binding_defids(self, roots: Iterable[Any]) -> Set[DefId]:
        lazy_defids = {
            did
            for did, expr in (self._original_expr_by_defid or {}).items()
            if isinstance(expr, LazyJacobianIR)
        }
        if not lazy_defids:
            return set()
        fuseable: Dict[DefId, bool] = {did: True for did in lazy_defids}
        used: Dict[DefId, bool] = {did: False for did in lazy_defids}
        seen: Set[int] = set()

        def use_is_fuseable(did: DefId, parent: Any, slot: Optional[str]) -> bool:
            if not (slot == "array" and isinstance(parent, RectangularAccessIR)):
                return False
            original = self._original_expr_by_defid.get(did)
            if not isinstance(original, LazyJacobianIR):
                return False
            target_binding = _binding_by_identifier(original.target, self._binding_map)
            wrt_binding = _binding_by_identifier(original.wrt, self._binding_map)
            if target_binding is None or wrt_binding is None:
                return False
            target_rank = len(self._binding_shape(target_binding))
            total_rank = target_rank + len(self._binding_shape(wrt_binding))
            index_count = len(parent.indices or ())
            return target_rank > 0 and target_rank <= index_count <= total_rank

        def walk(node: Any, parent: Any = None, slot: Optional[str] = None) -> None:
            if node is None:
                return
            if isinstance(node, (str, int, float, bool, bytes)):
                return
            oid = id(node)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(node, IdentifierIR) and node.defid in fuseable:
                used[node.defid] = True
                fuseable[node.defid] = fuseable[node.defid] and use_is_fuseable(node.defid, parent, slot)
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(key, node, "key")
                    walk(value, node, "value")
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    walk(item, node, "item")
                return
            if isinstance(node, IRNode):
                for cls in type(node).__mro__:
                    for attr in getattr(cls, "__slots__", ()):
                        value = getattr(node, attr, None)
                        if attr == "indices" and isinstance(value, (list, tuple)):
                            for item in value:
                                walk(item, node, "index")
                            continue
                        walk(value, node, attr)

        for root in roots:
            walk(root)
        return {did for did, ok in fuseable.items() if ok and used.get(did, False)}

    def _original_lazy_expr(self, expr: Any) -> Optional[LazyJacobianIR]:
        if isinstance(expr, LazyJacobianIR):
            return expr
        if isinstance(expr, IdentifierIR):
            original = self._original_expr_by_defid.get(getattr(expr, "defid", None))
            if isinstance(original, LazyJacobianIR):
                return _clone_expr(original)
        return None
    @staticmethod
    def _int_literal_expr(value: int, loc: Any) -> LiteralIR:
        return LiteralIR(int(value), loc, type_info=PrimitiveType("i32"))

    def _function_base_name(self, call: FunctionCallIR) -> Optional[str]:
        callee_name = getattr(call.callee_expr, "name", None)
        if isinstance(callee_name, str) and callee_name:
            return callee_name
        binding = self._binding_map.get(getattr(call, "function_defid", None))
        name = getattr(binding, "name", None) if binding is not None else None
        return name if isinstance(name, str) and name else None

    def _array_literal_ints(self, expr: Any) -> Optional[List[int]]:
        if not isinstance(expr, ArrayLiteralIR):
            return None
        out: List[int] = []
        for elem in expr.elements or ():
            if not isinstance(elem, LiteralIR) or not isinstance(elem.value, int):
                return None
            out.append(int(elem.value))
        return out

    def _pool_dim_expr(self, input_dim: ExpressionIR, kernel: int, stride: int, pad: int, loc: Any) -> ExpressionIR:
        i32 = PrimitiveType("i32")
        two_pad = LiteralIR(int(2 * pad), loc, type_info=i32)
        kernel_lit = LiteralIR(int(kernel), loc, type_info=i32)
        stride_lit = LiteralIR(int(stride), loc, type_info=i32)
        one_lit = LiteralIR(1, loc, type_info=i32)
        numerator = BinaryOpIR(
            BinaryOp.SUB,
            BinaryOpIR(
                BinaryOp.ADD,
                _clone_expr(input_dim),
                two_pad,
                loc,
                type_info=i32,
            ),
            kernel_lit,
            loc,
            type_info=i32,
        )
        return BinaryOpIR(
            BinaryOp.ADD,
            BinaryOpIR(BinaryOp.DIV, numerator, stride_lit, loc, type_info=i32),
            one_lit,
            loc,
            type_info=i32,
        )

    def _infer_expr_shape(self, expr: Optional[ExpressionIR], seen: Optional[set] = None) -> Tuple[ExpressionIR, ...]:
        if expr is None:
            return ()
        shape = getattr(expr, "shape_info", None)
        if isinstance(shape, tuple):
            return tuple(_clone_expr(dim) for dim in shape if dim is not None)
        if isinstance(shape, list):
            return tuple(_clone_expr(dim) for dim in shape if dim is not None)
        if isinstance(expr, IdentifierIR) and expr.defid is not None:
            if seen is None:
                seen = set()
            if expr.defid in seen:
                return ()
            seen.add(expr.defid)
            binding = self._binding_map.get(expr.defid)
            return self._binding_shape(binding, seen)
        if isinstance(expr, FunctionCallIR):
            base_name = self._function_base_name(expr) or ""
            args = list(expr.arguments or ())
            if base_name.startswith("relu") and args:
                return self._infer_expr_shape(args[0], seen)
            if (base_name.startswith("max_pool") or base_name.startswith("average_pool")) and len(args) >= 4:
                input_shape = self._infer_expr_shape(args[0], seen)
                kernel = self._array_literal_ints(args[1])
                strides = self._array_literal_ints(args[2])
                pads = self._array_literal_ints(args[3])
                if input_shape and kernel is not None and strides is not None and pads is not None and len(input_shape) >= 2:
                    spatial_rank = len(kernel)
                    if len(strides) == spatial_rank and len(pads) == spatial_rank and len(input_shape) >= 2 + spatial_rank:
                        prefix = tuple(_clone_expr(dim) for dim in input_shape[:2])
                        spatial = tuple(
                            self._pool_dim_expr(input_shape[2 + axis], kernel[axis], strides[axis], pads[axis], expr.location)
                            for axis in range(spatial_rank)
                        )
                        return prefix + spatial
        return ()

    def _binding_shape(self, binding: Optional[BindingIR], seen: Optional[set] = None) -> Tuple[ExpressionIR, ...]:
        raw_shape = _binding_shape(binding)
        if binding is None:
            return ()

        expr = getattr(binding, "expr", None)
        type_shape = None
        type_info = getattr(expr, "type_info", None) if expr is not None else None
        if isinstance(type_info, RectangularType):
            type_shape = getattr(type_info, "shape", None)
        placeholder_zero_shape = bool(
            raw_shape
            and getattr(expr, "shape_info", None) is None
            and isinstance(type_shape, tuple)
            and any(dim is None for dim in type_shape)
            and all(isinstance(dim, LiteralIR) and getattr(dim, "value", None) == 0 for dim in raw_shape)
        )

        if not placeholder_zero_shape and raw_shape:
            return tuple(_clone_expr(dim) for dim in raw_shape)
        inferred = self._infer_expr_shape(expr, seen)
        if inferred:
            return tuple(_clone_expr(dim) for dim in inferred)
        if raw_shape:
            return tuple(_clone_expr(dim) for dim in raw_shape)
        return ()

    def _sum_contributions(
        self,
        binding: BindingIR,
        contributions: List[ExpressionIR],
        loc: Any,
    ) -> ExpressionIR:
        if not contributions:
            return self._default_seed(binding, loc, 0.0)
        acc = contributions[0]
        for expr in contributions[1:]:
            acc = _simplify(
                BinaryOpIR(
                    BinaryOp.ADD,
                    acc,
                    expr,
                    loc,
                    type_info=_tensor_type_info(binding),
                    shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(binding)) or (),
                )
            )
        return acc

    def _build_shared_scalar_target_bars(
        self,
        target_defid: DefId,
        wrt_defids: List[DefId],
        loc: Any,
        *,
        target_seed: Optional[ExpressionIR] = None,
    ) -> Tuple[List[BindingIR], Dict[DefId, IdentifierIR]]:
        scope_bindings: Dict[Any, BindingIR] = {}
        scope_bindings.update(self._local_contexts.get(target_defid) or {})
        target_binding_global = self._binding_map.get(target_defid)
        if target_binding_global is not None:
            scope_bindings[target_defid] = target_binding_global
        for wrt_defid in wrt_defids:
            wrt_binding = self._binding_map.get(wrt_defid)
            if wrt_binding is not None and wrt_defid not in scope_bindings:
                scope_bindings[wrt_defid] = wrt_binding
        ctx = _AutodiffBindingContext(bindings=scope_bindings, dep_cache=_DependencyQueryCache(scope_bindings))
        requested = set(wrt_defids)

        needed_memo: Dict[DefId, bool] = {}

        def needed(did: DefId) -> bool:
            if did in needed_memo:
                return needed_memo[did]
            if did in requested:
                needed_memo[did] = True
                return True
            needed_memo[did] = False
            for dep in ctx.deps_for(did):
                if dep in scope_bindings and needed(dep):
                    needed_memo[did] = True
                    return True
            return False

        if not needed(target_defid):
            return [], {}

        order: List[DefId] = []
        seen: Set[DefId] = set()

        def dfs(did: DefId) -> None:
            if did in seen:
                return
            seen.add(did)
            for dep in ctx.deps_for(did):
                if dep in scope_bindings and needed(dep):
                    dfs(dep)
            order.append(did)

        dfs(target_defid)
        target_first = list(reversed(order))

        bar_binding_by_defid: Dict[DefId, BindingIR] = {}
        bar_ident_by_defid: Dict[DefId, IdentifierIR] = {}
        for did in target_first:
            binding = scope_bindings[did]
            bar_defid = self._fresh_defid()
            bar_name = self._fresh_name(f"__ad_bar_{binding.name or 'tmp'}")
            ident = IdentifierIR(
                bar_name,
                loc,
                defid=bar_defid,
                type_info=_tensor_type_info(binding),
                shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(binding)) or (),
            )
            bar_ident_by_defid[did] = ident
            bar_binding_by_defid[did] = BindingIR(
                bar_name,
                None,
                location=loc,
                defid=bar_defid,
                type_info=_tensor_type_info(binding),
            )

        contributions: Dict[DefId, List[ExpressionIR]] = {did: [] for did in target_first}
        target_binding = scope_bindings[target_defid]
        contributions[target_defid].append(
            target_seed if target_seed is not None else self._default_seed(target_binding, loc, 1.0)
        )

        emitted: List[BindingIR] = []
        for did in target_first:
            binding = scope_bindings[did]
            bar_expr = self._sum_contributions(binding, contributions.get(did, []), loc)
            bar_expr = self._simplify_autodiff_tensor_ir(bar_expr, loc)
            emitted.append(
                BindingIR(
                    bar_ident_by_defid[did].name,
                    bar_expr,
                    location=loc,
                    defid=bar_ident_by_defid[did].defid,
                    type_info=_tensor_type_info(binding),
                )
            )
            for dep in ctx.deps_for(did):
                if dep not in scope_bindings or not needed(dep):
                    continue
                source_binding = scope_bindings[did]
                dep_binding = scope_bindings[dep]
                contribution = self._lower_vjp(
                    VjpIR(
                        target=_binding_identifier(source_binding),
                        wrt=_binding_identifier(dep_binding),
                        location=loc,
                        cotangent=_clone_expr(bar_ident_by_defid[did]),
                        type_info=_tensor_type_info(dep_binding),
                        shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(dep_binding)),
                    ),
                    cotangent_expr=_clone_expr(bar_ident_by_defid[did]),
                    emit_intermediate_bindings=False,
                    prefer_factored_scalar_target=False,
                )
                contributions.setdefault(dep, []).append(contribution)
        return emitted, {did: bar_ident_by_defid[did] for did in requested if did in bar_ident_by_defid}

    def rewrite_statement_list(self, statements: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        for stmt in list(statements):
            out.extend(self.rewrite_statement(stmt))
        return out

    def rewrite_statement(self, stmt: Any) -> List[Any]:
        if isinstance(stmt, BindingIR):
            original_expr = self._original_expr_by_defid.get(getattr(stmt, "defid", None))
            if stmt.defid in self._removable_lazy_binding_defids and isinstance(original_expr, LazyJacobianIR):
                return []
            if stmt.expr is not None:
                stmt.expr = self.rewrite_expr(stmt.expr)
            if stmt.defid is not None:
                self._binding_map[stmt.defid] = stmt
                self._differentiator._ctx.bindings[stmt.defid] = stmt
                self._differentiator._ctx.binding_deps.pop(stmt.defid, None)
                for context in self._local_contexts.values():
                    if stmt.defid in context:
                        context[stmt.defid] = stmt
            return [stmt]
        if isinstance(stmt, ExpressionIR):
            return [self.rewrite_expr(stmt)]
        return [stmt]

    def _fresh_defid(self) -> DefId:
        if self._resolver is not None:
            return self._resolver.allocate_for_local()
        return _template_defid(self._counter)

    def _fresh_name(self, prefix: str) -> str:
        idx = self._counter[0]
        self._counter[0] += 1
        return f"{prefix}_{idx}"

    def _default_seed(self, binding: Optional[BindingIR], loc: Any, value: float) -> ExpressionIR:
        return _tensor_constant_like(binding, self._resolver, loc, value)

    def _make_seed_placeholder(self, name: str, binding: Optional[BindingIR], loc: Any) -> Tuple[DefId, IdentifierIR, BindingIR]:
        seed_defid = self._fresh_defid()
        seed_ident = IdentifierIR(
            name,
            loc,
            defid=seed_defid,
            type_info=_tensor_type_info(binding),
            shape_info=getattr(getattr(binding, "expr", None), "shape_info", None) if binding is not None else None,
        )
        seed_binding = BindingIR(
            name,
            None,
            type_info=_tensor_type_info(binding),
            location=loc,
            defid=seed_defid,
        )
        return seed_defid, seed_ident, seed_binding

    def _differentiate_with_seed(
        self,
        expr: ExpressionIR,
        wrt_binding: BindingIR,
        seed_expr: ExpressionIR,
        loc: Any,
        *,
        local_bindings: Optional[Dict[DefId, BindingIR]] = None,
        fallback_type: Any = None,
        fallback_shape: Any = None,
        emit_intermediate_bindings: bool = True,
    ) -> ExpressionIR:
        seed_defid, seed_ident, seed_binding = self._make_seed_placeholder("__autodiff_tangent", wrt_binding, loc)
        scoped_locals = dict(local_bindings or {})
        scoped_locals[seed_defid] = seed_binding
        emitted_bindings: Optional[List[BindingIR]] = [] if emit_intermediate_bindings else None
        template = self._differentiator.differentiate_expr(
            expr,
            wrt_binding.defid,
            loc,
            symbolic=False,
            local_bindings=scoped_locals,
            seed_override=seed_ident,
            emitted_bindings=emitted_bindings,
        )
        if emitted_bindings:
            template = BlockExpressionIR(
                list(emitted_bindings),
                loc,
                template,
                type_info=getattr(template, "type_info", fallback_type),
                shape_info=getattr(template, "shape_info", fallback_shape),
            )
        lowered = _simplify(_substitute_identifiers(template, {seed_defid: seed_expr}))
        return _stamp_expr_metadata(
            lowered,
            self._binding_map,
            fallback_type=fallback_type,
            fallback_shape=fallback_shape,
        )

    def _inline_basis_tensor_accesses(self, expr: ExpressionIR, loc: Any) -> ExpressionIR:
        def rewrite(node: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
            if node is None:
                return None
            if isinstance(node, RectangularAccessIR):
                array = rewrite(node.array) if isinstance(node.array, ExpressionIR) else node.array
                indices = tuple(rewrite(idx) if isinstance(idx, ExpressionIR) else idx for idx in (node.indices or ()))
                selected_axes = _builtin_basis_selected_axes(array) if isinstance(array, ExpressionIR) else None
                if selected_axes is not None:
                    if len(selected_axes) != len(indices):
                        return LiteralIR(0.0, node.location or loc, type_info=PrimitiveType("f32"))
                    cond: Optional[ExpressionIR] = None
                    for lhs, rhs in zip(indices, selected_axes):
                        eq = BinaryOpIR(
                            BinaryOp.EQ,
                            _clone_expr(lhs),
                            _clone_expr(rhs),
                            node.location or loc,
                            type_info=PrimitiveType("bool"),
                            shape_info=(),
                        )
                        cond = eq if cond is None else BinaryOpIR(
                            BinaryOp.AND,
                            cond,
                            eq,
                            node.location or loc,
                            type_info=PrimitiveType("bool"),
                            shape_info=(),
                        )
                    return IfExpressionIR(
                        cond or LiteralIR(True, node.location or loc, type_info=PrimitiveType("bool")),
                        LiteralIR(1.0, node.location or loc, type_info=PrimitiveType("f32")),
                        node.location or loc,
                        else_expr=LiteralIR(0.0, node.location or loc, type_info=PrimitiveType("f32")),
                        type_info=PrimitiveType("f32"),
                        shape_info=(),
                    )
                return RectangularAccessIR(
                    array,
                    list(indices),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, BinaryOpIR):
                return BinaryOpIR(
                    node.operator,
                    rewrite(node.left),
                    rewrite(node.right),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, UnaryOpIR):
                return UnaryOpIR(
                    node.operator,
                    rewrite(node.operand),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, IfExpressionIR):
                return IfExpressionIR(
                    rewrite(node.condition),
                    rewrite(node.then_expr),
                    node.location,
                    else_expr=rewrite(node.else_expr) if node.else_expr is not None else None,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, BuiltinCallIR):
                return BuiltinCallIR(
                    builtin_name=node.builtin_name,
                    args=tuple(rewrite(arg) for arg in (node.args or ())),
                    location=node.location,
                    defid=node.defid,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, BlockExpressionIR):
                return BlockExpressionIR(
                    [
                        stmt if not isinstance(stmt, BindingIR) else BindingIR(
                            stmt.name,
                            rewrite(stmt.expr) if stmt.expr is not None else None,
                            location=stmt.location,
                            defid=stmt.defid,
                            type_info=stmt.type_info,
                        )
                        for stmt in (node.statements or ())
                    ],
                    node.location,
                    rewrite(node.final_expr) if node.final_expr is not None else None,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, ReductionExpressionIR):
                return ReductionExpressionIR(
                    node.operation,
                    list(node.loop_vars or ()),
                    rewrite(node.body),
                    node.location,
                    where_clause=node.where_clause,
                    loop_var_ranges=dict(node.loop_var_ranges or {}),
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, EinsteinIR):
                return EinsteinIR(
                    clauses=[
                        EinsteinClauseIR(
                            indices=[_clone_expr(idx) for idx in (clause.indices or ())],
                            value=rewrite(clause.value),
                            location=clause.location,
                            where_clause=clause.where_clause,
                            variable_ranges=dict(clause.variable_ranges or {}),
                        )
                        for clause in (node.clauses or ())
                    ],
                    shape=list(node.shape or ()) if node.shape is not None else None,
                    element_type=node.element_type,
                    location=node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, CastExpressionIR):
                return CastExpressionIR(
                    rewrite(node.expr),
                    node.target_type,
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            return node

        rewritten = rewrite(expr)
        return rewritten if rewritten is not None else expr

    def _simplify_autodiff_tensor_ir(
        self,
        expr: ExpressionIR,
        loc: Any,
        *,
        protected_tensor_defids: Optional[Set[DefId]] = None,
    ) -> ExpressionIR:
        protected_tensor_defids = set(protected_tensor_defids or ())

        def flatten_and(node: ExpressionIR) -> List[ExpressionIR]:
            if isinstance(node, BinaryOpIR) and node.operator == BinaryOp.AND:
                return flatten_and(node.left) + flatten_and(node.right)
            return [node]

        def rebuild_and(terms: List[ExpressionIR]) -> Optional[ExpressionIR]:
            if not terms:
                return None
            out = terms[0]
            for term in terms[1:]:
                out = BinaryOpIR(
                    BinaryOp.AND,
                    out,
                    term,
                    loc,
                    type_info=PrimitiveType("bool"),
                    shape_info=(),
                )
            return out

        def simplify_reduction(node: ReductionExpressionIR) -> Optional[ExpressionIR]:
            loop_vars = list(node.loop_vars or ())
            if len(loop_vars) != 1:
                return None
            loop_var = loop_vars[0]
            loop_defid = getattr(loop_var, "defid", None)
            if loop_defid is None:
                return None
            body = node.body
            if not isinstance(body, BinaryOpIR) or body.operator != BinaryOp.MUL:
                return None
            indicator = _is_one_zero_indicator(body.left)
            term = body.right
            if indicator is None:
                indicator = _is_one_zero_indicator(body.right)
                term = body.left
            if indicator is None:
                return None
            conditions = flatten_and(indicator)
            matched_expr: Optional[ExpressionIR] = None
            remaining: List[ExpressionIR] = []
            for cond in conditions:
                if (
                    isinstance(cond, BinaryOpIR)
                    and cond.operator == BinaryOp.EQ
                    and (
                        getattr(cond.left, "defid", None) == loop_defid
                        or getattr(cond.right, "defid", None) == loop_defid
                    )
                ):
                    lhs_has = getattr(cond.left, "defid", None) == loop_defid
                    rhs_has = getattr(cond.right, "defid", None) == loop_defid
                    if lhs_has and not rhs_has and loop_defid not in _collect_defids(cond.right):
                        matched_expr = cond.right
                        continue
                    if rhs_has and not lhs_has and loop_defid not in _collect_defids(cond.left):
                        matched_expr = cond.left
                        continue
                remaining.append(cond)
            if matched_expr is None:
                return None
            substituted = _substitute_identifiers(term, {loop_defid: _clone_expr(matched_expr)})
            guard = rebuild_and(remaining)
            if guard is None:
                return substituted
            return IfExpressionIR(
                guard,
                substituted,
                node.location or loc,
                else_expr=LiteralIR(0.0, node.location or loc, type_info=PrimitiveType("f32")),
                type_info=getattr(substituted, "type_info", None) or PrimitiveType("f32"),
                shape_info=getattr(substituted, "shape_info", None) or (),
            )

        def rewrite(node: Optional[ExpressionIR], local_exprs: Optional[Dict[DefId, ExpressionIR]] = None) -> Optional[ExpressionIR]:
            local_exprs = dict(local_exprs or {})
            if node is None:
                return None
            if isinstance(node, IdentifierIR):
                repl = local_exprs.get(getattr(node, "defid", None))
                if repl is not None:
                    return rewrite(_clone_expr(repl), local_exprs)
                return node
            if isinstance(node, RectangularAccessIR):
                array = rewrite(node.array, local_exprs) if isinstance(node.array, ExpressionIR) else node.array
                indices = tuple(rewrite(idx, local_exprs) if isinstance(idx, ExpressionIR) else idx for idx in (node.indices or ()))
                if isinstance(array, EinsteinIR):
                    clauses = list(array.clauses or ())
                    if len(clauses) == 1:
                        clause = clauses[0]
                        if not (clause.where_clause and clause.where_clause.constraints):
                            clause_indices = list(clause.indices or ())
                            if len(indices) <= len(clause_indices):
                                repl: Dict[DefId, ExpressionIR] = {}
                                ok = True
                                for idx_expr, access_expr in zip(clause_indices[: len(indices)], indices):
                                    did = getattr(idx_expr, "defid", None)
                                    if did is None:
                                        if not _collect_defids(idx_expr):
                                            if not _expr_eq(idx_expr, access_expr):
                                                ok = False
                                                break
                                        else:
                                            ok = False
                                            break
                                    else:
                                        repl[did] = _clone_expr(access_expr)
                                if ok:
                                    substituted = _substitute_identifiers(clause.value, repl)
                                    if len(indices) == len(clause_indices):
                                        return rewrite(substituted)
                                    remaining_indices = [_clone_expr(idx) for idx in clause_indices[len(indices) :]]
                                    remaining_ranges = {
                                        did: _clone_expr(rng)
                                        for did, rng in (clause.variable_ranges or {}).items()
                                        if did in {
                                            getattr(idx, "defid", None)
                                            for idx in clause_indices[len(indices) :]
                                            if getattr(idx, "defid", None) is not None
                                        }
                                    }
                                    residual = EinsteinIR(
                                        clauses=[
                                            EinsteinClauseIR(
                                                indices=remaining_indices,
                                                value=substituted,
                                                location=clause.location,
                                                where_clause=clause.where_clause,
                                                variable_ranges=remaining_ranges,
                                            )
                                        ],
                                        shape=list((array.shape or ())[len(indices) :]) if array.shape is not None else None,
                                        element_type=array.element_type,
                                        location=array.location,
                                        type_info=array.type_info,
                                        shape_info=tuple((array.shape or ())[len(indices) :]) if array.shape is not None else array.shape_info,
                                    )
                                    return rewrite(residual)
                access_target: Optional[ExpressionIR] = array if isinstance(array, ExpressionIR) else None
                if isinstance(array, IdentifierIR):
                    array_defid = getattr(array, "defid", None)
                    binding = None
                    if array_defid not in protected_tensor_defids:
                        binding = self._binding_map.get(array_defid)
                    bound_expr = getattr(binding, "expr", None) if binding is not None else None
                    if isinstance(bound_expr, ExpressionIR):
                        access_target = bound_expr
                clause_info = _single_clause_info(access_target) if access_target is not None else None
                if clause_info is not None:
                    clause_indices, clause_value = clause_info
                    if len(clause_indices) == len(indices):
                        repl: Dict[DefId, ExpressionIR] = {}
                        ok = True
                        for idx_expr, access_expr in zip(clause_indices, indices):
                            did = getattr(idx_expr, "defid", None)
                            if did is None:
                                if not _collect_defids(idx_expr):
                                    if not _expr_eq(idx_expr, access_expr):
                                        ok = False
                                        break
                                else:
                                    ok = False
                                    break
                            else:
                                repl[did] = _clone_expr(access_expr)
                        if ok:
                            return rewrite(_substitute_identifiers(clause_value, repl))
                return RectangularAccessIR(
                    array,
                    list(indices),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, BinaryOpIR):
                return BinaryOpIR(
                    node.operator,
                    rewrite(node.left, local_exprs),
                    rewrite(node.right, local_exprs),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, UnaryOpIR):
                return UnaryOpIR(
                    node.operator,
                    rewrite(node.operand, local_exprs),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, IfExpressionIR):
                return IfExpressionIR(
                    rewrite(node.condition, local_exprs),
                    rewrite(node.then_expr, local_exprs),
                    node.location,
                    else_expr=rewrite(node.else_expr, local_exprs) if node.else_expr is not None else None,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, ReductionExpressionIR):
                rebuilt = ReductionExpressionIR(
                    node.operation,
                    list(node.loop_vars or ()),
                    rewrite(node.body, local_exprs),
                    node.location,
                    where_clause=node.where_clause,
                    loop_var_ranges=dict(node.loop_var_ranges or {}),
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
                if node.operation == "sum" or getattr(node.operation, "value", None) == "sum":
                    simplified = simplify_reduction(rebuilt)
                    if simplified is not None:
                        return rewrite(simplified)
                return rebuilt
            if isinstance(node, BlockExpressionIR):
                new_statements: List[Any] = []
                block_locals = dict(local_exprs)
                for stmt in (node.statements or ()):
                    if isinstance(stmt, BindingIR):
                        rewritten_expr = rewrite(stmt.expr, block_locals) if stmt.expr is not None else None
                        new_stmt = BindingIR(
                            stmt.name,
                            rewritten_expr,
                            location=stmt.location,
                            defid=stmt.defid,
                            type_info=stmt.type_info,
                        )
                        new_statements.append(new_stmt)
                        if stmt.defid is not None and rewritten_expr is not None:
                            block_locals[stmt.defid] = rewritten_expr
                    else:
                        new_statements.append(rewrite(stmt, block_locals) if isinstance(stmt, ExpressionIR) else stmt)
                final_expr = rewrite(node.final_expr, block_locals) if node.final_expr is not None else None
                live_defids = _collect_defids(final_expr)
                pruned_statements_rev: List[Any] = []
                for stmt in reversed(new_statements):
                    if isinstance(stmt, BindingIR):
                        stmt_defid = getattr(stmt, "defid", None)
                        if stmt_defid is None:
                            pruned_statements_rev.append(stmt)
                            continue
                        if stmt_defid not in live_defids:
                            continue
                        if stmt.expr is not None:
                            live_defids.update(_collect_defids(stmt.expr))
                        pruned_statements_rev.append(stmt)
                        continue
                    if isinstance(stmt, ExpressionIR):
                        live_defids.update(_collect_defids(stmt))
                    pruned_statements_rev.append(stmt)
                pruned_statements_rev.reverse()
                return BlockExpressionIR(
                    pruned_statements_rev,
                    node.location,
                    final_expr,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            if isinstance(node, EinsteinIR):
                return EinsteinIR(
                    clauses=[
                        EinsteinClauseIR(
                            indices=[_clone_expr(idx) for idx in (clause.indices or ())],
                            value=rewrite(clause.value, local_exprs),
                            location=clause.location,
                            where_clause=clause.where_clause,
                            variable_ranges=dict(clause.variable_ranges or {}),
                        )
                        for clause in (node.clauses or ())
                    ],
                    shape=list(node.shape or ()) if node.shape is not None else None,
                    element_type=node.element_type,
                    location=node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
            return node

        rewritten = rewrite(expr)
        return _simplify(rewritten if rewritten is not None else expr)

    def _hoist_tensor_arrays_from_reductions(self, expr: ExpressionIR) -> ExpressionIR:
        def make_rewriter(local_blocked: Set[DefId], bindings: List[BindingIR], memo: Dict[int, IdentifierIR], inherited_blocked: Set[DefId]):
            def rewrite(cur: ExpressionIR) -> ExpressionIR:
                if isinstance(cur, RectangularAccessIR):
                    arr = rewrite(cur.array) if isinstance(cur.array, ExpressionIR) else cur.array
                    idxs = [rewrite(idx) if isinstance(idx, ExpressionIR) else idx for idx in (cur.indices or ())]
                    candidate = arr
                    if (
                        isinstance(candidate, (EinsteinIR, BlockExpressionIR, IfExpressionIR))
                        and not (_collect_defids(candidate) & local_blocked)
                    ):
                        key = id(candidate)
                        ident = memo.get(key)
                        if ident is None:
                            did = self._fresh_defid()
                            name = self._fresh_name("__ad_hoist")
                            ident = IdentifierIR(
                                name,
                                candidate.location or cur.location,
                                defid=did,
                                type_info=getattr(candidate, "type_info", None),
                                shape_info=getattr(candidate, "shape_info", None),
                            )
                            memo[key] = ident
                            bindings.append(
                                BindingIR(
                                    name,
                                    candidate,
                                    location=candidate.location or cur.location,
                                    defid=did,
                                    type_info=getattr(candidate, "type_info", None),
                                )
                            )
                        return RectangularAccessIR(
                            _clone_expr(ident),
                            idxs,
                            cur.location,
                            type_info=cur.type_info,
                            shape_info=cur.shape_info,
                        )
                    return RectangularAccessIR(arr, idxs, cur.location, type_info=cur.type_info, shape_info=cur.shape_info)
                if isinstance(cur, BinaryOpIR):
                    return BinaryOpIR(cur.operator, rewrite(cur.left), rewrite(cur.right), cur.location, type_info=cur.type_info, shape_info=cur.shape_info)
                if isinstance(cur, UnaryOpIR):
                    return UnaryOpIR(cur.operator, rewrite(cur.operand), cur.location, type_info=cur.type_info, shape_info=cur.shape_info)
                if isinstance(cur, IfExpressionIR):
                    return IfExpressionIR(
                        rewrite(cur.condition),
                        rewrite(cur.then_expr),
                        cur.location,
                        else_expr=rewrite(cur.else_expr) if cur.else_expr is not None else None,
                        type_info=cur.type_info,
                        shape_info=cur.shape_info,
                    )
                if isinstance(cur, BuiltinCallIR):
                    return BuiltinCallIR(
                        builtin_name=cur.builtin_name,
                        args=tuple(rewrite(arg) for arg in (cur.args or ())),
                        location=cur.location,
                        defid=cur.defid,
                        type_info=cur.type_info,
                        shape_info=cur.shape_info,
                    )
                    if isinstance(cur, FunctionCallIR):
                        rebuilt = FunctionCallIR(
                            callee_expr=rewrite(cur.callee_expr),
                            location=cur.location,
                            arguments=tuple(rewrite(arg) for arg in (cur.arguments or ())),
                            module_path=cur.module_path,
                            type_info=cur.type_info,
                            shape_info=cur.shape_info,
                            coordinate_args=tuple(rewrite(arg) for arg in (getattr(cur, "coordinate_args", ()) or ())),
                        )
                        rebuilt.coordinate_layout = getattr(cur, "coordinate_layout", None)
                        rebuilt.coordinate_address_domain = getattr(cur, "coordinate_address_domain", None)
                        return rebuilt
                if isinstance(cur, BlockExpressionIR):
                    return BlockExpressionIR(
                        [stmt if not isinstance(stmt, ExpressionIR) else rewrite(stmt) for stmt in (cur.statements or ())],
                        cur.location,
                        rewrite(cur.final_expr) if cur.final_expr is not None else None,
                        type_info=cur.type_info,
                        shape_info=cur.shape_info,
                    )
                if isinstance(cur, ReductionExpressionIR):
                    return transform(cur, inherited_blocked) or cur
                if isinstance(cur, ArrayComprehensionIR):
                    return transform(cur, inherited_blocked) or cur
                if isinstance(cur, EinsteinIR):
                    return transform(cur, inherited_blocked) or cur
                return cur
            return rewrite

        def transform(node: Optional[ExpressionIR], blocked: Set[DefId]) -> Optional[ExpressionIR]:
            if node is None:
                return None
            if isinstance(node, ReductionExpressionIR):
                local_blocked = {
                    getattr(loop_var, "defid", None)
                    for loop_var in (node.loop_vars or ())
                    if getattr(loop_var, "defid", None) is not None
                }
                body = transform(node.body, blocked | local_blocked)
                if body is None:
                    return node
                bindings: List[BindingIR] = []
                memo: Dict[int, IdentifierIR] = {}
                rewrite = make_rewriter(local_blocked, bindings, memo, blocked | local_blocked)
                rebuilt = ReductionExpressionIR(
                    node.operation,
                    list(node.loop_vars or ()),
                    rewrite(body),
                    node.location,
                    where_clause=node.where_clause,
                    loop_var_ranges=dict(node.loop_var_ranges or {}),
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
                if bindings:
                    return BlockExpressionIR(bindings, node.location, rebuilt, type_info=rebuilt.type_info, shape_info=rebuilt.shape_info)
                return rebuilt
            if isinstance(node, ArrayComprehensionIR):
                local_blocked = {
                    getattr(loop_var, "defid", None)
                    for loop_var in (node.loop_vars or ())
                    if getattr(loop_var, "defid", None) is not None
                }
                body = transform(node.body, blocked | local_blocked)
                if body is None:
                    return node
                bindings: List[BindingIR] = []
                memo: Dict[int, IdentifierIR] = {}
                rewrite = make_rewriter(local_blocked, bindings, memo, blocked | local_blocked)
                rebuilt = ArrayComprehensionIR(
                    rewrite(body),
                    list(node.loop_vars or ()),
                    list(node.ranges or ()),
                    node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                    constraints=list(node.constraints or ()),
                )
                if bindings:
                    return BlockExpressionIR(bindings, node.location, rebuilt, type_info=rebuilt.type_info, shape_info=rebuilt.shape_info)
                return rebuilt
            if isinstance(node, EinsteinIR):
                bindings: List[BindingIR] = []
                memo: Dict[int, IdentifierIR] = {}
                new_clauses: List[EinsteinClauseIR] = []
                for clause in node.clauses or ():
                    local_blocked = {
                        getattr(idx, "defid", None)
                        for idx in (clause.indices or ())
                        if getattr(idx, "defid", None) is not None
                    }
                    value = transform(clause.value, blocked | local_blocked)
                    rewrite = make_rewriter(local_blocked, bindings, memo, blocked | local_blocked)
                    new_clause = EinsteinClauseIR(
                        indices=[_clone_expr(idx) for idx in (clause.indices or ())],
                        value=rewrite(value) if value is not None else None,
                        location=clause.location,
                        where_clause=clause.where_clause,
                        variable_ranges=dict(clause.variable_ranges or {}),
                    )
                    new_clauses.append(new_clause)
                rebuilt = EinsteinIR(
                    clauses=new_clauses,
                    shape=list(node.shape or ()) if node.shape is not None else None,
                    element_type=node.element_type,
                    location=node.location,
                    type_info=node.type_info,
                    shape_info=node.shape_info,
                )
                if bindings:
                    return BlockExpressionIR(bindings, node.location, rebuilt, type_info=rebuilt.type_info, shape_info=rebuilt.shape_info)
                return rebuilt
            return node

        transformed = transform(expr, set())
        return transformed if transformed is not None else expr

    def _normalize_tensor_body(
        self,
        expr: ExpressionIR,
        shape: Tuple[ExpressionIR, ...],
        loc: Any,
        *,
        type_info: Any,
    ) -> ExpressionIR:
        if not shape:
            return expr
        temp_defid = self._fresh_defid()
        temp_name = self._fresh_name("__autodiff_tensor_body")
        temp_binding = BindingIR(
            temp_name,
            expr,
            location=loc,
            defid=temp_defid,
            type_info=type_info,
        )
        temp_ident = IdentifierIR(
            temp_name,
            loc,
            defid=temp_defid,
            type_info=type_info,
            shape_info=tuple(_clone_expr(dim) for dim in shape),
        )
        axes = self._make_comprehension_axes(shape, "_ad_body_", loc)
        axis_idents = [ident for ident, _rng, _dim in axes]
        tensor_expr = self._nest_array_comprehensions(
            RectangularAccessIR(
                temp_ident,
                list(axis_idents),
                loc,
                type_info=_tensor_element_type(_binding_by_identifier(temp_ident, self._binding_map), PrimitiveType("f32")),
                shape_info=(),
            ),
            axes,
            loc,
            tuple(_clone_expr(dim) for dim in shape),
            type_info,
        )
        return _stamp_expr_metadata(
            BlockExpressionIR(
                [temp_binding],
                loc,
                tensor_expr,
                type_info=type_info,
                shape_info=tuple(_clone_expr(dim) for dim in shape),
            ),
            self._binding_map,
            fallback_type=type_info,
            fallback_shape=tuple(_clone_expr(dim) for dim in shape),
        )

    def _lower_jvp(self, node: JvpIR, tangent_expr: Optional[ExpressionIR]) -> ExpressionIR:
        target_binding = _binding_by_identifier(node.target, self._binding_map)
        wrt_binding = _binding_by_identifier(node.wrt, self._binding_map)
        if target_binding is None or wrt_binding is None or target_binding.expr is None:
            return node
        loc = node.location
        tangent_value = tangent_expr or self._default_seed(wrt_binding, loc, 1.0)
        local_bindings = dict(self._local_contexts.get(target_binding.defid) or {})
        return self._differentiate_with_seed(
            target_binding.expr,
            wrt_binding,
            tangent_value,
            loc,
            local_bindings=local_bindings,
            fallback_type=node.type_info,
            fallback_shape=node.shape_info,
            emit_intermediate_bindings=True,
        )

    def _lower_vjp(
        self,
        node: VjpIR,
        cotangent_expr: Optional[ExpressionIR],
        *,
        emit_intermediate_bindings: bool = True,
        prefer_factored_scalar_target: bool = True,
    ) -> ExpressionIR:
        target_binding = _binding_by_identifier(node.target, self._binding_map)
        wrt_binding = _binding_by_identifier(node.wrt, self._binding_map)
        if target_binding is None or wrt_binding is None or target_binding.expr is None:
            return node
        loc = node.location
        cotangent_value = cotangent_expr or self._default_seed(target_binding, loc, 1.0)
        cot_defid, cot_ident, cot_binding = self._make_seed_placeholder("__autodiff_cotangent", target_binding, loc)
        local_bindings = dict(self._local_contexts.get(target_binding.defid) or {})
        local_bindings[cot_defid] = cot_binding
        objective = _make_cotangent_objective(target_binding, cot_ident, self._counter)
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        target_expr = getattr(target_binding, "expr", None)
        clause_info = _single_clause_info(target_expr) if isinstance(target_expr, ExpressionIR) else None
        if clause_info is not None and target_shape == wrt_shape:
            clause_indices, clause_value = clause_info
            if _matches_identity_rect_access(clause_value, wrt_binding, clause_indices):
                return _stamp_expr_metadata(
                    _clone_expr(cotangent_value),
                    self._binding_map,
                    fallback_type=node.type_info,
                    fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                )
            if isinstance(clause_value, BinaryOpIR):
                left_is_id = _matches_identity_rect_access(clause_value.left, wrt_binding, clause_indices)
                right_is_id = _matches_identity_rect_access(clause_value.right, wrt_binding, clause_indices)
                left_depends = wrt_binding.defid in _collect_defids(clause_value.left)
                right_depends = wrt_binding.defid in _collect_defids(clause_value.right)
                if clause_value.operator == BinaryOp.ADD:
                    if left_is_id and not right_depends:
                        return _stamp_expr_metadata(
                            _clone_expr(cotangent_value),
                            self._binding_map,
                            fallback_type=node.type_info,
                            fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                        )
                    if right_is_id and not left_depends:
                        return _stamp_expr_metadata(
                            _clone_expr(cotangent_value),
                            self._binding_map,
                            fallback_type=node.type_info,
                            fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                        )
                if clause_value.operator == BinaryOp.SUB:
                    if left_is_id and not right_depends:
                        return _stamp_expr_metadata(
                            _clone_expr(cotangent_value),
                            self._binding_map,
                            fallback_type=node.type_info,
                            fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                        )
                    if right_is_id and not left_depends:
                        return _stamp_expr_metadata(
                            UnaryOpIR(
                                UnaryOp.NEG,
                                _clone_expr(cotangent_value),
                                loc,
                                type_info=node.type_info,
                                shape_info=tuple(_clone_expr(dim) for dim in wrt_shape),
                            ),
                            self._binding_map,
                            fallback_type=node.type_info,
                            fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                        )
        if prefer_factored_scalar_target and not target_shape and wrt_shape:
            emitted, bar_ident_by_defid = self._build_shared_scalar_target_bars(
                target_binding.defid,
                [wrt_binding.defid],
                loc,
                target_seed=_clone_expr(cotangent_value),
            )
            wrt_bar = bar_ident_by_defid.get(wrt_binding.defid)
            if wrt_bar is not None:
                lowered: ExpressionIR = wrt_bar
                if emitted and emit_intermediate_bindings:
                    lowered = BlockExpressionIR(
                        emitted,
                        loc,
                        lowered,
                        type_info=_tensor_type_info(wrt_binding, node.type_info),
                        shape_info=tuple(_clone_expr(dim) for dim in wrt_shape),
                    )
                lowered = self._inline_basis_tensor_accesses(lowered, loc)
                lowered = self._simplify_autodiff_tensor_ir(
                    lowered,
                    loc,
                    protected_tensor_defids={wrt_binding.defid},
                )
                lowered = self._hoist_tensor_arrays_from_reductions(lowered)
                lowered = self._simplify_autodiff_tensor_ir(
                    lowered,
                    loc,
                    protected_tensor_defids={wrt_binding.defid},
                )
                return _stamp_expr_metadata(
                    lowered,
                    self._binding_map,
                    fallback_type=node.type_info,
                    fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
                )
        if len(self._binding_shape(wrt_binding)) == 0:
            lowered = self._differentiate_with_seed(
                objective,
                wrt_binding,
                self._default_seed(wrt_binding, loc, 1.0),
                loc,
                local_bindings=local_bindings,
                fallback_type=node.type_info,
                fallback_shape=node.shape_info,
                emit_intermediate_bindings=emit_intermediate_bindings,
            )
            lowered = _simplify(_substitute_identifiers(lowered, {cot_defid: cotangent_value}))
            return _stamp_expr_metadata(
                lowered,
                self._binding_map,
                fallback_type=node.type_info,
                fallback_shape=node.shape_info,
            )

        wrt_axes, variable_ranges = self._make_einstein_axes(wrt_shape, "_ad_wrt_", loc)
        gradient_component = self._differentiate_with_seed(
            objective,
            wrt_binding,
            self._make_basis_tensor(wrt_binding, wrt_axes, loc),
            loc,
            local_bindings=local_bindings,
            fallback_type=_tensor_element_type(wrt_binding, node.type_info),
            fallback_shape=(),
            emit_intermediate_bindings=emit_intermediate_bindings,
        )
        gradient_component = _simplify(_substitute_identifiers(gradient_component, {cot_defid: cotangent_value}))
        gradient_component = self._inline_basis_tensor_accesses(gradient_component, loc)
        gradient_component = self._hoist_tensor_arrays_from_reductions(gradient_component)
        gradient_component = self._simplify_autodiff_tensor_ir(
            gradient_component,
            loc,
            protected_tensor_defids={wrt_binding.defid},
        )
        lowered = EinsteinIR(
            clauses=[
                EinsteinClauseIR(
                    indices=list(wrt_axes),
                    value=gradient_component,
                    location=loc,
                    variable_ranges=variable_ranges,
                )
            ],
            shape=[_clone_expr(dim) for dim in wrt_shape],
            element_type=_tensor_element_type(wrt_binding, node.type_info),
            location=loc,
            type_info=node.type_info,
            shape_info=tuple(_clone_expr(dim) for dim in wrt_shape),
        )
        return _stamp_expr_metadata(
            lowered,
            self._binding_map,
            fallback_type=node.type_info,
            fallback_shape=tuple(_clone_expr(dim) for dim in wrt_shape),
        )

    def _choose_lazy_mode(self, target_binding: BindingIR, wrt_binding: BindingIR) -> str:
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        target_size = _shape_size_estimate(target_shape)
        wrt_size = _shape_size_estimate(wrt_shape)
        if target_size is not None and wrt_size is not None:
            return "jvp" if wrt_size < target_size else "vjp"
        return "jvp" if len(wrt_shape) < len(target_shape) else "vjp"

    def _make_comprehension_axes(
        self,
        shape: Tuple[ExpressionIR, ...],
        prefix: str,
        loc: Any,
    ) -> List[Tuple[IdentifierIR, RangeIR, ExpressionIR]]:
        axes: List[Tuple[IdentifierIR, RangeIR, ExpressionIR]] = []
        for axis, dim in enumerate(shape):
            defid = self._fresh_defid()
            ident = IdentifierIR(
                f"{prefix}{axis}",
                loc,
                defid=defid,
                type_info=PrimitiveType("i32"),
            )
            rng = RangeIR(
                LiteralIR(0, loc, type_info=PrimitiveType("i32")),
                _clone_expr(dim),
                loc,
                type_info=PrimitiveType("range"),
            )
            axes.append((ident, rng, _clone_expr(dim)))
        return axes

    def _nest_array_comprehensions(
        self,
        body: ExpressionIR,
        axes: List[Tuple[IdentifierIR, RangeIR, ExpressionIR]],
        loc: Any,
        final_shape: Tuple[ExpressionIR, ...],
        type_info: Any,
    ) -> ExpressionIR:
        expr: ExpressionIR = body
        shape_info = tuple(_clone_expr(dim) for dim in final_shape)
        for ident, rng, _dim in reversed(axes):
            expr = ArrayComprehensionIR(
                expr,
                [ident],
                [rng],
                loc,
                type_info=type_info,
                shape_info=shape_info,
            )
        return expr

    def _make_basis_tensor(
        self,
        binding: BindingIR,
        selected_axes: List[ExpressionIR],
        loc: Any,
    ) -> ExpressionIR:
        shape = self._binding_shape(binding)
        if not shape:
            return self._default_seed(binding, loc, 1.0)
        return BuiltinCallIR(
            builtin_name="__basis_tensor",
            args=(
                ArrayLiteralIR(
                    [_clone_expr(dim) for dim in shape],
                    loc,
                    type_info=PrimitiveType("i32"),
                ),
                ArrayLiteralIR(
                    [_clone_expr(selected) for selected in selected_axes],
                    loc,
                    type_info=PrimitiveType("i32"),
                ),
            ),
            location=loc,
            defid=fixed_builtin_defid("__basis_tensor"),
            type_info=_tensor_type_info(binding),
            shape_info=tuple(_clone_expr(dim) for dim in shape),
        )

    def _lower_lazy_row_via_vjp(
        self,
        expr: LazyJacobianIR,
        target_binding: BindingIR,
        wrt_binding: BindingIR,
        output_indices: List[ExpressionIR],
        loc: Any,
    ) -> ExpressionIR:
        cotangent_basis = self._make_basis_tensor(target_binding, output_indices, loc)
        return self._lower_vjp(
            VjpIR(
                target=_clone_expr(expr.target),
                wrt=_clone_expr(expr.wrt),
                location=loc,
                cotangent=cotangent_basis,
                type_info=_tensor_type_info(wrt_binding, expr.type_info),
                shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(wrt_binding)),
            ),
            cotangent_expr=cotangent_basis,
        )

    def _lower_lazy_column_via_jvp(
        self,
        expr: LazyJacobianIR,
        target_binding: BindingIR,
        wrt_binding: BindingIR,
        wrt_indices: List[ExpressionIR],
        loc: Any,
    ) -> ExpressionIR:
        tangent_basis = self._make_basis_tensor(wrt_binding, wrt_indices, loc)
        return self._lower_jvp(
            JvpIR(
                target=_clone_expr(expr.target),
                wrt=_clone_expr(expr.wrt),
                location=loc,
                tangent=tangent_basis,
                type_info=_tensor_type_info(target_binding, expr.type_info),
                shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(target_binding)),
            ),
            tangent_expr=tangent_basis,
        )

    def _rewrite_lazy_access(self, expr: RectangularAccessIR, lazy_expr: LazyJacobianIR) -> Optional[ExpressionIR]:
        target_binding = _binding_by_identifier(lazy_expr.target, self._binding_map)
        wrt_binding = _binding_by_identifier(lazy_expr.wrt, self._binding_map)
        if target_binding is None or wrt_binding is None:
            return None
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        target_rank = len(target_shape)
        wrt_rank = len(wrt_shape)
        if target_rank == 0:
            return None
        rewritten_indices = [
            rewritten
            for idx in (expr.indices or ())
            for rewritten in [self.rewrite_expr(idx)]
            if rewritten is not None
        ]
        if len(rewritten_indices) < target_rank or len(rewritten_indices) > target_rank + wrt_rank:
            return None
        loc = expr.location or lazy_expr.location
        out_indices = rewritten_indices[:target_rank]
        wrt_indices = rewritten_indices[target_rank:]

        if len(rewritten_indices) == target_rank + wrt_rank and wrt_indices:
            mode = self._choose_lazy_mode(target_binding, wrt_binding)
            if mode == "jvp":
                column_expr = self._lower_lazy_column_via_jvp(
                    lazy_expr,
                    target_binding,
                    wrt_binding,
                    list(wrt_indices),
                    loc,
                )
                return self.rewrite_expr(
                    RectangularAccessIR(
                        column_expr,
                        list(out_indices),
                        loc,
                        type_info=expr.type_info,
                        shape_info=expr.shape_info,
                    )
                )

        row_expr = self._lower_lazy_row_via_vjp(
            lazy_expr,
            target_binding,
            wrt_binding,
            list(out_indices),
            loc,
        )
        if wrt_indices:
            return self.rewrite_expr(
                RectangularAccessIR(
                    row_expr,
                    list(wrt_indices),
                    loc,
                    type_info=expr.type_info,
                    shape_info=expr.shape_info,
                )
            )
        return _stamp_expr_metadata(
            row_expr,
            self._binding_map,
            fallback_type=expr.type_info,
            fallback_shape=expr.shape_info,
        )

    def _make_einstein_axes(
        self,
        shape: Tuple[ExpressionIR, ...],
        prefix: str,
        loc: Any,
    ) -> Tuple[List[IndexVarIR], Dict[DefId, RangeIR]]:
        axes: List[IndexVarIR] = []
        variable_ranges: Dict[DefId, RangeIR] = {}
        for axis, dim in enumerate(shape):
            defid = self._fresh_defid()
            idx = IndexVarIR(
                f"{prefix}{axis}",
                loc,
                defid,
                type_info=PrimitiveType("i32"),
            )
            rng = RangeIR(
                LiteralIR(0, loc, type_info=PrimitiveType("i32")),
                _clone_expr(dim),
                loc,
                type_info=PrimitiveType("range"),
            )
            idx.range_ir = rng
            axes.append(idx)
            variable_ranges[defid] = rng
        return axes, variable_ranges

    def _materialize_lazy_via_jvp(self, expr: LazyJacobianIR, target_binding: BindingIR, wrt_binding: BindingIR) -> ExpressionIR:
        loc = expr.location
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        wrt_axes, wrt_ranges = self._make_einstein_axes(wrt_shape, "_ad_wrt_", loc)
        out_axes, out_ranges = self._make_einstein_axes(target_shape, "_ad_out_", loc)
        tangent_basis = self._make_basis_tensor(wrt_binding, wrt_axes, loc)
        raw_body = self._lower_jvp(
            JvpIR(
                target=_clone_expr(expr.target),
                wrt=_clone_expr(expr.wrt),
                location=loc,
                tangent=tangent_basis,
                type_info=_tensor_type_info(target_binding, expr.type_info),
                shape_info=tuple(_clone_expr(dim) for dim in target_shape),
            ),
            tangent_expr=tangent_basis,
        )
        raw_body = self._inline_basis_tensor_accesses(raw_body, loc)
        temp_defid = self._fresh_defid()
        temp_name = self._fresh_name("__autodiff_jvp_body")
        temp_binding = BindingIR(
            temp_name,
            self._simplify_autodiff_tensor_ir(
                self._hoist_tensor_arrays_from_reductions(raw_body),
                loc,
                protected_tensor_defids={wrt_binding.defid},
            ),
            location=loc,
            defid=temp_defid,
            type_info=_tensor_type_info(target_binding, expr.type_info),
        )
        temp_ident = IdentifierIR(
            temp_name,
            loc,
            defid=temp_defid,
            type_info=_tensor_type_info(target_binding, expr.type_info),
            shape_info=tuple(_clone_expr(dim) for dim in target_shape),
        )
        variable_ranges: Dict[DefId, RangeIR] = {}
        variable_ranges.update(out_ranges)
        variable_ranges.update(wrt_ranges)
        final_shape = tuple(_clone_expr(dim) for dim in target_shape + wrt_shape)
        final_expr = EinsteinIR(
            clauses=[
                EinsteinClauseIR(
                    indices=list(out_axes) + list(wrt_axes),
                    value=BlockExpressionIR(
                        [temp_binding],
                        loc,
                        RectangularAccessIR(
                            temp_ident,
                            list(out_axes),
                            loc,
                            type_info=_tensor_element_type(target_binding),
                            shape_info=(),
                        ),
                        type_info=_tensor_element_type(target_binding),
                        shape_info=(),
                    ),
                    location=loc,
                    variable_ranges=variable_ranges,
                )
            ],
            shape=list(final_shape),
            element_type=_tensor_element_type(target_binding),
            location=loc,
            type_info=expr.type_info,
            shape_info=final_shape,
        )
        return _stamp_expr_metadata(
            final_expr,
            self._binding_map,
            fallback_type=expr.type_info,
            fallback_shape=final_shape,
        )

    def _materialize_lazy_via_vjp(self, expr: LazyJacobianIR, target_binding: BindingIR, wrt_binding: BindingIR) -> ExpressionIR:
        loc = expr.location
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        out_axes, out_ranges = self._make_einstein_axes(target_shape, "_ad_out_", loc)
        wrt_axes, wrt_ranges = self._make_einstein_axes(wrt_shape, "_ad_wrt_", loc)
        cotangent_basis = self._make_basis_tensor(target_binding, out_axes, loc)
        body = self._lower_vjp(
            VjpIR(
                target=_clone_expr(expr.target),
                wrt=_clone_expr(expr.wrt),
                location=loc,
                cotangent=cotangent_basis,
                type_info=_tensor_type_info(wrt_binding, expr.type_info),
                shape_info=tuple(_clone_expr(dim) for dim in wrt_shape),
            ),
            cotangent_expr=cotangent_basis,
        )
        body = self._inline_basis_tensor_accesses(body, loc)
        final_shape = tuple(_clone_expr(dim) for dim in target_shape + wrt_shape)
        temp_defid = self._fresh_defid()
        temp_name = self._fresh_name("__autodiff_vjp_body")
        temp_binding = BindingIR(
            temp_name,
            self._simplify_autodiff_tensor_ir(
                self._hoist_tensor_arrays_from_reductions(body),
                loc,
                protected_tensor_defids={wrt_binding.defid},
            ),
            location=loc,
            defid=temp_defid,
            type_info=_tensor_type_info(wrt_binding, expr.type_info),
        )
        temp_ident = IdentifierIR(
            temp_name,
            loc,
            defid=temp_defid,
            type_info=_tensor_type_info(wrt_binding, expr.type_info),
            shape_info=tuple(_clone_expr(dim) for dim in wrt_shape),
        )
        variable_ranges: Dict[DefId, RangeIR] = {}
        variable_ranges.update(out_ranges)
        variable_ranges.update(wrt_ranges)
        lowered = EinsteinIR(
            clauses=[
                EinsteinClauseIR(
                    indices=list(out_axes) + list(wrt_axes),
                    value=BlockExpressionIR(
                        [temp_binding],
                        loc,
                        RectangularAccessIR(
                            temp_ident,
                            list(wrt_axes),
                            loc,
                            type_info=_tensor_element_type(wrt_binding),
                            shape_info=(),
                        ),
                        type_info=_tensor_element_type(wrt_binding),
                        shape_info=(),
                    ),
                    location=loc,
                    variable_ranges=variable_ranges,
                )
            ],
            shape=[_clone_expr(dim) for dim in final_shape],
            element_type=_tensor_element_type(wrt_binding, expr.type_info),
            location=loc,
            type_info=expr.type_info,
            shape_info=final_shape,
        )
        return _stamp_expr_metadata(
            lowered,
            self._binding_map,
            fallback_type=expr.type_info,
            fallback_shape=final_shape,
        )

    def _lower_lazy(self, expr: LazyJacobianIR) -> ExpressionIR:
        target_binding = _binding_by_identifier(expr.target, self._binding_map)
        wrt_binding = _binding_by_identifier(expr.wrt, self._binding_map)
        if target_binding is None or wrt_binding is None:
            return expr
        if len(self._binding_shape(target_binding)) == 0:
            return self._lower_vjp(
                VjpIR(
                    target=_clone_expr(expr.target),
                    wrt=_clone_expr(expr.wrt),
                    location=expr.location,
                    type_info=_tensor_type_info(wrt_binding, expr.type_info),
                    shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(wrt_binding)),
                ),
                cotangent_expr=None,
            )
        if len(self._binding_shape(wrt_binding)) == 0:
            return self._lower_jvp(
                JvpIR(
                    target=_clone_expr(expr.target),
                    wrt=_clone_expr(expr.wrt),
                    location=expr.location,
                    type_info=_tensor_type_info(target_binding, expr.type_info),
                    shape_info=tuple(_clone_expr(dim) for dim in self._binding_shape(target_binding)),
                ),
                tangent_expr=None,
            )
        mode = self._choose_lazy_mode(target_binding, wrt_binding)
        if mode == "vjp":
            return self._materialize_lazy_via_vjp(expr, target_binding, wrt_binding)
        return self._materialize_lazy_via_jvp(expr, target_binding, wrt_binding)

    def rewrite_expr(self, expr: Optional[ExpressionIR]) -> Optional[ExpressionIR]:
        if expr is None:
            return None
        if isinstance(expr, LazyJacobianIR):
            return self._lower_lazy(expr)
        if isinstance(expr, JvpIR):
            tangent = self.rewrite_expr(expr.tangent) if expr.tangent is not None else None
            return self._lower_jvp(expr, tangent)
        if isinstance(expr, VjpIR):
            cotangent = self.rewrite_expr(expr.cotangent) if expr.cotangent is not None else None
            return self._lower_vjp(expr, cotangent)
        if isinstance(expr, BuiltinCallIR):
            expr.args = tuple(
                rewritten
                for arg in (expr.args or ())
                for rewritten in [self.rewrite_expr(arg)]
                if rewritten is not None
            )
            return expr
        if isinstance(expr, BinaryOpIR):
            expr.left = self.rewrite_expr(expr.left)
            expr.right = self.rewrite_expr(expr.right)
            return expr
        if isinstance(expr, UnaryOpIR):
            expr.operand = self.rewrite_expr(expr.operand)
            return expr
        if isinstance(expr, RectangularAccessIR):
            lazy_expr = self._original_lazy_expr(expr.array)
            if lazy_expr is not None:
                fused = self._rewrite_lazy_access(expr, lazy_expr)
                if fused is not None:
                    return fused
            expr.array = self.rewrite_expr(expr.array)
            expr.indices = tuple(self.rewrite_expr(idx) for idx in (expr.indices or ()))
            return expr
        if isinstance(expr, CastExpressionIR):
            expr.expr = self.rewrite_expr(expr.expr)
            return expr
        if isinstance(expr, MemberAccessIR):
            expr.object = self.rewrite_expr(expr.object)
            return expr
        if isinstance(expr, ArrayLiteralIR):
            expr.elements = tuple(self.rewrite_expr(elem) for elem in (expr.elements or ()))
            return expr
        if isinstance(expr, TupleExpressionIR):
            expr.elements = tuple(self.rewrite_expr(elem) for elem in (expr.elements or ()))
            return expr
        if isinstance(expr, TupleAccessIR):
            expr.tuple_expr = self.rewrite_expr(expr.tuple_expr)
            return expr
        if isinstance(expr, BlockExpressionIR):
            expr.statements = tuple(self.rewrite_statement_list(expr.statements or ()))
            expr.final_expr = self.rewrite_expr(expr.final_expr)
            return expr
        if isinstance(expr, IfExpressionIR):
            expr.condition = self.rewrite_expr(expr.condition)
            expr.then_expr = self.rewrite_expr(expr.then_expr)
            expr.else_expr = self.rewrite_expr(expr.else_expr)
            return expr
        if isinstance(expr, RangeIR):
            expr.start = self.rewrite_expr(expr.start)
            expr.end = self.rewrite_expr(expr.end)
            return expr
        if isinstance(expr, ReductionExpressionIR):
            expr.body = self.rewrite_expr(expr.body)
            if expr.where_clause is not None:
                expr.where_clause.constraints = tuple(
                    self.rewrite_expr(constraint) for constraint in (expr.where_clause.constraints or ())
                )
            return expr
        if isinstance(expr, EinsteinIR):
            new_clauses: List[EinsteinClauseIR] = []
            for clause in expr.clauses or ():
                clause.value = self.rewrite_expr(clause.value)
                if clause.where_clause is not None:
                    clause.where_clause.constraints = tuple(
                        self.rewrite_expr(constraint) for constraint in (clause.where_clause.constraints or ())
                    )
                new_clauses.append(clause)
            expr.clauses = tuple(new_clauses)
            expr.shape = tuple(self.rewrite_expr(dim) for dim in (expr.shape or ())) if expr.shape is not None else None
            return expr
        if isinstance(expr, FunctionCallIR):
            expr.callee_expr = self.rewrite_expr(expr.callee_expr)
            expr.arguments = tuple(self.rewrite_expr(arg) for arg in (expr.arguments or ()))
            return expr
        if isinstance(expr, FunctionValueIR):
            if expr.body is not None:
                expr.body = self.rewrite_expr(expr.body)
            return expr
        if isinstance(expr, ArrayComprehensionIR):
            expr.body = self.rewrite_expr(expr.body)
            expr.ranges = tuple(self.rewrite_expr(rng) for rng in (expr.ranges or ()))
            expr.constraints = tuple(self.rewrite_expr(c) for c in (expr.constraints or ()))
            return expr
        if isinstance(expr, WhereExpressionIR):
            expr.expr = self.rewrite_expr(expr.expr)
            expr.constraints = tuple(self.rewrite_expr(c) for c in (expr.constraints or ()))
            if expr.guard_constraints is not None:
                expr.guard_constraints = tuple(self.rewrite_expr(c) for c in (expr.guard_constraints or ()))
            if expr.binding_constraints is not None:
                expr.binding_constraints = tuple(
                    binding if not isinstance(binding, BindingIR) or binding.expr is None else BindingIR(
                        binding.name,
                        self.rewrite_expr(binding.expr),
                        location=binding.location,
                        defid=binding.defid,
                        type_info=binding.type_info,
                    )
                    for binding in expr.binding_constraints
                )
            return expr
        return expr


class AutodiffRequestLoweringPass(BasePass):
    requires = ["AutodiffPass"]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        analysis = tcx.get_analysis(AutodiffPass)
        binding_map = dict(analysis.get("graph_binding_by_defid") or {})
        binding_map.update(
            {
                did: binding
                for did, binding in (analysis.get("graph_function_ir_map") or {}).items()
                if did is not None and isinstance(binding, BindingIR)
            }
        )
        for did, binding in (getattr(tcx, "function_ir_map", None) or {}).items():
            if did is None or not isinstance(binding, BindingIR) or did in binding_map:
                continue
            binding_map[did] = binding
        if not binding_map:
            binding_map = _binding_map(ir, tcx)
        local_contexts = dict(analysis.get("graph_local_contexts_by_defid") or {})
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        lowerer_roots: List[Any] = [ir]
        lowerer_roots.extend(binding for binding in function_ir_map.values() if isinstance(binding, BindingIR))
        lowerer = _PlainRequestLowerer(
            binding_map,
            local_contexts,
            getattr(tcx, "resolver", None),
            roots=lowerer_roots,
        )
        ir.statements = lowerer.rewrite_statement_list(ir.statements or ())
        ir.bindings = [stmt for stmt in (ir.statements or []) if isinstance(stmt, BindingIR)]
        for binding in function_ir_map.values():
            if isinstance(binding, BindingIR) and isinstance(binding.expr, FunctionValueIR):
                binding.expr = lowerer.rewrite_expr(binding.expr)
        analysis["graph_builtin_requests_by_expr_id"] = collect_autodiff_builtin_requests(ir)
        tcx.set_analysis(AutodiffPass, analysis)
        return ir


__all__ = ["AutodiffRequestLoweringPass"]

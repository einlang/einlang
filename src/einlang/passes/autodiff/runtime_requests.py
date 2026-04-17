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
) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool, bytes)):
        return obj
    if isinstance(obj, BindingIR):
        if obj.expr is not None:
            _stamp_expr_metadata(obj.expr, binding_map)
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
            _stamp_expr_metadata(elem, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
            for elem in (obj.elements or ())
        )
        if obj.type_info is None and obj.elements:
            obj.type_info = getattr(obj.elements[0], "type_info", None)
        return obj
    if isinstance(obj, TupleExpressionIR):
        obj.elements = tuple(
            _stamp_expr_metadata(elem, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
            for elem in (obj.elements or ())
        )
        if obj.type_info is None:
            obj.type_info = fallback_type
        return obj
    if isinstance(obj, TupleAccessIR):
        obj.tuple_expr = _stamp_expr_metadata(obj.tuple_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.type_info is None:
            obj.type_info = fallback_type
        return obj
    if isinstance(obj, UnaryOpIR):
        obj.operand = _stamp_expr_metadata(obj.operand, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(obj.operand, "type_info", None), fallback_type)
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = _first_non_none(getattr(obj.operand, "shape_info", None), fallback_shape)
        return obj
    if isinstance(obj, BinaryOpIR):
        obj.left = _stamp_expr_metadata(obj.left, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.right = _stamp_expr_metadata(obj.right, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
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
        obj.array = _stamp_expr_metadata(obj.array, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.indices = tuple(
            _stamp_expr_metadata(idx, binding_map, fallback_type=PrimitiveType("i32"))
            for idx in (obj.indices or ())
        )
        array_type = getattr(obj.array, "type_info", None)
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(array_type, "element_type", None), fallback_type)
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = fallback_shape or ()
        return obj
    if isinstance(obj, CastExpressionIR):
        obj.expr = _stamp_expr_metadata(obj.expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(obj, "target_type", None), fallback_type)
        return obj
    if isinstance(obj, MemberAccessIR):
        obj.object = _stamp_expr_metadata(obj.object, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.type_info is None:
            obj.type_info = fallback_type
        return obj
    if isinstance(obj, BuiltinCallIR):
        obj.args = tuple(
            _stamp_expr_metadata(arg, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
            for arg in (obj.args or ())
        )
        if obj.type_info is None:
            obj.type_info = fallback_type
        return obj
    if isinstance(obj, FunctionCallIR):
        obj.callee_expr = _stamp_expr_metadata(obj.callee_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.arguments = tuple(
            _stamp_expr_metadata(arg, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
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
            _stamp_expr_metadata(stmt, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
            for stmt in (obj.statements or ())
        )
        obj.final_expr = _stamp_expr_metadata(obj.final_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(obj.final_expr, "type_info", None), fallback_type)
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = _first_non_none(getattr(obj.final_expr, "shape_info", None), fallback_shape)
        return obj
    if isinstance(obj, IfExpressionIR):
        obj.condition = _stamp_expr_metadata(obj.condition, binding_map, fallback_type=PrimitiveType("bool"))
        obj.then_expr = _stamp_expr_metadata(obj.then_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.else_expr = _stamp_expr_metadata(obj.else_expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
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
        obj.start = _stamp_expr_metadata(obj.start, binding_map, fallback_type=PrimitiveType("i32"))
        obj.end = _stamp_expr_metadata(obj.end, binding_map, fallback_type=PrimitiveType("i32"))
        if obj.type_info is None:
            obj.type_info = PrimitiveType("range")
        return obj
    if isinstance(obj, EinsteinIR):
        stamped_clauses: List[EinsteinClauseIR] = []
        for clause in obj.clauses or ():
            clause.value = _stamp_expr_metadata(clause.value, binding_map, fallback_type=fallback_type, fallback_shape=())
            if clause.where_clause is not None:
                clause.where_clause.constraints = tuple(
                    _stamp_expr_metadata(constraint, binding_map, fallback_type=PrimitiveType("bool"))
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
        obj.body = _stamp_expr_metadata(obj.body, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        if obj.where_clause is not None:
            obj.where_clause.constraints = tuple(
                _stamp_expr_metadata(constraint, binding_map, fallback_type=PrimitiveType("bool"))
                for constraint in (obj.where_clause.constraints or ())
            )
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(obj.body, "type_info", None), fallback_type)
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = fallback_shape
        return obj
    if isinstance(obj, ArrayComprehensionIR):
        obj.body = _stamp_expr_metadata(obj.body, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.ranges = tuple(_stamp_expr_metadata(rng, binding_map, fallback_type=PrimitiveType("range")) for rng in (obj.ranges or ()))
        obj.constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool")) for c in (obj.constraints or ()))
        if obj.type_info is None:
            obj.type_info = fallback_type
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = fallback_shape
        return obj
    if isinstance(obj, WhereExpressionIR):
        obj.expr = _stamp_expr_metadata(obj.expr, binding_map, fallback_type=fallback_type, fallback_shape=fallback_shape)
        obj.constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool")) for c in (obj.constraints or ()))
        if obj.guard_constraints is not None:
            obj.guard_constraints = tuple(_stamp_expr_metadata(c, binding_map, fallback_type=PrimitiveType("bool")) for c in (obj.guard_constraints or ()))
        if obj.binding_constraints is not None:
            obj.binding_constraints = tuple(_stamp_expr_metadata(binding, binding_map) for binding in obj.binding_constraints)
        if obj.type_info is None:
            obj.type_info = _first_non_none(getattr(obj.expr, "type_info", None), fallback_type)
        if getattr(obj, "shape_info", None) is None:
            obj.shape_info = _first_non_none(getattr(obj.expr, "shape_info", None), fallback_shape)
        return obj
    if isinstance(obj, FunctionValueIR):
        if obj.body is not None:
            obj.body = _stamp_expr_metadata(obj.body, binding_map)
        return obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            _stamp_expr_metadata(key, binding_map)
            _stamp_expr_metadata(value, binding_map)
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _stamp_expr_metadata(item, binding_map)
        return obj
    return obj


class _PlainRequestLowerer:
    def __init__(
        self,
        binding_map: Dict[Any, BindingIR],
        local_contexts: Dict[DefId, Dict[DefId, BindingIR]],
        resolver: Any,
    ) -> None:
        self._binding_map = binding_map
        self._local_contexts = local_contexts
        self._resolver = resolver
        self._counter = [0]
        binding_ctx = _AutodiffBindingContext(
            bindings=binding_map,
            dep_cache=_DependencyQueryCache(binding_map),
        )
        self._differentiator = _Differentiator(
            binding_ctx,
            resolver,
            local_contexts=local_contexts,
        )
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
    ) -> Tuple[List[BindingIR], Dict[DefId, IdentifierIR]]:
        scope_bindings = dict(self._binding_map)
        scope_bindings.update(self._local_contexts.get(target_defid) or {})
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
        contributions[target_defid].append(self._default_seed(target_binding, loc, 1.0))

        emitted: List[BindingIR] = []
        for did in target_first:
            binding = scope_bindings[did]
            bar_expr = self._sum_contributions(binding, contributions.get(did, []), loc)
            bar_expr = self._hoist_tensor_arrays_from_reductions(bar_expr)
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
                        return FunctionCallIR(
                            callee_expr=rewrite(cur.callee_expr),
                            location=cur.location,
                            arguments=tuple(rewrite(arg) for arg in (cur.arguments or ())),
                            module_path=cur.module_path,
                            type_info=cur.type_info,
                            shape_info=cur.shape_info,
                        )
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

        wrt_shape = self._binding_shape(wrt_binding)
        wrt_axes = self._make_comprehension_axes(wrt_shape, "_ad_wrt_", loc)
        wrt_idents = [ident for ident, _rng, _dim in wrt_axes]
        gradient_component = self._differentiate_with_seed(
            objective,
            wrt_binding,
            self._make_basis_tensor(wrt_binding, wrt_idents, loc),
            loc,
            local_bindings=local_bindings,
            fallback_type=_tensor_element_type(wrt_binding, node.type_info),
            fallback_shape=(),
            emit_intermediate_bindings=emit_intermediate_bindings,
        )
        gradient_component = _simplify(_substitute_identifiers(gradient_component, {cot_defid: cotangent_value}))
        lowered = self._nest_array_comprehensions(
            gradient_component,
            wrt_axes,
            loc,
            tuple(_clone_expr(dim) for dim in wrt_shape),
            node.type_info,
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
        selected_axes: List[IdentifierIR],
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
        wrt_axes = self._make_comprehension_axes(wrt_shape, "_ad_wrt_", loc)
        wrt_idents = [ident for ident, _rng, _dim in wrt_axes]
        tangent_basis = self._make_basis_tensor(wrt_binding, wrt_idents, loc)
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
        body = self._normalize_tensor_body(
            raw_body,
            target_shape,
            loc,
            type_info=_tensor_type_info(target_binding, expr.type_info),
        )
        temp_shape = tuple(_clone_expr(dim) for dim in wrt_shape + target_shape)
        materialized = self._nest_array_comprehensions(
            body,
            wrt_axes,
            loc,
            temp_shape,
            expr.type_info,
        )
        temp_defid = self._fresh_defid()
        temp_name = self._fresh_name("__autodiff_jacobian_cols")
        temp_binding = BindingIR(
            temp_name,
            materialized,
            location=loc,
            defid=temp_defid,
            type_info=expr.type_info,
        )
        temp_ident = IdentifierIR(
            temp_name,
            loc,
            defid=temp_defid,
            type_info=expr.type_info,
            shape_info=temp_shape,
        )
        out_axes, variable_ranges = self._make_einstein_axes(target_shape, "_ad_out_", loc)
        wrt_access_axes, wrt_ranges = self._make_einstein_axes(wrt_shape, "_ad_in_", loc)
        variable_ranges.update(wrt_ranges)
        final_shape = tuple(_clone_expr(dim) for dim in target_shape + wrt_shape)
        final_expr = EinsteinIR(
            clauses=[
                EinsteinClauseIR(
                    indices=list(out_axes) + list(wrt_access_axes),
                    value=RectangularAccessIR(
                        temp_ident,
                        list(wrt_access_axes) + list(out_axes),
                        loc,
                        type_info=_tensor_element_type(target_binding),
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
        lowered = BlockExpressionIR(
            [temp_binding],
            loc,
            final_expr,
            type_info=expr.type_info,
            shape_info=final_shape,
        )
        return _stamp_expr_metadata(
            lowered,
            self._binding_map,
            fallback_type=expr.type_info,
            fallback_shape=final_shape,
        )

    def _materialize_lazy_via_vjp(self, expr: LazyJacobianIR, target_binding: BindingIR, wrt_binding: BindingIR) -> ExpressionIR:
        loc = expr.location
        target_shape = self._binding_shape(target_binding)
        wrt_shape = self._binding_shape(wrt_binding)
        out_axes = self._make_comprehension_axes(target_shape, "_ad_out_", loc)
        out_idents = [ident for ident, _rng, _dim in out_axes]
        cotangent_basis = self._make_basis_tensor(target_binding, out_idents, loc)
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
        final_shape = tuple(_clone_expr(dim) for dim in target_shape + wrt_shape)
        lowered = self._nest_array_comprehensions(
            body,
            out_axes,
            loc,
            final_shape,
            expr.type_info,
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
        lowerer = _PlainRequestLowerer(binding_map, local_contexts, getattr(tcx, "resolver", None))
        ir.statements = lowerer.rewrite_statement_list(ir.statements or ())
        ir.bindings = [stmt for stmt in (ir.statements or []) if isinstance(stmt, BindingIR)]
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        for binding in function_ir_map.values():
            if isinstance(binding, BindingIR) and isinstance(binding.expr, FunctionValueIR):
                binding.expr = lowerer.rewrite_expr(binding.expr)
        analysis["graph_builtin_requests_by_expr_id"] = collect_autodiff_builtin_requests(ir)
        tcx.set_analysis(AutodiffPass, analysis)
        return ir


__all__ = ["AutodiffRequestLoweringPass"]

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ...ir.nodes import (
    ArrayLiteralIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    CastExpressionIR,
    DifferentialIR,
    EinsteinClauseIR,
    EinsteinIR,
    ExpressionIR,
    FunctionCallIR,
    FunctionValueIR,
    IRVisitor,
    IdentifierIR,
    IfExpressionIR,
    JaggedAccessIR,
    LambdaIR,
    LiteralIR,
    MemberAccessIR,
    PipelineExpressionIR,
    ProgramIR,
    RangeIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    TupleAccessIR,
    UnaryOpIR,
    is_function_binding,
)
from ...shared.defid import DefId
from ...shared.types import BinaryOp


def _is_diff_name(name: str) -> bool:
    return name.startswith("_@") or name.startswith("@")


class _DefIdCollector(IRVisitor[None]):
    """Collect all DefIds referenced in an expression tree."""

    def __init__(self) -> None:
        self.defids: Set[DefId] = set()

    def visit_identifier(self, n: IdentifierIR) -> None:
        if n.defid is not None:
            self.defids.add(n.defid)

    def visit_literal(self, n: LiteralIR) -> None:
        pass

    def visit_binary_op(self, n: BinaryOpIR) -> None:
        n.left.accept(self)
        n.right.accept(self)

    def visit_unary_op(self, n: UnaryOpIR) -> None:
        n.operand.accept(self)

    def visit_builtin_call(self, n: BuiltinCallIR) -> None:
        for a in n.args or []:
            a.accept(self)

    def visit_function_call(self, n: FunctionCallIR) -> None:
        if n.callee_expr is not None:
            n.callee_expr.accept(self)
        for a in n.arguments or []:
            a.accept(self)

    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        n.array.accept(self)
        for i in n.indices or []:
            i.accept(self)

    def visit_jagged_access(self, n: JaggedAccessIR) -> None:
        if n.base is not None:
            n.base.accept(self)

    def visit_block_expression(self, n: BlockExpressionIR) -> None:
        for s in n.statements or []:
            if isinstance(s, (BindingIR, ExpressionIR)):
                s.accept(self)
        if n.final_expr is not None:
            n.final_expr.accept(self)

    def visit_if_expression(self, n: IfExpressionIR) -> None:
        n.condition.accept(self)
        n.then_expr.accept(self)
        if n.else_expr is not None:
            n.else_expr.accept(self)

    def visit_cast_expression(self, n: CastExpressionIR) -> None:
        n.expr.accept(self)

    def visit_differential(self, n: DifferentialIR) -> None:
        n.operand.accept(self)

    def visit_lambda(self, n: LambdaIR) -> None:
        n.body.accept(self)

    def visit_range(self, n: RangeIR) -> None:
        n.start.accept(self)
        n.end.accept(self)

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> None:
        n.body.accept(self)

    def visit_where_expression(self, n: Any) -> None:
        n.expr.accept(self)
        for c in n.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, n: Any) -> None:
        n.left.accept(self)
        n.right.accept(self)

    def visit_array_comprehension(self, n: Any) -> None:
        n.body.accept(self)

    def visit_array_literal(self, n: ArrayLiteralIR) -> None:
        for e in n.elements or []:
            e.accept(self)

    def visit_tuple_expression(self, n: Any) -> None:
        for e in n.elements or []:
            e.accept(self)

    def visit_tuple_access(self, n: Any) -> None:
        n.tuple_expr.accept(self)

    def visit_member_access(self, n: MemberAccessIR) -> None:
        n.object.accept(self)

    def visit_function_value(self, n: FunctionValueIR) -> None:
        if n.body is not None:
            n.body.accept(self)

    def visit_try_expression(self, n: Any) -> None:
        n.operand.accept(self)

    def visit_match_expression(self, n: Any) -> None:
        n.scrutinee.accept(self)
        for arm in n.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_interpolated_string(self, n: Any) -> None:
        pass

    def visit_binding(self, n: BindingIR) -> None:
        if n.expr is not None:
            n.expr.accept(self)

    def visit_program(self, n: ProgramIR) -> None:
        for b in n.bindings or []:
            b.accept(self)

    def visit_einstein(self, n: EinsteinIR) -> None:
        for c in n.clauses or []:
            if isinstance(c, EinsteinClauseIR):
                c.accept(self)

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> None:
        if n.value is not None:
            n.value.accept(self)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> None:
        if n.primal_body is not None:
            n.primal_body.accept(self)
        if n.diff_body is not None:
            n.diff_body.accept(self)

    def visit_index_rest(self, n: Any) -> None:
        if getattr(n, "defid", None) is not None:
            self.defids.add(n.defid)

    def visit_module(self, n: Any) -> None:
        pass

    def visit_index_var(self, n: Any) -> None:
        pass

    def visit_identifier_pattern(self, n: Any) -> None:
        pass

    def visit_wildcard_pattern(self, n: Any) -> None:
        pass

    def visit_literal_pattern(self, n: Any) -> None:
        pass

    def visit_tuple_pattern(self, n: Any) -> None:
        pass

    def visit_array_pattern(self, n: Any) -> None:
        pass

    def visit_rest_pattern(self, n: Any) -> None:
        pass

    def visit_guard_pattern(self, n: Any) -> None:
        pass

    def visit_or_pattern(self, n: Any) -> None:
        pass

    def visit_constructor_pattern(self, n: Any) -> None:
        pass

    def visit_binding_pattern(self, n: Any) -> None:
        pass

    def visit_range_pattern(self, n: Any) -> None:
        pass


class _RectReadsRootDefIdVisitor(_DefIdCollector):
    """True after walk if any rectangular access uses ``tensor_did`` as array root."""

    def __init__(self, tensor_did: DefId) -> None:
        super().__init__()
        self._tensor_did = tensor_did
        self.found = False

    def visit_rectangular_access(self, n: RectangularAccessIR) -> None:
        a = n.array
        if isinstance(a, IdentifierIR) and a.defid == self._tensor_did:
            self.found = True
        if a is not None:
            a.accept(self)
        for i in n.indices or []:
            i.accept(self)


def _einstein_clause_values_rect_read_tensor(ein: EinsteinIR, tensor_did: DefId) -> bool:
    for c in ein.clauses or []:
        if c.value is None:
            continue
        vis = _RectReadsRootDefIdVisitor(tensor_did)
        c.value.accept(vis)
        if vis.found:
            return True
    return False


class _DependencyQueryCache:
    """Memoized dependency queries scoped to one binding map."""

    def __init__(self, bindings: Dict[DefId, BindingIR]) -> None:
        self._B = bindings
        self._expr_defids: Dict[int, frozenset[DefId]] = {}
        self._binding_expr_defids: Dict[DefId, frozenset[DefId]] = {}
        self._rhs_closure_by_expr: Dict[int, frozenset[DefId]] = {}
        self._reachable_by_src: Dict[DefId, frozenset[DefId]] = {}

    @property
    def bindings(self) -> Dict[DefId, BindingIR]:
        return self._B

    def collect_defids(self, expr: Optional[ExpressionIR]) -> frozenset[DefId]:
        if expr is None:
            return frozenset()
        key = id(expr)
        cached = self._expr_defids.get(key)
        if cached is not None:
            return cached
        collector = _DefIdCollector()
        expr.accept(collector)
        out = frozenset(collector.defids)
        self._expr_defids[key] = out
        return out

    def binding_expr_defids(self, did: DefId) -> frozenset[DefId]:
        cached = self._binding_expr_defids.get(did)
        if cached is not None:
            return cached
        binding = self._B.get(did)
        out = self.collect_defids(binding.expr) if binding is not None else frozenset()
        self._binding_expr_defids[did] = out
        return out

    def jacobian_rhs_closure(self, expr: Optional[ExpressionIR]) -> frozenset[DefId]:
        if expr is None:
            return frozenset()
        key = id(expr)
        cached = self._rhs_closure_by_expr.get(key)
        if cached is not None:
            return cached
        work = list(self.collect_defids(expr))
        closure: Set[DefId] = set()
        while work:
            did = work.pop()
            if did in closure:
                continue
            closure.add(did)
            binding = self._B.get(did)
            if binding is None or binding.expr is None or _is_diff_name(binding.name or ""):
                continue
            for dep in self.binding_expr_defids(did):
                if dep not in closure:
                    work.append(dep)
        out = frozenset(closure)
        self._rhs_closure_by_expr[key] = out
        return out

    def reachable_from(self, src: DefId) -> frozenset[DefId]:
        cached = self._reachable_by_src.get(src)
        if cached is not None:
            return cached
        seen: Set[DefId] = set()
        queue = [src]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for dep in self.binding_expr_defids(cur):
                if dep not in seen:
                    queue.append(dep)
        out = frozenset(seen)
        self._reachable_by_src[src] = out
        return out

    def fork(self, bindings: Dict[DefId, BindingIR]) -> "_DependencyQueryCache":
        if bindings is self._B:
            return self
        child = _DependencyQueryCache(bindings)
        child._expr_defids = self._expr_defids
        return child


def _collect_defids(expr: Optional[ExpressionIR]) -> Set[DefId]:
    if expr is None:
        return set()
    c = _DefIdCollector()
    expr.accept(c)
    return c.defids


def _function_call_ir_label(n: FunctionCallIR) -> str:
    name = n.function_name or "<non-identifier callee>"
    mp = n.module_path
    if mp:
        return "::".join(mp) + "::" + name
    return name


def _autodiff_primal_data_defids(
    expr: Optional[ExpressionIR],
    bindings: Dict[DefId, Any],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> Set[DefId]:
    if expr is None:
        return set()
    out: Set[DefId] = set()
    defs = dep_cache.collect_defids(expr) if dep_cache is not None else _collect_defids(expr)
    for d in defs:
        if d is None:
            continue
        bb = bindings.get(d)
        if bb is not None and is_function_binding(bb):
            continue
        out.add(d)
    return out


def _rectangular_read_root_defid(expr: Optional[ExpressionIR]) -> Optional[DefId]:
    if expr is None:
        return None
    cur: ExpressionIR = expr
    while isinstance(cur, RectangularAccessIR):
        arr = cur.array
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return arr.defid
        cur = arr
    return None


def _binding_is_rect_slice_of_tensor(wrt: DefId, tensor_did: DefId, bindings: Dict[DefId, BindingIR]) -> bool:
    wb = bindings.get(wrt)
    if wb is None or wb.expr is None:
        return False
    return _rectangular_read_root_defid(wb.expr) == tensor_did


def _jacobian_rhs_depends_on_wrt(
    expr: Optional[ExpressionIR],
    wrt: DefId,
    bindings: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache] = None,
) -> bool:
    if expr is None:
        return False
    if dep_cache is not None:
        closure = dep_cache.jacobian_rhs_closure(expr)
    else:
        work = [d for d in _collect_defids(expr) if d is not None]
        closure: Set[DefId] = set()
        while work:
            d = work.pop()
            if d in closure:
                continue
            closure.add(d)
            b = bindings.get(d)
            if b is None or b.expr is None or _is_diff_name(b.name or ""):
                continue
            for e in _collect_defids(b.expr):
                if e is not None and e not in closure:
                    work.append(e)
    if wrt in closure:
        return True
    wb = bindings.get(wrt)
    if wb is not None and wb.expr is not None:
        root = _rectangular_read_root_defid(wb.expr)
        if root is not None and root in closure:
            return True
    return False


class _TargetCollector(_DefIdCollector):
    """Single walk: differential targets + quotient pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: List[Tuple[DefId, str]] = []
        self.quotient_pairs: List[Tuple[DefId, DefId]] = []

    @staticmethod
    def _tgt_op(op: Any) -> Optional[Tuple[DefId, str]]:
        if isinstance(op, IdentifierIR) and op.defid is not None:
            return (op.defid, op.name or "")
        return None

    @staticmethod
    def _diff_defid(e: Any) -> Optional[DefId]:
        if isinstance(e, DifferentialIR) and isinstance(e.operand, IdentifierIR):
            return e.operand.defid
        if isinstance(e, IdentifierIR):
            return e.defid
        return None

    def visit_differential(self, n: DifferentialIR) -> None:
        t = self._tgt_op(n.operand)
        if t is not None:
            self.targets.append(t)
        n.operand.accept(self)

    def visit_binary_op(self, n: BinaryOpIR) -> None:
        if (
            n.operator == BinaryOp.DIV
            and isinstance(n.left, DifferentialIR)
            and isinstance(n.right, DifferentialIR)
        ):
            for op in (n.left.operand, n.right.operand):
                tt = self._tgt_op(op)
                if tt is not None:
                    self.targets.append(tt)
            num, den = self._diff_defid(n.left), self._diff_defid(n.right)
            if num is not None and den is not None:
                self.quotient_pairs.append((num, den))
        n.left.accept(self)
        n.right.accept(self)

    def visit_program(self, n: ProgramIR) -> None:
        for b in n.bindings or []:
            b.accept(self)
        for s in n.statements or []:
            if not isinstance(s, BindingIR) and isinstance(s, ExpressionIR):
                s.accept(self)


def _collect_targets(node: Any) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    c = _TargetCollector()
    node.accept(c)
    return c.targets, c.quotient_pairs


def _collect_targets_expr(expr: Any) -> Tuple[List[Tuple[DefId, str]], List[Tuple[DefId, DefId]]]:
    return _collect_targets(expr)


def _is_reachable(src: DefId, tgt: DefId, bindings: Dict[DefId, BindingIR]) -> bool:
    return _is_reachable_with_cache(src, tgt, bindings, None)


def _is_reachable_with_cache(
    src: DefId,
    tgt: DefId,
    bindings: Dict[DefId, BindingIR],
    dep_cache: Optional[_DependencyQueryCache],
) -> bool:
    if dep_cache is not None:
        return tgt in dep_cache.reachable_from(src)
    vis: Set[DefId] = set()
    q = [src]
    while q:
        cur = q.pop()
        if cur == tgt:
            return True
        if cur in vis:
            continue
        vis.add(cur)
        b = bindings.get(cur)
        if b is not None and b.expr is not None:
            for d in _collect_defids(b.expr):
                if d not in vis:
                    q.append(d)
    return False


__all__ = [
    "_DefIdCollector",
    "_DependencyQueryCache",
    "_TargetCollector",
    "_autodiff_primal_data_defids",
    "_binding_is_rect_slice_of_tensor",
    "_collect_defids",
    "_collect_targets",
    "_collect_targets_expr",
    "_einstein_clause_values_rect_read_tensor",
    "_function_call_ir_label",
    "_is_reachable",
    "_is_reachable_with_cache",
    "_jacobian_rhs_depends_on_wrt",
    "_rectangular_read_root_defid",
]

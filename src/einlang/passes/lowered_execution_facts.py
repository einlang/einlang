"""Compiler-owned lowered execution facts and reduction kernel plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .base import BasePass, TyCtxt
from ..ir.nodes import (
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    ExpressionIR,
    FunctionCallIR,
    IRNode,
    IfExpressionIR,
    IndexVarIR,
    IdentifierIR,
    LiteralIR,
    LoweredEinsteinClauseIR,
    LoweredEinsteinIR,
    LoweredRecurrenceIR,
    LoweredReductionIR,
    LoweredSelectAtArgmaxIR,
    ProgramIR,
    RectangularAccessIR,
)
from ..shared.types import BinaryOp, ReductionOp


@dataclass(frozen=True)
class LoweredReductionExecutionFacts:
    facts_id: int
    contains_nested_reduction_or_select: bool
    contains_if_expression: bool
    contains_lowered_einstein: bool
    body_defids_by_name: Dict[str, Tuple[Any, ...]]
    guard_defids_by_name: Dict[str, Tuple[Any, ...]]


@dataclass(frozen=True)
class ReductionKernelPlan:
    plan_id: int
    kind: str
    left: RectangularAccessIR
    right: RectangularAccessIR
    bias: Optional[ExpressionIR]
    scale: Optional[float]


@dataclass(frozen=True)
class LoweredEinsteinClauseExecutionFacts:
    facts_id: int
    has_literal_index: bool
    loop_defids: Tuple[Any, ...]
    loop_defids_nonnull: Tuple[Any, ...]
    loop_names_by_defid: Dict[Any, str]
    call_arg_loop_defids: Tuple[Any, ...]
    static_loop_ranges: Tuple[Optional[Tuple[int, int]], ...]
    body_defids_by_name: Dict[str, Tuple[Any, ...]]
    body_reduction_dim_count: int
    body_reduction_uses_clause_var_in_bounds: bool
    body_contains_if_expression: bool
    body_contains_nested_reduction_or_select: bool
    body_contains_call_using_loop_var: bool
    body_is_elementwise_call: bool
    body_has_direct_nested_lowered_binding: bool


@dataclass(frozen=True)
class LoweredSelectAtArgmaxExecutionFacts:
    facts_id: int
    depids: Tuple[Any, ...]
    primal_depids: Tuple[Any, ...]
    loop_names_by_defid: Dict[Any, str]
    body_defids_by_name: Dict[str, Tuple[Any, ...]]


@dataclass(frozen=True)
class LoweredEinsteinExecutionFacts:
    facts_id: int
    contains_select_at_argmax: bool
    depids: Tuple[Any, ...]


@dataclass
class _NodeSummary:
    all_defids: set[Any] = field(default_factory=set)
    defids_by_name: Dict[str, set[Any]] = field(default_factory=dict)
    call_arg_defids: set[Any] = field(default_factory=set)
    reduction_bound_defids: set[Any] = field(default_factory=set)
    max_reduction_dims: int = 0
    contains_if_expression: bool = False
    contains_lowered_einstein: bool = False
    contains_select_at_argmax: bool = False
    contains_nested_reduction_or_select: bool = False
    contains_nested_lowered_ir: bool = False


def _merge_summary(dst: _NodeSummary, src: _NodeSummary) -> _NodeSummary:
    if not src:
        return dst
    dst.all_defids.update(src.all_defids)
    for name, dids in src.defids_by_name.items():
        dst.defids_by_name.setdefault(name, set()).update(dids)
    dst.call_arg_defids.update(src.call_arg_defids)
    dst.reduction_bound_defids.update(src.reduction_bound_defids)
    dst.max_reduction_dims = max(dst.max_reduction_dims, src.max_reduction_dims)
    dst.contains_if_expression = dst.contains_if_expression or src.contains_if_expression
    dst.contains_lowered_einstein = dst.contains_lowered_einstein or src.contains_lowered_einstein
    dst.contains_select_at_argmax = dst.contains_select_at_argmax or src.contains_select_at_argmax
    dst.contains_nested_reduction_or_select = (
        dst.contains_nested_reduction_or_select or src.contains_nested_reduction_or_select
    )
    dst.contains_nested_lowered_ir = dst.contains_nested_lowered_ir or src.contains_nested_lowered_ir
    return dst


class _SummaryCollector:
    def __init__(self) -> None:
        self._memo: Dict[int, _NodeSummary] = {}
        self._lowered_reductions: Dict[int, LoweredReductionIR] = {}
        self._lowered_clauses: Dict[int, LoweredEinsteinClauseIR] = {}
        self._lowered_selects: Dict[int, LoweredSelectAtArgmaxIR] = {}
        self._lowered_einsteins: Dict[int, LoweredEinsteinIR] = {}

    def collect(self, node: Any) -> _NodeSummary:
        if node is None:
            return _NodeSummary()
        if isinstance(node, (list, tuple, dict, IRNode)):
            cached = self._memo.get(id(node))
            if cached is not None:
                return cached

        if isinstance(node, list):
            out = _NodeSummary()
            self._memo[id(node)] = out
            for item in node:
                _merge_summary(out, self.collect(item))
            return out
        if isinstance(node, tuple):
            out = _NodeSummary()
            self._memo[id(node)] = out
            for item in node:
                _merge_summary(out, self.collect(item))
            return out
        if isinstance(node, dict):
            out = _NodeSummary()
            self._memo[id(node)] = out
            for key, value in node.items():
                _merge_summary(out, self.collect(key))
                _merge_summary(out, self.collect(value))
            return out
        if not isinstance(node, IRNode):
            return _NodeSummary()

        out = _NodeSummary()
        self._memo[id(node)] = out

        if isinstance(node, LoweredReductionIR):
            self._lowered_reductions[id(node)] = node
        if isinstance(node, LoweredEinsteinClauseIR):
            self._lowered_clauses[id(node)] = node
        if isinstance(node, LoweredSelectAtArgmaxIR):
            self._lowered_selects[id(node)] = node
        if isinstance(node, LoweredEinsteinIR):
            self._lowered_einsteins[id(node)] = node

        if isinstance(node, (IdentifierIR, IndexVarIR)):
            defid = getattr(node, "defid", None)
            name = getattr(node, "name", None)
            if defid is not None:
                out.all_defids.add(defid)
                if name is not None:
                    out.defids_by_name.setdefault(name, set()).add(defid)
            range_ir = getattr(node, "range_ir", None)
            if range_ir is not None:
                _merge_summary(out, self.collect(range_ir))
            return out

        if isinstance(node, FunctionCallIR):
            callee_summary = self.collect(node.callee_expr)
            _merge_summary(out, callee_summary)
            out.call_arg_defids.update(callee_summary.all_defids)
            for arg in node.arguments or ():
                arg_summary = self.collect(arg)
                _merge_summary(out, arg_summary)
                out.call_arg_defids.update(arg_summary.all_defids)
            return out

        if isinstance(node, LoweredReductionIR):
            out.contains_nested_reduction_or_select = True
            out.contains_nested_lowered_ir = True
            out.max_reduction_dims = max(out.max_reduction_dims, len(node.loops or ()))
            for loop in node.loops or ():
                _merge_summary(out, self.collect(loop.variable))
                iterable_summary = self.collect(loop.iterable)
                _merge_summary(out, iterable_summary)
                out.reduction_bound_defids.update(iterable_summary.all_defids)
            _merge_summary(out, self.collect(node.body))
            _merge_summary(out, self.collect(node.bindings))
            _merge_summary(out, self.collect(node.guards))
            return out

        if isinstance(node, LoweredSelectAtArgmaxIR):
            out.contains_select_at_argmax = True
            out.contains_nested_reduction_or_select = True
            out.contains_nested_lowered_ir = True
            for loop in node.loops or ():
                _merge_summary(out, self.collect(loop.variable))
                _merge_summary(out, self.collect(loop.iterable))
            _merge_summary(out, self.collect(node.primal_body))
            _merge_summary(out, self.collect(node.diff_body))
            _merge_summary(out, self.collect(node.bindings))
            _merge_summary(out, self.collect(node.guards))
            return out

        if isinstance(node, LoweredEinsteinIR):
            out.contains_lowered_einstein = True
            out.contains_nested_lowered_ir = True
            _merge_summary(out, self.collect(node.items))
            _merge_summary(out, self.collect(node.shape))
            return out

        if isinstance(node, LoweredRecurrenceIR):
            out.contains_nested_lowered_ir = True
            _merge_summary(out, self.collect(node.initial))
            _merge_summary(out, self.collect(node.recurrence_loop))
            _merge_summary(out, self.collect(node.body))
            return out

        if isinstance(node, IfExpressionIR):
            out.contains_if_expression = True

        for slot in _iter_slots(type(node)):
            _merge_summary(out, self.collect(getattr(node, slot, None)))
        return out

    def lowered_reductions(self) -> List[LoweredReductionIR]:
        return list(self._lowered_reductions.values())

    def lowered_clauses(self) -> List[LoweredEinsteinClauseIR]:
        return list(self._lowered_clauses.values())

    def lowered_selects(self) -> List[LoweredSelectAtArgmaxIR]:
        return list(self._lowered_selects.values())

    def lowered_einsteins(self) -> List[LoweredEinsteinIR]:
        return list(self._lowered_einsteins.values())


def _summary_root_is_terminal_call(node: Any) -> bool:
    if isinstance(node, FunctionCallIR):
        return True
    if isinstance(node, BlockExpressionIR):
        return _summary_root_is_terminal_call(node.final_expr)
    return False


def _freeze_summary_defids_by_name(mapping: Dict[str, set[Any]]) -> Dict[str, Tuple[Any, ...]]:
    return {
        key: tuple(sorted(values, key=lambda d: (getattr(d, "krate", -1), getattr(d, "index", -1))))
        for key, values in mapping.items()
    }


def build_lowered_execution_facts_analysis(ir: ProgramIR, tcx: Any) -> Dict[str, Any]:
    analyzer = _LoweredExecutionFactsAnalyzer()
    analyzer.walk(ir)
    function_ir_map = getattr(tcx, "function_ir_map", None) or {}
    for binding in function_ir_map.values():
        analyzer.walk(binding)
    analyzer.finalize()
    return {
        "reduction_facts_by_id": analyzer.reduction_facts_by_id,
        "reduction_kernel_plans_by_id": analyzer.reduction_kernel_plans_by_id,
        "clause_facts_by_id": analyzer.clause_facts_by_id,
        "select_facts_by_id": analyzer.select_facts_by_id,
        "einstein_facts_by_id": analyzer.einstein_facts_by_id,
    }


class LoweredExecutionFactsPass(BasePass):
    """Attach compiler-owned lowered execution facts for backend reduction execution."""

    requires = ["RecurrenceOrderPass"]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        tcx.set_analysis(
            LoweredExecutionFactsPass,
            build_lowered_execution_facts_analysis(ir, tcx),
        )
        return ir


class _LoweredExecutionFactsAnalyzer:
    def __init__(self) -> None:
        self._next_id = 1
        self.reduction_facts_by_id: Dict[int, LoweredReductionExecutionFacts] = {}
        self.reduction_kernel_plans_by_id: Dict[int, ReductionKernelPlan] = {}
        self.clause_facts_by_id: Dict[int, LoweredEinsteinClauseExecutionFacts] = {}
        self.select_facts_by_id: Dict[int, LoweredSelectAtArgmaxExecutionFacts] = {}
        self.einstein_facts_by_id: Dict[int, LoweredEinsteinExecutionFacts] = {}
        self._summary = _SummaryCollector()

    def _new_id(self) -> int:
        current = self._next_id
        self._next_id += 1
        return current

    def walk(self, node: Any) -> Any:
        self._summary.collect(node)
        return None

    def finalize(self) -> None:
        for node in self._summary.lowered_reductions():
            self._annotate_lowered_reduction(node)
        for node in self._summary.lowered_clauses():
            self._annotate_lowered_einstein_clause(node)
        for node in self._summary.lowered_selects():
            self._annotate_lowered_select_at_argmax(node)
        for node in self._summary.lowered_einsteins():
            self._annotate_lowered_einstein(node)

    def _annotate_lowered_reduction(self, node: LoweredReductionIR) -> None:
        facts_id = node.execution_facts_id
        if facts_id is not None and facts_id in self.reduction_facts_by_id:
            return
        if facts_id is None:
            facts_id = self._new_id()
            node.execution_facts_id = facts_id

        body_summary = self._summary.collect(node.body)
        guard_summary = self._summary.collect([g.condition for g in (node.guards or ())])
        facts = LoweredReductionExecutionFacts(
            facts_id=facts_id,
            contains_nested_reduction_or_select=body_summary.contains_nested_reduction_or_select,
            contains_if_expression=body_summary.contains_if_expression,
            contains_lowered_einstein=body_summary.contains_lowered_einstein,
            body_defids_by_name=_freeze_summary_defids_by_name(body_summary.defids_by_name),
            guard_defids_by_name=_freeze_summary_defids_by_name(guard_summary.defids_by_name),
        )
        self.reduction_facts_by_id[facts_id] = facts

        plan = _recognize_reduction_kernel(node)
        if plan is not None:
            if node.kernel_plan_id is None:
                node.kernel_plan_id = plan.plan_id
            self.reduction_kernel_plans_by_id[plan.plan_id] = plan

    def _annotate_lowered_einstein_clause(self, node: LoweredEinsteinClauseIR) -> None:
        facts_id = node.execution_facts_id
        if facts_id is not None and facts_id in self.clause_facts_by_id:
            return
        if facts_id is None:
            facts_id = self._new_id()
            node.execution_facts_id = facts_id

        loop_defids = tuple(getattr(getattr(loop, "variable", None), "defid", None) for loop in (node.loops or ()))
        loop_defids_nonnull = tuple(defid for defid in loop_defids if defid is not None)
        loop_names_by_defid = {
            loop.variable.defid: loop.variable.name
            for loop in (node.loops or ())
            if getattr(getattr(loop, "variable", None), "defid", None) is not None
        }
        body = node.body
        body_summary = self._summary.collect(body)
        loop_defid_set = set(loop_defids_nonnull)
        call_arg_loop_defids = tuple(
            sorted(
                (did for did in body_summary.call_arg_defids if did in loop_defid_set),
                key=lambda did: (getattr(did, "krate", -1), getattr(did, "index", -1)),
            )
        )
        static_loop_ranges = tuple(_static_loop_range(loop) for loop in (node.loops or ()))
        facts = LoweredEinsteinClauseExecutionFacts(
            facts_id=facts_id,
            has_literal_index=any(isinstance(idx, LiteralIR) for idx in (node.indices or ())),
            loop_defids=loop_defids,
            loop_defids_nonnull=loop_defids_nonnull,
            loop_names_by_defid=loop_names_by_defid,
            call_arg_loop_defids=call_arg_loop_defids,
            static_loop_ranges=static_loop_ranges,
            body_defids_by_name=_freeze_summary_defids_by_name(body_summary.defids_by_name),
            body_reduction_dim_count=body_summary.max_reduction_dims,
            body_reduction_uses_clause_var_in_bounds=bool(body_summary.reduction_bound_defids & loop_defid_set),
            body_contains_if_expression=body_summary.contains_if_expression,
            body_contains_nested_reduction_or_select=body_summary.contains_nested_reduction_or_select,
            body_contains_call_using_loop_var=bool(call_arg_loop_defids),
            body_is_elementwise_call=(
                bool(loop_defid_set)
                and set(call_arg_loop_defids) == loop_defid_set
                and _summary_root_is_terminal_call(body)
            ),
            body_has_direct_nested_lowered_binding=(
                isinstance(body, BlockExpressionIR)
                and body_summary.contains_nested_lowered_ir
            ),
        )
        self.clause_facts_by_id[facts_id] = facts

    def _annotate_lowered_select_at_argmax(self, node: LoweredSelectAtArgmaxIR) -> None:
        facts_id = node.execution_facts_id
        if facts_id is not None and facts_id in self.select_facts_by_id:
            return
        if facts_id is None:
            facts_id = self._new_id()
            node.execution_facts_id = facts_id

        loop_names_by_defid = {
            loop.variable.defid: loop.variable.name
            for loop in (node.loops or ())
            if getattr(getattr(loop, "variable", None), "defid", None) is not None
        }
        primal_summary = self._summary.collect(node.primal_body)
        diff_summary = self._summary.collect(node.diff_body)
        iterable_summary = self._summary.collect([getattr(loop, "iterable", None) for loop in (node.loops or ())])
        body_defids_by_name: Dict[str, set[Any]] = {}
        _merge_summary_maps(body_defids_by_name, primal_summary.defids_by_name)
        _merge_summary_maps(body_defids_by_name, diff_summary.defids_by_name)
        depids = tuple(
            sorted(
                (primal_summary.all_defids | diff_summary.all_defids | iterable_summary.all_defids),
                key=lambda d: (d.krate, d.index),
            )
        )
        primal_depids = tuple(
            sorted(
                (primal_summary.all_defids | iterable_summary.all_defids),
                key=lambda d: (d.krate, d.index),
            )
        )
        self.select_facts_by_id[facts_id] = LoweredSelectAtArgmaxExecutionFacts(
            facts_id=facts_id,
            depids=depids,
            primal_depids=primal_depids,
            loop_names_by_defid=loop_names_by_defid,
            body_defids_by_name=_freeze_summary_defids_by_name(body_defids_by_name),
        )

    def _annotate_lowered_einstein(self, node: LoweredEinsteinIR) -> None:
        facts_id = getattr(node, "execution_facts_id", None)
        if facts_id is not None and facts_id in self.einstein_facts_by_id:
            return
        if facts_id is None:
            facts_id = self._new_id()
            node.execution_facts_id = facts_id
        summary = self._summary.collect(node)
        self.einstein_facts_by_id[facts_id] = LoweredEinsteinExecutionFacts(
            facts_id=facts_id,
            contains_select_at_argmax=summary.contains_select_at_argmax,
            depids=tuple(
                sorted(
                    (did for did in summary.all_defids if did is not None),
                    key=lambda d: (d.krate, d.index),
                )
            ),
        )


def _contains_node_type(node: Any, *node_types: type) -> bool:
    if node is None:
        return False
    if isinstance(node, node_types):
        return True
    if isinstance(node, list):
        return any(_contains_node_type(item, *node_types) for item in node)
    if isinstance(node, tuple):
        return any(_contains_node_type(item, *node_types) for item in node)
    if isinstance(node, dict):
        return any(
            _contains_node_type(key, *node_types) or _contains_node_type(value, *node_types)
            for key, value in node.items()
        )
    if isinstance(node, IRNode):
        for slot in _iter_slots(type(node)):
            if _contains_node_type(getattr(node, slot, None), *node_types):
                return True
    return False


def _merge_defids_by_name(a: Dict[str, List[Any]], b: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    for key, values in b.items():
        bucket = a.setdefault(key, [])
        for value in values:
            if value not in bucket:
                bucket.append(value)
    return a


def _merge_summary_maps(a: Dict[str, set[Any]], b: Dict[str, set[Any]]) -> Dict[str, set[Any]]:
    for key, values in b.items():
        a.setdefault(key, set()).update(values)
    return a


class _DefidsByNameCollector:
    def __init__(self) -> None:
        self._memo: Dict[int, Dict[str, List[Any]]] = {}

    def collect(self, node: Any) -> Dict[str, List[Any]]:
        if node is None:
            return {}
        if isinstance(node, (list, tuple, IRNode)):
            oid = id(node)
            cached = self._memo.get(oid)
            if cached is not None:
                return cached
        if isinstance(node, list):
            out: Dict[str, List[Any]] = {}
            self._memo[id(node)] = out
            for item in node:
                _merge_defids_by_name(out, self.collect(item))
            return out
        if isinstance(node, tuple):
            out: Dict[str, List[Any]] = {}
            self._memo[id(node)] = out
            for item in node:
                _merge_defids_by_name(out, self.collect(item))
            return out
        if isinstance(node, (IdentifierIR, IndexVarIR)):
            if node.name is not None and node.defid is not None:
                return {node.name: [node.defid]}
            return {}
        if isinstance(node, RectangularAccessIR):
            out = self.collect(node.array)
            for idx in node.indices or []:
                _merge_defids_by_name(out, self.collect(idx))
            return out
        if isinstance(node, BinaryOpIR):
            out = self.collect(node.left)
            _merge_defids_by_name(out, self.collect(node.right))
            return out
        if isinstance(node, FunctionCallIR):
            out = self.collect(node.callee_expr)
            for arg in node.arguments or []:
                _merge_defids_by_name(out, self.collect(arg))
            return out
        if isinstance(node, BindingIR):
            return self.collect(node.expr)
        if isinstance(node, IfExpressionIR):
            out = self.collect(node.condition)
            _merge_defids_by_name(out, self.collect(node.then_expr))
            _merge_defids_by_name(out, self.collect(node.else_expr))
            return out
        if isinstance(node, IRNode):
            out: Dict[str, List[Any]] = {}
            self._memo[id(node)] = out
            for slot in _iter_slots(type(node)):
                _merge_defids_by_name(out, self.collect(getattr(node, slot, None)))
            return out
        return {}

@lru_cache(maxsize=None)
def _iter_slots(cls: type) -> Tuple[str, ...]:
    out: List[str] = []
    for c in cls.__mro__:
        slots = getattr(c, "__slots__", ())
        if isinstance(slots, str):
            out.append(slots)
        else:
            out.extend(slots)
    seen: set[str] = set()
    ordered: List[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _freeze_defids_by_name(mapping: Dict[str, List[Any]]) -> Dict[str, Tuple[Any, ...]]:
    return {key: tuple(values) for key, values in mapping.items()}


def _try_static_int(expr: Any) -> Optional[int]:
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, LiteralIR):
        try:
            return int(expr.value)
        except (TypeError, ValueError):
            return None
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


def _static_loop_range(loop: Any) -> Optional[Tuple[int, int]]:
    iterable = getattr(loop, "iterable", None)
    if iterable is None:
        return None
    if isinstance(iterable, LiteralIR) and isinstance(iterable.value, range):
        try:
            return (int(iterable.value.start), int(iterable.value.stop))
        except (TypeError, ValueError):
            return None
    start_node = getattr(iterable, "start", None)
    end_node = getattr(iterable, "end", None)
    if start_node is None and end_node is None:
        return None
    end_value = _try_static_int(end_node)
    if end_value is None:
        return None
    start_value = 0 if start_node is None else _try_static_int(start_node)
    if start_value is None:
        return None
    return (int(start_value), int(end_value))

def _expr_contains_defid(expr: Any, target_defid: Any) -> bool:
    if expr is None or target_defid is None:
        return False
    if isinstance(expr, (IdentifierIR, IndexVarIR)):
        return expr.defid == target_defid
    if isinstance(expr, BinaryOpIR):
        return _expr_contains_defid(expr.left, target_defid) or _expr_contains_defid(expr.right, target_defid)
    if isinstance(expr, RectangularAccessIR):
        return _expr_contains_defid(expr.array, target_defid) or any(
            _expr_contains_defid(idx, target_defid) for idx in (expr.indices or [])
        )
    if isinstance(expr, FunctionCallIR):
        return _expr_contains_defid(expr.callee_expr, target_defid) or any(
            _expr_contains_defid(arg, target_defid) for arg in (expr.arguments or [])
        )
    if isinstance(expr, IfExpressionIR):
        return (
            _expr_contains_defid(expr.condition, target_defid)
            or _expr_contains_defid(expr.then_expr, target_defid)
            or _expr_contains_defid(expr.else_expr, target_defid)
        )
    if isinstance(expr, BindingIR):
        return _expr_contains_defid(expr.expr, target_defid)
    if isinstance(expr, IRNode):
        for cls in type(expr).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if _expr_contains_defid(getattr(expr, slot, None), target_defid):
                    return True
    if isinstance(expr, (list, tuple)):
        return any(_expr_contains_defid(item, target_defid) for item in expr)
    return False


def _conv_spatial_stride_from_index(
    index_expr: Any,
    kernel_red_defid: Any,
) -> Optional[int]:
    if isinstance(index_expr, (IdentifierIR, IndexVarIR)):
        return 1
    if not isinstance(index_expr, BinaryOpIR) or index_expr.operator != BinaryOp.ADD:
        return None
    left = index_expr.left
    right = index_expr.right
    if _expr_contains_defid(left, kernel_red_defid) and _expr_contains_defid(right, kernel_red_defid):
        return None
    if _expr_contains_defid(right, kernel_red_defid):
        left, right = right, left
    if not _expr_contains_defid(left, kernel_red_defid):
        return None
    if isinstance(right, (IdentifierIR, IndexVarIR)):
        return 1
    if isinstance(right, BinaryOpIR) and right.operator == BinaryOp.MUL:
        lit = None
        other = None
        if isinstance(right.left, LiteralIR):
            lit = right.left
            other = right.right
        elif isinstance(right.right, LiteralIR):
            lit = right.right
            other = right.left
        if lit is not None and isinstance(other, (IdentifierIR, IndexVarIR)):
            try:
                return int(lit.value)
            except (TypeError, ValueError):
                return None
    return None


def _decompose_sumprod_body(
    body: ExpressionIR,
) -> Optional[Tuple[RectangularAccessIR, RectangularAccessIR, Optional[ExpressionIR], Optional[float]]]:
    mul_left: Optional[RectangularAccessIR] = None
    mul_right: Optional[RectangularAccessIR] = None
    bias: Optional[ExpressionIR] = None
    scale: Optional[float] = None

    if isinstance(body, BinaryOpIR) and body.operator == BinaryOp.ADD:
        if isinstance(body.left, BinaryOpIR) and body.left.operator == BinaryOp.MUL:
            mul_left = body.left.left if isinstance(body.left.left, RectangularAccessIR) else None
            mul_right = body.left.right if isinstance(body.left.right, RectangularAccessIR) else None
            bias = body.right
        elif isinstance(body.right, BinaryOpIR) and body.right.operator == BinaryOp.MUL:
            mul_left = body.right.left if isinstance(body.right.left, RectangularAccessIR) else None
            mul_right = body.right.right if isinstance(body.right.right, RectangularAccessIR) else None
            bias = body.left
    elif isinstance(body, BinaryOpIR) and body.operator == BinaryOp.MUL:
        if isinstance(body.left, RectangularAccessIR) and isinstance(body.right, RectangularAccessIR):
            mul_left = body.left
            mul_right = body.right
        elif isinstance(body.left, BinaryOpIR) and body.left.operator == BinaryOp.MUL and isinstance(body.right, LiteralIR):
            if isinstance(body.left.left, RectangularAccessIR) and isinstance(body.left.right, RectangularAccessIR):
                mul_left = body.left.left
                mul_right = body.left.right
                try:
                    scale = float(body.right.value)
                except (TypeError, ValueError):
                    scale = None
        elif isinstance(body.right, BinaryOpIR) and body.right.operator == BinaryOp.MUL and isinstance(body.left, LiteralIR):
            if isinstance(body.right.left, RectangularAccessIR) and isinstance(body.right.right, RectangularAccessIR):
                mul_left = body.right.left
                mul_right = body.right.right
                try:
                    scale = float(body.left.value)
                except (TypeError, ValueError):
                    scale = None
    elif isinstance(body, BinaryOpIR) and body.operator == BinaryOp.DIV:
        if isinstance(body.left, BinaryOpIR) and body.left.operator == BinaryOp.MUL:
            if isinstance(body.left.left, RectangularAccessIR) and isinstance(body.left.right, RectangularAccessIR):
                mul_left = body.left.left
                mul_right = body.left.right
                if isinstance(body.right, LiteralIR):
                    try:
                        value = float(body.right.value)
                        if value != 0.0:
                            scale = 1.0 / value
                    except (TypeError, ValueError):
                        scale = None
    if mul_left is None or mul_right is None:
        return None
    return mul_left, mul_right, bias, scale


def _reduction_indices_are_simple(indices: List[Any], reduction_defids: List[Any]) -> bool:
    for idx in indices:
        for red_defid in reduction_defids:
            if _expr_contains_defid(idx, red_defid):
                if not (isinstance(idx, (IdentifierIR, IndexVarIR)) and idx.defid in reduction_defids):
                    return False
                break
    return True


def _recognize_windowed_sumprod(
    loops: List[Any],
    left: RectangularAccessIR,
    right: RectangularAccessIR,
) -> bool:
    n_red = len(loops)
    if n_red not in (2, 3, 4):
        return False
    red_defids = [loop.variable.defid for loop in loops if loop.variable is not None]
    if len(red_defids) != n_red or any(d is None for d in red_defids):
        return False

    il = list(left.indices or [])
    ir = list(right.indices or [])
    if n_red in (3, 4):
        if len(il) != n_red and len(il) != n_red + 1:
            il, ir = ir, il
        if len(il) != n_red and len(il) != n_red + 1:
            return False
        if len(ir) != n_red + 1:
            il, ir = ir, il
        if len(ir) != n_red + 1:
            return False
        if any(_expr_contains_defid(ir[0], rd) for rd in red_defids):
            return False
        n_batch = len(il) - n_red
        if n_batch not in (0, 1):
            return False
        if n_batch == 1 and any(_expr_contains_defid(il[0], rd) for rd in red_defids):
            return False
        ch_idx = il[n_batch]
        if not _expr_contains_defid(ch_idx, red_defids[0]):
            return False
        for k in range(1, n_red + 1):
            if not _expr_contains_defid(ir[k], red_defids[k - 1]):
                return False
        n_spatial = n_red - 1
        for s in range(n_spatial):
            sp_idx = il[n_batch + 1 + s]
            if _conv_spatial_stride_from_index(sp_idx, red_defids[s + 1]) is None:
                return False
        return True

    # 1D case
    if len(il) != 2 or len(ir) != 3:
        il, ir = ir, il
    if len(il) != 2 or len(ir) != 3:
        return False
    return (
        _expr_contains_defid(il[0], red_defids[0])
        and _expr_contains_defid(ir[1], red_defids[0])
        and _expr_contains_defid(ir[2], red_defids[1])
        and _conv_spatial_stride_from_index(il[1], red_defids[1]) is not None
    )


def _recognize_reduction_kernel(expr: LoweredReductionIR) -> Optional[ReductionKernelPlan]:
    if expr.operation != ReductionOp.SUM:
        return None
    if expr.guards or expr.bindings:
        return None
    loops = list(expr.loops or [])
    if not loops:
        return None
    decomposition = _decompose_sumprod_body(expr.body)
    if decomposition is None:
        return None
    left, right, bias, scale = decomposition
    reduction_defids = [loop.variable.defid for loop in loops if loop.variable is not None]
    if len(reduction_defids) != len(loops) or any(d is None for d in reduction_defids):
        return None
    if not _reduction_indices_are_simple(list(left.indices or []) + list(right.indices or []), reduction_defids):
        return None
    if _recognize_windowed_sumprod(loops, left, right):
        kind = "windowed_sumprod"
    elif len(loops) in (1, 2):
        kind = "matmul_sumprod"
    else:
        kind = "einsum_sumprod"
    plan_id = id(expr)
    return ReductionKernelPlan(
        plan_id=plan_id,
        kind=kind,
        left=left,
        right=right,
        bias=bias,
        scale=scale,
    )

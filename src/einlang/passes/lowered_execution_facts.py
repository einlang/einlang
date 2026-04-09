"""Compiler-owned lowered execution facts and reduction kernel plans."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .base import BasePass, TyCtxt
from ..ir.nodes import (
    BinaryOpIR,
    BindingIR,
    ExpressionIR,
    FunctionCallIR,
    IRNode,
    IfExpressionIR,
    IndexVarIR,
    IdentifierIR,
    LiteralIR,
    LoweredEinsteinIR,
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


def build_lowered_execution_facts_analysis(ir: ProgramIR, tcx: Any) -> Dict[str, Any]:
    analyzer = _LoweredExecutionFactsAnalyzer()
    analyzer.walk(ir)
    function_ir_map = getattr(tcx, "function_ir_map", None) or {}
    for binding in function_ir_map.values():
        analyzer.walk(binding)
    try:
        from .autodiff import AutodiffPass

        autodiff_analysis = tcx.get_analysis(AutodiffPass)
        analyzer.walk(autodiff_analysis.get("diff_block"))
    except RuntimeError:
        pass
    return {
        "reduction_facts_by_id": analyzer.reduction_facts_by_id,
        "reduction_kernel_plans_by_id": analyzer.reduction_kernel_plans_by_id,
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
        self._seen_object_ids: set[int] = set()

    def _new_id(self) -> int:
        current = self._next_id
        self._next_id += 1
        return current

    def walk(self, node: Any) -> Any:
        if node is None:
            return None
        if isinstance(node, (list, tuple, dict, IRNode)):
            oid = id(node)
            if oid in self._seen_object_ids:
                return None
            self._seen_object_ids.add(oid)
        if isinstance(node, list):
            for item in node:
                self.walk(item)
            return None
        if isinstance(node, tuple):
            for item in node:
                self.walk(item)
            return None
        if isinstance(node, dict):
            for key, value in node.items():
                self.walk(key)
                self.walk(value)
            return None
        if isinstance(node, LoweredReductionIR):
            self._annotate_lowered_reduction(node)
        if isinstance(node, IRNode):
            for slot in _iter_slots(type(node)):
                self.walk(getattr(node, slot, None))
        return None

    def _annotate_lowered_reduction(self, node: LoweredReductionIR) -> None:
        facts_id = node.execution_facts_id
        if facts_id is not None and facts_id in self.reduction_facts_by_id:
            return
        if facts_id is None:
            facts_id = self._new_id()
            node.execution_facts_id = facts_id

        facts = LoweredReductionExecutionFacts(
            facts_id=facts_id,
            contains_nested_reduction_or_select=_contains_node_type(
                node.body, LoweredReductionIR, LoweredSelectAtArgmaxIR
            ),
            contains_if_expression=_contains_node_type(node.body, IfExpressionIR),
            contains_lowered_einstein=_contains_node_type(node.body, LoweredEinsteinIR),
            body_defids_by_name=_freeze_defids_by_name(_collect_defids_by_name(node.body)),
            guard_defids_by_name=_freeze_defids_by_name(_collect_defids_by_name([g.condition for g in (node.guards or [])])),
        )
        self.reduction_facts_by_id[facts_id] = facts

        plan = _recognize_reduction_kernel(node)
        if plan is not None:
            if node.kernel_plan_id is None:
                node.kernel_plan_id = plan.plan_id
            self.reduction_kernel_plans_by_id[plan.plan_id] = plan


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


def _collect_defids_by_name(node: Any) -> Dict[str, List[Any]]:
    return _DefidsByNameCollector().collect(node)


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

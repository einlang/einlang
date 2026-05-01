"""
Coordinate grounding and shorthand expansion.

This pass is intentionally narrow: it records coordinate layouts introduced by
Einstein bindings, checks bracketed coordinate calls against those layouts, and
expands single-axis selection shorthand such as ``argmax[class](logits)`` into
an explicit indexed reduction body when the tensor layout makes that possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .base import BasePass, TyCtxt
from .rest_pattern_preprocessing import RestPatternPreprocessingPass
from ..ir.nodes import (
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    CastExpressionIR,
    DifferentialIR,
    EinsteinIR,
    FunctionCallIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IndexVarIR,
    MatchExpressionIR,
    ProgramIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    TupleExpressionIR,
    UnaryOpIR,
    WhereExpressionIR,
)
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation
from ..shared.types import ReductionOp, RectangularType


@dataclass(frozen=True)
class CoordinateFacts:
    value_layouts: Dict[DefId, Tuple[str, ...]]
    address_domains: Dict[int, str]


class CoordinateGroundingPass(BasePass):
    """Ground coordinate call arguments and expand selection shorthand."""

    requires = [RestPatternPreprocessingPass]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        self.tcx = tcx
        self.value_layouts: Dict[DefId, Tuple[str, ...]] = {}
        self.address_domains: Dict[int, str] = {}
        self._scope_stack: List[Dict[DefId, Tuple[str, ...]]] = []
        self._current_indices: Dict[str, DefId] = {}

        for stmt in ir.statements or []:
            self._visit_binding(stmt) if isinstance(stmt, BindingIR) else self._visit_expr(stmt)

        tcx.set_analysis(
            CoordinateGroundingPass,
            CoordinateFacts(
                value_layouts=dict(self.value_layouts),
                address_domains=dict(self.address_domains),
            ),
        )
        return ir

    def _lookup_layout(self, defid: Optional[DefId]) -> Optional[Tuple[str, ...]]:
        if defid is None:
            return None
        for scope in reversed(self._scope_stack):
            if defid in scope:
                return scope[defid]
        return self.value_layouts.get(defid)

    def _set_layout(self, defid: Optional[DefId], layout: Optional[Sequence[str]]) -> None:
        if defid is None or not layout:
            return
        normalized = tuple(str(axis) for axis in layout)
        if self._scope_stack:
            self._scope_stack[-1][defid] = normalized
        else:
            self.value_layouts[defid] = normalized

    def _visit_binding(self, node: BindingIR) -> Optional[Tuple[str, ...]]:
        if isinstance(node.expr, FunctionValueIR):
            self._visit_function_value(node.expr)
            return None

        layout = self._visit_expr(node.expr)
        if layout:
            self._set_layout(node.defid, layout)
        return layout

    def _visit_function_value(self, node: FunctionValueIR) -> None:
        scope: Dict[DefId, Tuple[str, ...]] = {}
        for param in node.parameters or ():
            layout = self._layout_from_type(getattr(param, "param_type", None))
            if layout and getattr(param, "defid", None) is not None:
                scope[param.defid] = layout
        self._scope_stack.append(scope)
        try:
            self._visit_expr(node.body)
            custom_diff = getattr(node, "custom_diff_body", None)
            if custom_diff is not None:
                self._visit_expr(custom_diff)
        finally:
            self._scope_stack.pop()

    def _layout_from_type(self, ty: Any) -> Optional[Tuple[str, ...]]:
        if not isinstance(ty, RectangularType) or ty.shape is None:
            return None
        axes = [str(dim) for dim in ty.shape if isinstance(dim, str)]
        return tuple(axes) if axes else None

    def _layout_from_einstein(self, node: EinsteinIR) -> Optional[Tuple[str, ...]]:
        clauses = tuple(node.clauses or ())
        if not clauses:
            return None
        axes: List[str] = []
        for idx in clauses[0].indices or ():
            name = getattr(idx, "name", None)
            if name:
                axes.append(str(name))
        return tuple(axes) if axes else None

    def _visit_einstein(self, node: EinsteinIR) -> Optional[Tuple[str, ...]]:
        layout = self._layout_from_einstein(node)
        for clause in node.clauses or ():
            previous = self._current_indices
            local_indices = dict(previous)
            for idx in clause.indices or ():
                name = getattr(idx, "name", None)
                defid = getattr(idx, "defid", None)
                if name and defid is not None:
                    local_indices[str(name)] = defid
            self._current_indices = local_indices
            try:
                self._visit_expr(clause.value)
            finally:
                self._current_indices = previous
        return layout

    def _visit_expr(self, expr: Any) -> Optional[Tuple[str, ...]]:
        if expr is None:
            return None
        if isinstance(expr, IdentifierIR):
            return self._lookup_layout(expr.defid)
        if isinstance(expr, EinsteinIR):
            return self._visit_einstein(expr)
        if isinstance(expr, RectangularAccessIR):
            return self._visit_rectangular_access(expr)
        if isinstance(expr, FunctionCallIR):
            return self._visit_function_call(expr)
        if isinstance(expr, ReductionExpressionIR):
            return self._visit_reduction(expr)
        if isinstance(expr, BlockExpressionIR):
            return self._visit_block(expr)
        if isinstance(expr, IfExpressionIR):
            self._visit_expr(expr.condition)
            then_layout = self._visit_expr(expr.then_expr)
            else_layout = self._visit_expr(expr.else_expr)
            return then_layout or else_layout
        if isinstance(expr, WhereExpressionIR):
            layout = self._visit_expr(expr.expr)
            for constraint in expr.constraints or ():
                self._visit_expr(constraint)
            return layout
        if isinstance(expr, DifferentialIR):
            return self._visit_expr(expr.operand)
        if isinstance(expr, CastExpressionIR):
            return self._visit_expr(expr.expr)
        if isinstance(expr, UnaryOpIR):
            return self._visit_expr(expr.operand)
        if isinstance(expr, BinaryOpIR):
            self._visit_expr(expr.left)
            self._visit_expr(expr.right)
            return None
        if isinstance(expr, TupleExpressionIR):
            for item in expr.elements or ():
                self._visit_expr(item)
            return None
        if isinstance(expr, MatchExpressionIR):
            self._visit_expr(expr.scrutinee)
            for arm in expr.arms or ():
                self._visit_expr(getattr(arm, "body", None))
            return None
        if hasattr(expr, "arguments"):
            for arg in getattr(expr, "arguments", ()) or ():
                self._visit_expr(arg)
        return None

    def _visit_block(self, node: BlockExpressionIR) -> Optional[Tuple[str, ...]]:
        self._scope_stack.append({})
        try:
            for stmt in node.statements or ():
                self._visit_binding(stmt) if isinstance(stmt, BindingIR) else self._visit_expr(stmt)
            return self._visit_expr(node.final_expr)
        finally:
            self._scope_stack.pop()

    def _visit_rectangular_access(self, node: RectangularAccessIR) -> Optional[Tuple[str, ...]]:
        base_layout = self._visit_expr(node.array)
        for idx in node.indices or ():
            self._visit_expr(idx)
        if not base_layout:
            return None
        remaining = []
        for axis, idx in zip(base_layout, node.indices or ()):
            if isinstance(idx, (IdentifierIR, IndexVarIR)) and idx.name == axis:
                continue
            remaining.append(axis)
        if len(node.indices or ()) >= len(base_layout):
            return tuple(remaining) if remaining else None
        return tuple(base_layout[len(node.indices or ()):])

    def _visit_function_call(self, node: FunctionCallIR) -> Optional[Tuple[str, ...]]:
        arg_layouts = [self._visit_expr(arg) for arg in node.arguments or ()]
        grounded = set()
        for arg in node.arguments or ():
            grounded.update(self._grounded_coordinates(arg))
        for layout in arg_layouts:
            grounded.update(layout or ())

        for coord_arg in node.coordinate_args or ():
            name = getattr(coord_arg, "name", None)
            if name and name not in grounded:
                self._report_ungrounded(name, node.location, getattr(node, "function_name", "call"))

        return next((layout for layout in arg_layouts if layout), None)

    def _visit_reduction(self, node: ReductionExpressionIR) -> Optional[Tuple[str, ...]]:
        selected = tuple(
            var.name
            for var in node.loop_vars or ()
            if getattr(var, "name", None)
        )
        if node.operation in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
            self._expand_selection_shorthand(node, selected)

        body_layout = self._visit_expr(node.body)
        grounded = self._grounded_coordinates(node.body)
        grounded.update(body_layout or ())
        if node.operation in (ReductionOp.ARGMAX, ReductionOp.ARGMIN):
            for name in selected:
                if name not in grounded:
                    self._report_ungrounded(name, node.location, node.operation.value)

        if node.where_clause:
            for constraint in node.where_clause.constraints or ():
                self._visit_expr(constraint)

        if node.operation in (ReductionOp.ARGMAX, ReductionOp.ARGMIN) and selected:
            self.address_domains[id(node)] = selected[0]

        if body_layout:
            remaining = tuple(axis for axis in body_layout if axis not in selected)
            return remaining if remaining else None
        return None

    def _expand_selection_shorthand(
        self, node: ReductionExpressionIR, selected: Sequence[str]
    ) -> None:
        if len(selected) != 1 or not isinstance(node.body, IdentifierIR):
            return
        body_layout = self._lookup_layout(node.body.defid)
        if not body_layout or selected[0] not in body_layout:
            return

        loop_var_by_name = {
            var.name: var
            for var in node.loop_vars or ()
            if getattr(var, "name", None)
        }
        indices = []
        for axis in body_layout:
            loop_var = loop_var_by_name.get(axis)
            if loop_var is not None:
                indices.append(
                    IdentifierIR(axis, node.location, defid=getattr(loop_var, "defid", None))
                )
                continue
            defid = self._current_indices.get(axis)
            if defid is None:
                self._report_error(
                    f"coordinate shorthand for `{node.operation.value}[{selected[0]}]` "
                    f"needs remaining coordinate `{axis}` from the result context",
                    node.location,
                    help="Use an explicit indexed body such as logits[b, class].",
                )
                return
            indices.append(IdentifierIR(axis, node.location, defid=defid))

        node.body = RectangularAccessIR(
            array=IdentifierIR(node.body.name, node.body.location, defid=node.body.defid),
            indices=indices,
            location=node.location or node.body.location or SourceLocation("", 0, 0),
        )

    def _grounded_coordinates(self, expr: Any) -> Set[str]:
        if expr is None:
            return set()
        if isinstance(expr, IdentifierIR):
            return set(self._lookup_layout(expr.defid) or ())
        if isinstance(expr, DifferentialIR):
            return self._grounded_coordinates(expr.operand)
        if isinstance(expr, RectangularAccessIR):
            names = set(self._grounded_coordinates(expr.array))
            for idx in expr.indices or ():
                if isinstance(idx, (IdentifierIR, IndexVarIR)) and idx.name:
                    names.add(idx.name)
                names.update(self._grounded_coordinates(idx))
            return names
        if isinstance(expr, ReductionExpressionIR):
            names = set()
            for var in expr.loop_vars or ():
                if getattr(var, "name", None):
                    names.add(var.name)
            names.update(self._grounded_coordinates(expr.body))
            return names
        if isinstance(expr, FunctionCallIR):
            names = set()
            for arg in expr.arguments or ():
                names.update(self._grounded_coordinates(arg))
            return names
        return set()

    def _report_ungrounded(self, coord: str, location: Optional[SourceLocation], context: str) -> None:
        self._report_error(
            f"coordinate `{coord}` is not grounded in `{context}`",
            location,
            code="E0701",
            help="Annotate the value with coordinates or use an explicit indexed argument.",
        )

    def _report_error(
        self,
        message: str,
        location: Optional[SourceLocation],
        code: Optional[str] = None,
        help: Optional[str] = None,
    ) -> None:
        reporter = getattr(self.tcx, "reporter", None)
        if reporter is not None:
            reporter.report_error(message, location, code=code, help=help)
        else:
            raise ValueError(message)

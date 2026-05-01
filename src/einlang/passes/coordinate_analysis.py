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
from .coordinate_utils import (
    coord_param_name,
    coordinate_arg_names,
    instantiate_symbolic_sequence,
    is_coord_pack_param,
    layout_from_type,
    match_symbolic_sequence,
)
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
from ..shared.types import ReductionOp


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
        self._function_map: Dict[DefId, BindingIR] = {}
        for binding in list(getattr(ir, "functions", ()) or ()) + list(getattr(ir, "statements", ()) or ()):
            if (
                isinstance(binding, BindingIR)
                and isinstance(binding.expr, FunctionValueIR)
                and binding.defid is not None
            ):
                self._function_map[binding.defid] = binding

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

    def _stamp_layout(self, expr: Any, layout: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        normalized = tuple(str(axis) for axis in layout or ())
        try:
            expr.coordinate_layout = normalized if normalized else None
        except AttributeError:
            pass
        return normalized if normalized else None

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
            self._stamp_layout(node.expr, layout)
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
        return layout_from_type(ty)

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
            return self._stamp_layout(expr, self._lookup_layout(expr.defid))
        if isinstance(expr, EinsteinIR):
            return self._stamp_layout(expr, self._visit_einstein(expr))
        if isinstance(expr, RectangularAccessIR):
            return self._stamp_layout(expr, self._visit_rectangular_access(expr))
        if isinstance(expr, FunctionCallIR):
            return self._stamp_layout(expr, self._visit_function_call(expr))
        if isinstance(expr, ReductionExpressionIR):
            return self._stamp_layout(expr, self._visit_reduction(expr))
        if isinstance(expr, BlockExpressionIR):
            return self._stamp_layout(expr, self._visit_block(expr))
        if isinstance(expr, IfExpressionIR):
            self._visit_expr(expr.condition)
            then_layout = self._visit_expr(expr.then_expr)
            else_layout = self._visit_expr(expr.else_expr)
            return self._stamp_layout(expr, then_layout or else_layout)
        if isinstance(expr, WhereExpressionIR):
            layout = self._visit_expr(expr.expr)
            for constraint in expr.constraints or ():
                self._visit_expr(constraint)
            return self._stamp_layout(expr, layout)
        if isinstance(expr, DifferentialIR):
            return self._stamp_layout(expr, self._visit_expr(expr.operand))
        if isinstance(expr, CastExpressionIR):
            return self._stamp_layout(expr, self._visit_expr(expr.expr))
        if isinstance(expr, UnaryOpIR):
            return self._stamp_layout(expr, self._visit_expr(expr.operand))
        if isinstance(expr, BinaryOpIR):
            left_layout = self._visit_expr(expr.left)
            right_layout = self._visit_expr(expr.right)
            if left_layout and right_layout:
                return self._stamp_layout(expr, left_layout if left_layout == right_layout else None)
            return self._stamp_layout(expr, left_layout or right_layout)
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
        signature_layout = self._layout_from_call_signature(node, arg_layouts)
        grounded = set()
        for arg in node.arguments or ():
            grounded.update(self._grounded_coordinates(arg))
        for layout in arg_layouts:
            grounded.update(layout or ())
        grounded.update(signature_layout or ())

        for coord_arg in node.coordinate_args or ():
            for name in coordinate_arg_names(coord_arg):
                if name and name not in grounded:
                    self._report_ungrounded(name, node.location, getattr(node, "function_name", "call"))

        return signature_layout or next((layout for layout in arg_layouts if layout), None)

    def _function_value_for_call(self, node: FunctionCallIR) -> Optional[FunctionValueIR]:
        defid = node.function_defid
        if defid is None:
            return None
        func_map = getattr(self.tcx, "function_ir_map", None)
        binding = func_map.get(defid) if func_map else None
        if binding is None:
            binding = self._function_map.get(defid)
        expr = getattr(binding, "expr", None)
        return expr if isinstance(expr, FunctionValueIR) else None

    def _layout_from_call_signature(
        self,
        node: FunctionCallIR,
        arg_layouts: Sequence[Optional[Tuple[str, ...]]],
    ) -> Optional[Tuple[str, ...]]:
        func = self._function_value_for_call(node)
        if func is None or not getattr(func, "coordinate_params", None):
            return None

        coord_params = tuple(func.coordinate_params or ())
        coord_args = tuple(node.coordinate_args or ())
        axes: Dict[str, str] = {}
        axis_packs: Dict[str, Tuple[str, ...]] = {}
        if len(coord_args) > len(coord_params):
            self._report_error(
                f"function `{node.function_name}` expects {len(coord_params)} coordinate "
                f"argument{'s' if len(coord_params) != 1 else ''} at most, got {len(coord_args)}",
                node.location,
                code="E0061",
            )
            return None
        for param, arg in zip(coord_params, coord_args):
            names = coordinate_arg_names(arg)
            if is_coord_pack_param(param):
                if not isinstance(arg, TupleExpressionIR):
                    self._report_error(
                        f"coordinate parameter `{param}` expects a parenthesized coordinate group",
                        node.location,
                        code="E0061",
                    )
                    return None
                axis_packs[coord_param_name(param)] = names
            else:
                if isinstance(arg, TupleExpressionIR) or len(names) != 1:
                    self._report_error(
                        f"coordinate parameter `{param}` expects one coordinate",
                        node.location,
                        code="E0061",
                    )
                    return None
                axes[str(param)] = names[0]

        dims: Dict[str, str] = {}
        packs: Dict[str, Tuple[str, ...]] = dict(axis_packs)
        for param, layout in zip(func.parameters or (), arg_layouts):
            if not layout:
                continue
            formal = self._layout_from_type(getattr(param, "param_type", None))
            if formal is not None:
                match_symbolic_sequence(formal, layout, axes=axes, axis_packs=axis_packs, dims=dims, packs=packs)

        formal_return = self._layout_from_type(getattr(func, "return_type", None))
        if formal_return is None:
            return None
        instantiated = instantiate_symbolic_sequence(formal_return, axes=axes, dims=dims, packs=packs)
        node.coordinate_axis_bindings = {
            str(name): str(axis)
            for name, axis in dict(list(dims.items()) + list(axes.items())).items()
        }
        node.coordinate_pack_bindings = {
            str(name): tuple(str(axis) for axis in axes_tuple)
            for name, axes_tuple in packs.items()
        }
        return tuple(str(axis) for axis in instantiated) if instantiated else None

    def _visit_reduction(self, node: ReductionExpressionIR) -> Optional[Tuple[str, ...]]:
        selected = tuple(
            var.name
            for var in node.loop_vars or ()
            if getattr(var, "name", None)
        )
        previous = self._current_indices
        local_indices = dict(previous)
        for var in node.loop_vars or ():
            name = getattr(var, "name", None)
            defid = getattr(var, "defid", None)
            if name and defid is not None:
                local_indices[str(name)] = defid
        self._current_indices = local_indices
        try:
            self._expand_reduction_shorthand(node, selected)

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
        finally:
            self._current_indices = previous

        if node.operation in (ReductionOp.ARGMAX, ReductionOp.ARGMIN) and selected:
            self.address_domains[id(node)] = selected[0]
            node.coordinate_address_domain = selected[0]

        if body_layout:
            remaining = tuple(axis for axis in body_layout if axis not in selected)
            return remaining if remaining else None
        return None

    def _expand_reduction_shorthand(
        self, node: ReductionExpressionIR, selected: Sequence[str]
    ) -> None:
        if not selected:
            return
        if node.operation in (ReductionOp.ARGMAX, ReductionOp.ARGMIN) and len(selected) != 1:
            return
        selected_defids: Dict[str, Optional[DefId]] = {
            var.name: getattr(var, "defid", None)
            for var in node.loop_vars or ()
            if getattr(var, "name", None)
        }
        expanded = self._expand_expr_for_selected_axis(
            node.body, set(selected), node.location, selected_defids
        )
        if expanded is None:
            return
        node.body = expanded

    def _expand_expr_for_selected_axis(
        self,
        expr: Any,
        selected: Set[str],
        location: Optional[SourceLocation],
        selected_defids: Dict[str, Optional[DefId]],
    ) -> Optional[Any]:
        if isinstance(expr, IdentifierIR):
            body_layout = self._lookup_layout(expr.defid)
            if not body_layout or not selected.intersection(body_layout):
                return None

            indices = []
            for axis in body_layout:
                if axis in selected:
                    indices.append(IdentifierIR(axis, location, defid=selected_defids.get(axis)))
                    continue
                defid = self._current_indices.get(axis)
                if defid is None:
                    self._report_error(
                        f"coordinate shorthand for reduction over `{', '.join(sorted(selected))}` "
                        f"needs remaining coordinate `{axis}` from the result context",
                        location,
                        help=(
                            "Use an explicit indexed body such as "
                            f"value[{axis}, {', '.join(sorted(selected))}]."
                        ),
                    )
                    return None
                indices.append(IdentifierIR(axis, location, defid=defid))

            return RectangularAccessIR(
                array=IdentifierIR(expr.name, expr.location, defid=expr.defid),
                indices=indices,
                location=location or expr.location or SourceLocation("", 0, 0),
            )

        if isinstance(expr, BinaryOpIR):
            left = self._expand_expr_for_selected_axis(expr.left, selected, location, selected_defids)
            right = self._expand_expr_for_selected_axis(expr.right, selected, location, selected_defids)
            if left is not None:
                expr.left = left
            if right is not None:
                expr.right = right
            return expr if left is not None or right is not None else None

        if isinstance(expr, UnaryOpIR):
            operand = self._expand_expr_for_selected_axis(expr.operand, selected, location, selected_defids)
            if operand is None:
                return None
            expr.operand = operand
            return expr

        if isinstance(expr, CastExpressionIR):
            inner = self._expand_expr_for_selected_axis(expr.expr, selected, location, selected_defids)
            if inner is None:
                return None
            expr.expr = inner
            return expr

        if isinstance(expr, IfExpressionIR):
            then_expr = self._expand_expr_for_selected_axis(
                expr.then_expr, selected, location, selected_defids
            )
            else_expr = self._expand_expr_for_selected_axis(
                expr.else_expr, selected, location, selected_defids
            )
            if then_expr is not None:
                expr.then_expr = then_expr
            if else_expr is not None:
                expr.else_expr = else_expr
            return expr if then_expr is not None or else_expr is not None else None

        return None

    def _grounded_coordinates(self, expr: Any) -> Set[str]:
        if expr is None:
            return set()
        if isinstance(expr, IdentifierIR):
            return set(self._lookup_layout(expr.defid) or ())
        if isinstance(expr, DifferentialIR):
            return self._grounded_coordinates(expr.operand)
        if isinstance(expr, CastExpressionIR):
            return self._grounded_coordinates(expr.expr)
        if isinstance(expr, UnaryOpIR):
            return self._grounded_coordinates(expr.operand)
        if isinstance(expr, BinaryOpIR):
            names = self._grounded_coordinates(expr.left)
            names.update(self._grounded_coordinates(expr.right))
            return names
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

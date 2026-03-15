"""
Autodiff pass: expand @ and @y/@x into plain IR (forward diff).

We implement **forward diff** only: from dx to dy. For each binding y = f(x1, x2, ...)
we emit d_y = (∂f/∂x1)*d_x1 + (∂f/∂x2)*d_x2 + ... in execution order. No backward
pass (no propagation from dy to dx), no pullbacks, no tape.

- DifferentialIR (@expr): add d_* bindings; @x -> d_x, @(expr) -> d(expr) via chain rule.
- DIV(@num,@den): @(expr)/@x -> derivative expr (e.g. 2*x); @y/@x -> d_y when d_x=1.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BasePass, TyCtxt
from .type_inference import TypeInferencePass
from .shape_analysis import UnifiedShapeAnalysisPass
from ..ir.nodes import (
    ProgramIR,
    BindingIR,
    BlockExpressionIR,
    DifferentialIR,
    BinaryOpIR,
    UnaryOpIR,
    BuiltinCallIR,
    LiteralIR,
    IdentifierIR,
    ExpressionIR,
    IRVisitor,
    RectangularAccessIR,
    IndexVarIR,
    FunctionCallIR,
    FunctionValueIR,
    EinsteinIR,
    EinsteinClauseIR,
    ReductionExpressionIR,
    WhereClauseIR,
)
from ..shared.types import BinaryOp, UnaryOp
from ..shared.defid import DefId
from ..shared.source_location import SourceLocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bindings_in_block(block: Any, program: Optional[ProgramIR] = None) -> List[BindingIR]:
    if block is program or isinstance(block, ProgramIR):
        return [b for b in (program.bindings or []) if isinstance(b, BindingIR)] if program else []
    if isinstance(block, BlockExpressionIR):
        return [s for s in (block.statements or []) if isinstance(s, BindingIR)]
    return []


def _differential_target_from_operand(operand: ExpressionIR) -> Optional[Tuple[DefId, str]]:
    if isinstance(operand, IdentifierIR) and operand.defid is not None:
        return (operand.defid, operand.name or "")
    return None


def _defid_from_expr(expr: Optional[ExpressionIR]) -> Optional[DefId]:
    if expr is None:
        return None
    if isinstance(expr, IdentifierIR):
        return expr.defid
    return None


def _flatten_product(expr: ExpressionIR) -> Optional[List[Tuple[ExpressionIR, List[Any]]]]:
    if isinstance(expr, RectangularAccessIR):
        arr = expr.array
        indices = list(expr.indices or [])
        if isinstance(arr, IdentifierIR) and arr.defid is not None:
            return [(arr, indices)]
        return None
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.MUL:
        left = _flatten_product(expr.left)
        right = _flatten_product(expr.right)
        if left is not None and right is not None:
            return left + right
    return None


def _index_defids(indices: List[Any]) -> Set[DefId]:
    out: Set[DefId] = set()
    for idx in indices:
        if isinstance(idx, (IndexVarIR, IdentifierIR)) and idx.defid is not None:
            out.add(idx.defid)
    return out


# ---------------------------------------------------------------------------
# DefId collector (for dependency graph)
# ---------------------------------------------------------------------------

class _DefIdCollector(IRVisitor[None]):
    def __init__(self, out: Set[DefId]) -> None:
        self._out = out

    def visit_identifier(self, node: IdentifierIR) -> None:
        if node.defid is not None:
            self._out.add(node.defid)

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        node.operand.accept(self)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        for a in node.args or []:
            a.accept(self)

    def visit_differential(self, node: DifferentialIR) -> None:
        node.operand.accept(self)

    def visit_literal(self, node: Any) -> None:
        pass

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        for s in node.statements or []:
            if hasattr(s, "accept"):
                s.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_rectangular_access(self, node: Any) -> None:
        node.array.accept(self)
        for i in node.indices or []:
            i.accept(self)

    def visit_if_expression(self, node: Any) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_cast_expression(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_tuple_expression(self, node: Any) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_access(self, node: Any) -> None:
        node.tuple_expr.accept(self)

    def visit_member_access(self, node: Any) -> None:
        node.object.accept(self)

    def visit_function_call(self, node: Any) -> None:
        if node.callee_expr is not None:
            node.callee_expr.accept(self)
        for a in node.arguments or []:
            a.accept(self)

    def visit_range(self, node: Any) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_reduction_expression(self, node: Any) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: Any) -> None:
        node.expr.accept(self)
        for c in node.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_array_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_array_literal(self, node: Any) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_lambda(self, node: Any) -> None:
        node.body.accept(self)

    def visit_function_value(self, node: Any) -> None:
        if node.body is not None:
            node.body.accept(self)

    def visit_try_expression(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_jagged_access(self, node: Any) -> None:
        if node.base is not None:
            node.base.accept(self)

    def visit_binding(self, node: BindingIR) -> None:
        if node.expr is not None:
            node.expr.accept(self)

    def visit_program(self, node: ProgramIR) -> None:
        for b in node.bindings or []:
            b.accept(self)

    def visit_einstein(self, node: Any) -> None:
        for c in getattr(node, "clauses", None) or []:
            if hasattr(c, "accept"):
                c.accept(self)

    def visit_einstein_clause(self, node: Any) -> None:
        if getattr(node, "value", None) is not None:
            node.value.accept(self)

    def visit_module(self, node: Any) -> None:
        pass

    def visit_interpolated_string(self, node: Any) -> None:
        pass

    def visit_identifier_pattern(self, node: Any) -> None:
        pass

    def visit_wildcard_pattern(self, node: Any) -> None:
        pass

    def visit_literal_pattern(self, node: Any) -> None:
        pass

    def visit_tuple_pattern(self, node: Any) -> None:
        pass

    def visit_array_pattern(self, node: Any) -> None:
        pass

    def visit_rest_pattern(self, node: Any) -> None:
        pass

    def visit_guard_pattern(self, node: Any) -> None:
        pass

    def visit_or_pattern(self, node: Any) -> None:
        pass

    def visit_constructor_pattern(self, node: Any) -> None:
        pass

    def visit_binding_pattern(self, node: Any) -> None:
        pass

    def visit_range_pattern(self, node: Any) -> None:
        pass

    def visit_index_var(self, node: Any) -> None:
        pass

    def visit_index_rest(self, node: Any) -> None:
        pass


def _collect_defids_in_expr(expr: Optional[ExpressionIR]) -> Set[DefId]:
    out: Set[DefId] = set()
    if expr is not None:
        expr.accept(_DefIdCollector(out))
    return out


# ---------------------------------------------------------------------------
# Differential targets collector
# ---------------------------------------------------------------------------

class _DifferentialCollector(IRVisitor[None]):
    def __init__(self) -> None:
        self.targets: List[Tuple[DefId, str]] = []

    def visit_differential(self, node: DifferentialIR) -> None:
        t = _differential_target_from_operand(node.operand)
        if t is not None:
            self.targets.append(t)
        node.operand.accept(self)

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        node.operand.accept(self)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        for a in node.args or []:
            a.accept(self)

    def visit_identifier(self, node: IdentifierIR) -> None:
        pass

    def visit_literal(self, node: Any) -> None:
        pass

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        for s in node.statements or []:
            if hasattr(s, "accept"):
                s.accept(self)
        if node.final_expr is not None:
            node.final_expr.accept(self)

    def visit_binding(self, node: BindingIR) -> None:
        if node.expr is not None:
            node.expr.accept(self)

    def visit_program(self, node: ProgramIR) -> None:
        for b in node.bindings or []:
            b.accept(self)

    def visit_function_value(self, node: Any) -> None:
        if node.body is not None:
            node.body.accept(self)

    def visit_if_expression(self, node: Any) -> None:
        node.condition.accept(self)
        node.then_expr.accept(self)
        if node.else_expr is not None:
            node.else_expr.accept(self)

    def visit_rectangular_access(self, node: Any) -> None:
        node.array.accept(self)
        for i in node.indices or []:
            i.accept(self)

    def visit_cast_expression(self, node: Any) -> None:
        node.expr.accept(self)

    def visit_tuple_expression(self, node: Any) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_tuple_access(self, node: Any) -> None:
        node.tuple_expr.accept(self)

    def visit_member_access(self, node: Any) -> None:
        node.object.accept(self)

    def visit_reduction_expression(self, node: Any) -> None:
        node.body.accept(self)

    def visit_where_expression(self, node: Any) -> None:
        node.expr.accept(self)
        for c in node.constraints or []:
            c.accept(self)

    def visit_pipeline_expression(self, node: Any) -> None:
        node.left.accept(self)
        node.right.accept(self)

    def visit_range(self, node: Any) -> None:
        node.start.accept(self)
        node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> None:
        node.body.accept(self)

    def visit_array_literal(self, node: Any) -> None:
        for e in node.elements or []:
            e.accept(self)

    def visit_lambda(self, node: Any) -> None:
        node.body.accept(self)

    def visit_function_call(self, node: Any) -> None:
        if node.callee_expr is not None:
            node.callee_expr.accept(self)
        for a in node.arguments or []:
            a.accept(self)

    def visit_try_expression(self, node: Any) -> None:
        node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> None:
        node.scrutinee.accept(self)
        for arm in node.arms or []:
            if getattr(arm, "body", None) is not None:
                arm.body.accept(self)

    def visit_jagged_access(self, node: Any) -> None:
        if node.base is not None:
            node.base.accept(self)

    def visit_einstein(self, node: Any) -> None:
        for c in getattr(node, "clauses", None) or []:
            if hasattr(c, "accept"):
                c.accept(self)

    def visit_einstein_clause(self, node: Any) -> None:
        if getattr(node, "value", None) is not None:
            node.value.accept(self)

    def visit_module(self, node: Any) -> None:
        pass

    def visit_interpolated_string(self, node: Any) -> None:
        pass

    def visit_identifier_pattern(self, node: Any) -> None:
        pass

    def visit_wildcard_pattern(self, node: Any) -> None:
        pass

    def visit_literal_pattern(self, node: Any) -> None:
        pass

    def visit_tuple_pattern(self, node: Any) -> None:
        pass

    def visit_array_pattern(self, node: Any) -> None:
        pass

    def visit_rest_pattern(self, node: Any) -> None:
        pass

    def visit_guard_pattern(self, node: Any) -> None:
        pass

    def visit_or_pattern(self, node: Any) -> None:
        pass

    def visit_constructor_pattern(self, node: Any) -> None:
        pass

    def visit_binding_pattern(self, node: Any) -> None:
        pass

    def visit_range_pattern(self, node: Any) -> None:
        pass

    def visit_index_var(self, node: Any) -> None:
        pass

    def visit_index_rest(self, node: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Forward diff: d_y from d_x1, d_x2 (chain rule in forward direction)
# ---------------------------------------------------------------------------

def _forward_binary(
    node: BinaryOpIR,
    d_left: ExpressionIR,
    d_right: ExpressionIR,
    loc: SourceLocation,
) -> ExpressionIR:
    """Forward: d_y = (∂y/∂left)*d_left + (∂y/∂right)*d_right."""
    left, right = node.left, node.right
    op = node.operator
    if op == BinaryOp.ADD:
        return BinaryOpIR(BinaryOp.ADD, d_left, d_right, loc)
    if op == BinaryOp.SUB:
        return BinaryOpIR(BinaryOp.SUB, d_left, d_right, loc)
    if op == BinaryOp.MUL:
        return BinaryOpIR(
            BinaryOp.ADD,
            BinaryOpIR(BinaryOp.MUL, d_left, right, loc),
            BinaryOpIR(BinaryOp.MUL, left, d_right, loc),
            loc,
        )
    if op == BinaryOp.DIV:
        # d(y) = d(left/right) = d_left/right - left*d_right/right^2
        t1 = BinaryOpIR(BinaryOp.DIV, d_left, right, loc)
        t2 = BinaryOpIR(BinaryOp.DIV, BinaryOpIR(BinaryOp.MUL, left, d_right, loc), BinaryOpIR(BinaryOp.POW, right, LiteralIR(2, loc), loc), loc)
        return BinaryOpIR(BinaryOp.SUB, t1, t2, loc)
    if op == BinaryOp.POW:
        # y = left^right => d_y = right*left^(right-1)*d_left + left^right*ln(left)*d_right (skip second if const)
        b_minus_one = BinaryOpIR(BinaryOp.SUB, right, LiteralIR(1, loc), loc)
        a_bm1 = BinaryOpIR(BinaryOp.POW, left, b_minus_one, loc)
        term_left = BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.MUL, right, a_bm1, loc), d_left, loc)
        a_b = BinaryOpIR(BinaryOp.POW, left, right, loc)
        term_right = BinaryOpIR(BinaryOp.MUL, a_b, d_right, loc)
        return BinaryOpIR(BinaryOp.ADD, term_left, term_right, loc)
    raise ValueError(f"Autodiff: unsupported binary op: {op}")


def _forward_unary(
    node: UnaryOpIR,
    d_operand: ExpressionIR,
    loc: SourceLocation,
) -> ExpressionIR:
    """Forward: d_y = (∂y/∂operand)*d_operand."""
    if node.operator == UnaryOp.NEG:
        return UnaryOpIR(UnaryOp.NEG, d_operand, loc)
    if node.operator == UnaryOp.POS:
        return d_operand
    raise ValueError(f"Autodiff: unsupported unary op: {node.operator}")


def _diff_einstein_wrt(
    expr: EinsteinIR,
    wrt_defid: DefId,
    loc: SourceLocation,
    binding_by_defid: Optional[Dict[DefId, Any]] = None,
    resolver: Optional[Any] = None,
) -> ExpressionIR:
    """∂(Einstein)/∂(wrt): build EinsteinIR for derivative tensor. Follows AUTODIFF_EINSTEIN.md."""
    if binding_by_defid is None:
        raise ValueError("Autodiff: cannot differentiate EinsteinIR without binding context")
    if resolver is None:
        raise ValueError("Autodiff: cannot differentiate EinsteinIR without resolver (need DefIds for new index vars)")
    derivative_clauses: List[EinsteinClauseIR] = []
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR) or (val.operation or "").lower() != "sum":
            raise ValueError("Autodiff: Einstein ∂/∂wrt only supports sum-of-products clauses")
        inner = val.body
        factors = _flatten_product(inner) if inner else None
        if not factors:
            raise ValueError("Autodiff: Einstein clause body is not a product of indexed arrays")
        # All indices of wrt in this clause (AUTODIFF_EINSTEIN.md §4 multi-clause; §5 multiple factors).
        wrt_occurrence_positions: List[int] = [
            i for i, (arr_expr, _) in enumerate(factors)
            if isinstance(arr_expr, IdentifierIR) and arr_expr.defid == wrt_defid
        ]
        if not wrt_occurrence_positions:
            continue
        # Derivative indices R: one new index var per dimension of the wrt array (same for all occurrences).
        first_wrt_indices = factors[wrt_occurrence_positions[0]][1]
        new_index_vars: List[IndexVarIR] = []
        for p in range(len(first_wrt_indices)):
            new_defid = resolver.allocate_for_local()
            new_index_vars.append(IndexVarIR("_ad_%d" % p, loc, new_defid))
        clause_indices = list(clause.indices or [])
        loop_vars = list(val.loop_vars or [])
        # Original reduction where-clause φ_red (AUTODIFF_EINSTEIN.md §6: conjoin with delta).
        original_red_constraints: List[ExpressionIR] = (
            list(getattr(val.where_clause, "constraints", None) or [])
            if getattr(val, "where_clause", None) is not None else []
        )
        reduction_terms: List[ExpressionIR] = []
        for pos in wrt_occurrence_positions:
            _, wrt_indices = factors[pos]
            # Other factors: all except this occurrence (doc §5: one term per occurrence).
            other_factors = [factors[i] for i in range(len(factors)) if i != pos]
            # Delta (J = R): constraints so index tuple J equals derivative position R.
            delta_constraints: List[ExpressionIR] = [
                BinaryOpIR(BinaryOp.EQ, idx_expr, new_var, loc)
                for idx_expr, new_var in zip(wrt_indices, new_index_vars)
            ]
            combined_constraints = original_red_constraints + delta_constraints
            where = WhereClauseIR(constraints=combined_constraints, location=loc) if combined_constraints else None
            # Body: product of other factors (wrt factor replaced by δ via where).
            if not other_factors:
                body = LiteralIR(1, loc)
            else:
                b = binding_by_defid.get(other_factors[0][0].defid) if isinstance(other_factors[0][0], IdentifierIR) else None
                nm = other_factors[0][0].name or (b.name if b else "") or ""
                ref0 = IdentifierIR(nm, loc, other_factors[0][0].defid) if isinstance(other_factors[0][0], IdentifierIR) else other_factors[0][0]
                body = RectangularAccessIR(ref0, list(other_factors[0][1]), loc)
                for arr_expr, idx_list in other_factors[1:]:
                    b = binding_by_defid.get(arr_expr.defid) if isinstance(arr_expr, IdentifierIR) else None
                    nm = arr_expr.name or (b.name if b else "") or ""
                    ref = IdentifierIR(nm, loc, arr_expr.defid) if isinstance(arr_expr, IdentifierIR) else arr_expr
                    body = BinaryOpIR(BinaryOp.MUL, body, RectangularAccessIR(ref, list(idx_list), loc), loc)
            red = ReductionExpressionIR(
                operation=val.operation,
                loop_vars=loop_vars,
                body=body,
                location=loc,
                where_clause=where,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                type_info=val.type_info,
                shape_info=val.shape_info,
            )
            reduction_terms.append(red)
        # Sum of reduction terms (one per wrt occurrence; doc §5).
        clause_value: ExpressionIR = reduction_terms[0]
        for r in reduction_terms[1:]:
            clause_value = BinaryOpIR(BinaryOp.ADD, clause_value, r, loc)
        new_indices = clause_indices + new_index_vars
        new_variable_ranges = dict(clause.variable_ranges or {})
        for p in range(len(first_wrt_indices)):
            idx_expr = first_wrt_indices[p]
            if isinstance(idx_expr, (IndexVarIR, IdentifierIR)) and getattr(idx_expr, "defid", None) is not None:
                rng = (clause.variable_ranges or {}).get(idx_expr.defid)
                if rng is not None and new_index_vars[p].defid is not None:
                    new_variable_ranges[new_index_vars[p].defid] = rng
        new_clause = EinsteinClauseIR(
            indices=new_indices,
            value=clause_value,
            location=clause.location,
            where_clause=clause.where_clause,
            variable_ranges=new_variable_ranges,
        )
        derivative_clauses.append(new_clause)
    if not derivative_clauses:
        return LiteralIR(0, loc)
    # Single EinsteinIR with all derivative clauses (doc §4: multi-clause → add into same tensor).
    return EinsteinIR(
        clauses=derivative_clauses,
        shape=None,
        element_type=expr.element_type,
        location=expr.location,
        type_info=expr.type_info,
        shape_info=expr.shape_info,
    )


def _diff_expr_wrt(
    expr: ExpressionIR,
    wrt_defid: DefId,
    loc: SourceLocation,
    binding_by_defid: Optional[Dict[DefId, Any]] = None,
    resolver: Optional[Any] = None,
) -> ExpressionIR:
    """Symbolic partial ∂expr/∂wrt. Raises for unsupported expression kinds. binding_by_defid required for FunctionCallIR."""
    if isinstance(expr, BlockExpressionIR):
        if expr.final_expr is not None:
            return _diff_expr_wrt(expr.final_expr, wrt_defid, loc, binding_by_defid, resolver)
        raise ValueError("Autodiff: cannot differentiate block expression with no final expression")
    if isinstance(expr, IdentifierIR):
        if expr.defid == wrt_defid:
            return LiteralIR(1, loc)
        # Chain rule: identifier is a let-bound variable; differentiate its defining expr w.r.t. wrt_defid
        if binding_by_defid is not None and expr.defid is not None:
            binding = binding_by_defid.get(expr.defid)
            if binding is not None and binding.expr is not None:
                return _diff_expr_wrt(binding.expr, wrt_defid, loc, binding_by_defid, resolver)
        return LiteralIR(0, loc)
    if isinstance(expr, LiteralIR):
        return LiteralIR(0, loc)
    if isinstance(expr, BinaryOpIR):
        left, right = expr.left, expr.right
        d_left = _diff_expr_wrt(left, wrt_defid, loc, binding_by_defid, resolver)
        d_right = _diff_expr_wrt(right, wrt_defid, loc, binding_by_defid, resolver)
        op = expr.operator
        if op == BinaryOp.ADD:
            return BinaryOpIR(BinaryOp.ADD, d_left, d_right, loc)
        if op == BinaryOp.SUB:
            return BinaryOpIR(BinaryOp.SUB, d_left, d_right, loc)
        if op == BinaryOp.MUL:
            return BinaryOpIR(
                BinaryOp.ADD,
                BinaryOpIR(BinaryOp.MUL, d_left, right, loc),
                BinaryOpIR(BinaryOp.MUL, left, d_right, loc),
                loc,
            )
        if op == BinaryOp.DIV:
            num = BinaryOpIR(BinaryOp.SUB, BinaryOpIR(BinaryOp.MUL, d_left, right, loc), BinaryOpIR(BinaryOp.MUL, left, d_right, loc), loc)
            den = BinaryOpIR(BinaryOp.POW, right, LiteralIR(2, loc), loc)
            return BinaryOpIR(BinaryOp.DIV, num, den, loc)
        if op == BinaryOp.POW and isinstance(right, LiteralIR):
            c = right.value
            if isinstance(c, (int, float)) and c != 0:
                bm1 = LiteralIR(c - 1, loc)
                a_bm1 = BinaryOpIR(BinaryOp.POW, left, bm1, loc)
                return BinaryOpIR(BinaryOp.MUL, BinaryOpIR(BinaryOp.MUL, LiteralIR(c, loc), a_bm1, loc), d_left, loc)
        raise ValueError(f"Autodiff: unsupported binary operator in ∂expr/∂wrt: {op}")
    if isinstance(expr, UnaryOpIR):
        d_operand = _diff_expr_wrt(expr.operand, wrt_defid, loc, binding_by_defid, resolver)
        if expr.operator == UnaryOp.NEG:
            return UnaryOpIR(UnaryOp.NEG, d_operand, loc)
        if expr.operator == UnaryOp.POS:
            return d_operand
        raise ValueError(f"Autodiff: unsupported unary operator in ∂expr/∂wrt: {expr.operator}")
    if isinstance(expr, FunctionCallIR):
        callee_defid = expr.function_defid
        args = expr.arguments or []
        if binding_by_defid is None:
            raise ValueError(
                "Autodiff: cannot differentiate function call without binding context (binding_by_defid)"
            )
        if callee_defid is None or callee_defid not in binding_by_defid:
            raise ValueError(
                "Autodiff: function call callee is not a known user function (identifier or not in program)"
            )
        binding = binding_by_defid[callee_defid]
        rule_body = getattr(binding.expr, 'custom_diff_body', None) if isinstance(binding.expr, FunctionValueIR) else None
        if rule_body is not None:
            fv = binding.expr
            params = fv.parameters or [] if isinstance(fv, FunctionValueIR) else []
            if len(params) == len(args):
                replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
                terms = []
                for i, param in enumerate(params):
                    if param.defid is None:
                        continue
                    diff_replace_i = {params[j].defid: (LiteralIR(1, loc) if j == i else LiteralIR(0, loc)) for j in range(len(params)) if params[j].defid is not None}
                    coef_i = _substitute_expr_with_diffs(rule_body, replace_map, diff_replace_i, loc)
                    d_arg = _diff_expr_wrt(args[i], wrt_defid, loc, binding_by_defid, resolver)
                    terms.append(BinaryOpIR(BinaryOp.MUL, coef_i, d_arg, loc))
                if terms:
                    out = terms[0]
                    for t in terms[1:]:
                        out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
                    return out
        if not isinstance(binding.expr, FunctionValueIR):
            raise ValueError(
                "Autodiff: function call callee is not a user function value (e.g. builtin or unknown)"
            )
        fv = binding.expr
        body = fv.body
        params = fv.parameters or []
        if body is None:
            raise ValueError("Autodiff: user function has no body")
        if len(params) != len(args):
            raise ValueError(
                f"Autodiff: function call arity mismatch: {len(args)} arguments vs {len(params)} parameters"
            )
        replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
        terms = []
        for i, param in enumerate(params):
            if param.defid is None:
                raise ValueError("Autodiff: function parameter has no defid")
            partial = _diff_expr_wrt(body, param.defid, loc, binding_by_defid, resolver)
            partial_subst = _substitute_expr(partial, replace_map, loc)
            d_arg = _diff_expr_wrt(args[i], wrt_defid, loc, binding_by_defid, resolver)
            terms.append(BinaryOpIR(BinaryOp.MUL, partial_subst, d_arg, loc))
        if not terms:
            return LiteralIR(0, loc)
        out = terms[0]
        for t in terms[1:]:
            out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
        return out
    if isinstance(expr, EinsteinIR):
        return _diff_einstein_wrt(expr, wrt_defid, loc, binding_by_defid, resolver)
    raise ValueError(f"Autodiff: cannot differentiate expression type in ∂expr/∂wrt: {type(expr).__name__}")


def _build_differential_expr(
    expr: ExpressionIR,
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    loc: SourceLocation,
) -> ExpressionIR:
    """Build IR for d(expr): the differential of expr in terms of d_x refs. Raises for unsupported expr."""
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        ref = defid_to_d_ident.get(expr.defid)
        if ref is not None:
            return IdentifierIR(ref.name, expr.location, ref.defid)
        raise ValueError(
            f"Autodiff: @(expr) operand identifier not in differential targets (defid not in d_* map)"
        )
    if isinstance(expr, LiteralIR):
        return LiteralIR(0, loc)
    if isinstance(expr, BinaryOpIR):
        d_left = _build_differential_expr(expr.left, defid_to_d_ident, loc)
        d_right = _build_differential_expr(expr.right, defid_to_d_ident, loc)
        return _forward_binary(expr, d_left, d_right, expr.location or loc)
    if isinstance(expr, UnaryOpIR):
        d_operand = _build_differential_expr(expr.operand, defid_to_d_ident, loc)
        return _forward_unary(expr, d_operand, expr.location or loc)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _build_differential_expr(expr.final_expr, defid_to_d_ident, loc)
    raise ValueError(f"Autodiff: unsupported expression type for d(expr): {type(expr).__name__}")


def _substitute_expr(expr: ExpressionIR, replace_map: Dict[DefId, ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    if isinstance(expr, IdentifierIR) and expr.defid is not None and expr.defid in replace_map:
        return replace_map[expr.defid]
    if isinstance(expr, LiteralIR):
        return LiteralIR(expr.value, loc)
    if isinstance(expr, BinaryOpIR):
        return BinaryOpIR(expr.operator, _substitute_expr(expr.left, replace_map, loc), _substitute_expr(expr.right, replace_map, loc), loc)
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(expr.operator, _substitute_expr(expr.operand, replace_map, loc), loc)
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _substitute_expr(expr.final_expr, replace_map, loc)
    return expr


def _substitute_expr_with_diffs(
    expr: ExpressionIR,
    replace_map: Dict[DefId, ExpressionIR],
    diff_replace_map: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
) -> ExpressionIR:
    """Substitute identifiers by replace_map and DifferentialIR(IdentifierIR(defid)) by diff_replace_map[defid]. Used for @fn rule bodies."""
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        if expr.defid in replace_map:
            return replace_map[expr.defid]
    if isinstance(expr, DifferentialIR):
        op = expr.operand
        if isinstance(op, IdentifierIR) and op.defid is not None and op.defid in diff_replace_map:
            return diff_replace_map[op.defid]
        return DifferentialIR(operand=_substitute_expr_with_diffs(op, replace_map, diff_replace_map, loc), location=loc)
    if isinstance(expr, LiteralIR):
        return LiteralIR(expr.value, loc)
    if isinstance(expr, BinaryOpIR):
        return BinaryOpIR(
            expr.operator,
            _substitute_expr_with_diffs(expr.left, replace_map, diff_replace_map, loc),
            _substitute_expr_with_diffs(expr.right, replace_map, diff_replace_map, loc),
            loc,
        )
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _substitute_expr_with_diffs(expr.operand, replace_map, diff_replace_map, loc),
            loc,
        )
    if isinstance(expr, BlockExpressionIR) and expr.final_expr is not None:
        return _substitute_expr_with_diffs(expr.final_expr, replace_map, diff_replace_map, loc)
    if isinstance(expr, FunctionCallIR):
        new_args = [_substitute_expr_with_diffs(a, replace_map, diff_replace_map, loc) for a in (expr.arguments or [])]
        return FunctionCallIR(
            callee_expr=expr.callee_expr,
            location=expr.location,
            arguments=new_args,
            module_path=expr.module_path,
            type_info=expr.type_info,
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, RectangularAccessIR):
        arr = _substitute_expr_with_diffs(expr.array, replace_map, diff_replace_map, loc)
        indices = [_substitute_expr_with_diffs(i, replace_map, diff_replace_map, loc) for i in (expr.indices or [])]
        return RectangularAccessIR(array=arr, indices=indices, location=expr.location)
    return expr


def _forward_einstein_ir(
    expr: EinsteinIR,
    d_ref_by_defid: Dict[DefId, IdentifierIR],
    binding_by_defid: Dict[DefId, BindingIR],
    loc: SourceLocation,
) -> Optional[ExpressionIR]:
    """Forward diff for high-level EinsteinIR: sum-of-products body -> d_C as EinsteinIR."""
    clause_terms: List[ExpressionIR] = []
    for clause in expr.clauses or []:
        val = clause.value
        if not isinstance(val, ReductionExpressionIR) or (val.operation or "").lower() != "sum":
            continue
        inner = val.body
        factors = _flatten_product(inner) if inner else None
        if not factors or len(factors) < 1:
            continue
        loop_vars = list(val.loop_vars or [])
        reduction_parts: List[ExpressionIR] = []
        for arr_expr, index_list in factors:
            if not isinstance(arr_expr, IdentifierIR) or arr_expr.defid is None:
                continue
            if arr_expr.defid not in d_ref_by_defid:
                continue
            d_ref = d_ref_by_defid[arr_expr.defid]
            prod: ExpressionIR = RectangularAccessIR(d_ref, index_list, loc)
            for other_expr, other_indices in factors:
                if other_expr is arr_expr:
                    continue
                if not isinstance(other_expr, IdentifierIR) or other_expr.defid is None:
                    continue
                b = binding_by_defid.get(other_expr.defid)
                nm = other_expr.name or (b.name if b else "") or ""
                other_ref = IdentifierIR(nm, loc, other_expr.defid)
                prod = BinaryOpIR(BinaryOp.MUL, prod, RectangularAccessIR(other_ref, other_indices, loc), loc)
            red = ReductionExpressionIR(
                operation=val.operation,
                loop_vars=loop_vars,
                body=prod,
                location=loc,
                where_clause=val.where_clause,
                loop_var_ranges=dict(val.loop_var_ranges or {}),
                type_info=val.type_info,
                shape_info=val.shape_info,
            )
            reduction_parts.append(red)
        if not reduction_parts:
            continue
        combined_val: ExpressionIR = reduction_parts[0]
        for rp in reduction_parts[1:]:
            combined_val = BinaryOpIR(BinaryOp.ADD, combined_val, rp, loc)
        new_clause = EinsteinClauseIR(
            indices=list(clause.indices or []),
            value=combined_val,
            location=clause.location,
            where_clause=clause.where_clause,
            variable_ranges=dict(clause.variable_ranges or {}),
        )
        clause_terms.append(
            EinsteinIR(
                clauses=[new_clause],
                shape=expr.shape,
                element_type=expr.element_type,
                location=expr.location,
                type_info=expr.type_info,
                shape_info=expr.shape_info,
            )
        )
    if not clause_terms:
        return None
    out = clause_terms[0]
    for t in clause_terms[1:]:
        out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
    return out


def _forward_d_y_expr(
    binding: BindingIR,
    defid_to_d_ref: Dict[DefId, ExpressionIR],
    binding_by_defid: Dict[DefId, BindingIR],
    binding_to_deps: Dict[DefId, Set[DefId]],
    loc: SourceLocation,
    resolver: Optional[Any] = None,
) -> Optional[ExpressionIR]:
    """Forward: expression for d_y from d_x1, d_x2 (deps). Returns None if no deps or unsupported."""
    expr = binding.expr
    if expr is None:
        return None
    deps = binding_to_deps.get(binding.defid) or set()
    if not deps:
        return None
    if isinstance(expr, BinaryOpIR):
        left_defid = _defid_from_expr(expr.left)
        right_defid = _defid_from_expr(expr.right)
        d_left = defid_to_d_ref.get(left_defid) if left_defid else LiteralIR(0, loc)
        d_right = defid_to_d_ref.get(right_defid) if right_defid else LiteralIR(0, loc)
        if not isinstance(d_left, ExpressionIR):
            d_left = LiteralIR(0, loc)
        if not isinstance(d_right, ExpressionIR):
            d_right = LiteralIR(0, loc)
        return _forward_binary(expr, d_left, d_right, expr.location)
    if isinstance(expr, UnaryOpIR):
        operand_defid = _defid_from_expr(expr.operand)
        d_operand = defid_to_d_ref.get(operand_defid) if operand_defid else LiteralIR(0, loc)
        if not isinstance(d_operand, ExpressionIR):
            d_operand = LiteralIR(0, loc)
        return _forward_unary(expr, d_operand, expr.location)
    if isinstance(expr, BuiltinCallIR):
        return None
    if isinstance(expr, FunctionCallIR):
        callee_defid = expr.function_defid
        args = expr.arguments or []
        if callee_defid is None or len(args) == 0:
            return None
        callee_binding = binding_by_defid.get(callee_defid)
        rule_body = getattr(callee_binding.expr, 'custom_diff_body', None) if callee_binding is not None and isinstance(callee_binding.expr, FunctionValueIR) else None
        if rule_body is not None and callee_binding is not None:
            fv = callee_binding.expr
            params = fv.parameters or [] if isinstance(fv, FunctionValueIR) else []
            if len(params) == len(args):
                replace_map = {p.defid: args[j] for j, p in enumerate(params) if p.defid is not None}
                diff_replace_map = {}
                for i, param in enumerate(params):
                    if param.defid is None:
                        continue
                    arg_defid = _defid_from_expr(args[i])
                    d_arg = defid_to_d_ref.get(arg_defid) if arg_defid else None
                    diff_replace_map[param.defid] = d_arg if isinstance(d_arg, ExpressionIR) else LiteralIR(0, loc)
                d_y = _substitute_expr_with_diffs(rule_body, replace_map, diff_replace_map, expr.location or loc)
                return d_y
        if callee_binding is None or not isinstance(callee_binding.expr, FunctionValueIR):
            return None
        fv = callee_binding.expr
        params = fv.parameters or []
        body = fv.body
        if body is None or len(params) != len(args):
            return None
        replace_map = {p.defid: args[j] for j, p in enumerate(params)}
        terms = []
        for i, param in enumerate(params):
            partial = _diff_expr_wrt(body, param.defid, expr.location, binding_by_defid, resolver)
            partial_at_call = _substitute_expr(partial, replace_map, expr.location)
            arg_defid = _defid_from_expr(args[i])
            if arg_defid is None:
                continue
            d_arg = defid_to_d_ref.get(arg_defid)
            if d_arg is None:
                continue
            terms.append(BinaryOpIR(BinaryOp.MUL, d_arg, partial_at_call, expr.location))
        if not terms:
            return None
        out = terms[0]
        for t in terms[1:]:
            out = BinaryOpIR(BinaryOp.ADD, out, t, loc)
        return out
    if isinstance(expr, EinsteinIR):
        d_ref_map = {did: ref for did, ref in defid_to_d_ref.items() if isinstance(ref, IdentifierIR)}
        return _forward_einstein_ir(expr, d_ref_map, binding_by_defid, loc)
    return None


# ---------------------------------------------------------------------------
# Type info propagation (validation requires all nodes to have type_info)
# ---------------------------------------------------------------------------

def _set_type_info_on_expr(expr: Any, type_info: Any, shape_info: Any) -> None:
    """Set type_info and shape_info on expr and all descendant expressions (in-place). Only on nodes that have these attributes."""
    if expr is None:
        return
    if hasattr(expr, "type_info") and getattr(expr, "type_info", None) is None and type_info is not None:
        expr.type_info = type_info
    if hasattr(expr, "shape_info") and getattr(expr, "shape_info", None) is None and shape_info is not None:
        expr.shape_info = shape_info
    if isinstance(expr, BinaryOpIR):
        _set_type_info_on_expr(expr.left, type_info, shape_info)
        _set_type_info_on_expr(expr.right, type_info, shape_info)
    elif isinstance(expr, UnaryOpIR):
        _set_type_info_on_expr(expr.operand, type_info, shape_info)
    elif isinstance(expr, BlockExpressionIR):
        for s in expr.statements or []:
            if isinstance(s, ExpressionIR):
                _set_type_info_on_expr(s, type_info, shape_info)
        _set_type_info_on_expr(expr.final_expr, type_info, shape_info)
    elif isinstance(expr, EinsteinIR):
        for c in expr.clauses or []:
            for idx in getattr(c, "indices", None) or []:
                _set_type_info_on_expr(idx, type_info, shape_info)
            if getattr(c, "value", None) is not None:
                _set_type_info_on_expr(c.value, type_info, shape_info)
    elif isinstance(expr, ReductionExpressionIR):
        _set_type_info_on_expr(expr.body, type_info, shape_info)
        if getattr(expr, "where_clause", None) is not None:
            _set_type_info_on_expr(expr.where_clause, type_info, shape_info)
    elif isinstance(expr, RectangularAccessIR):
        _set_type_info_on_expr(expr.array, type_info, shape_info)
        for idx in expr.indices or []:
            if isinstance(idx, ExpressionIR):
                _set_type_info_on_expr(idx, type_info, shape_info)
    elif isinstance(expr, WhereClauseIR):
        for c in getattr(expr, "constraints", None) or []:
            _set_type_info_on_expr(c, type_info, shape_info)
    # LiteralIR, IdentifierIR, IndexVarIR, etc. already handled by the attr set above


# ---------------------------------------------------------------------------
# Quotient collection and expansion
# ---------------------------------------------------------------------------

def _defid_from_differential_or_id(expr: Any) -> Optional[DefId]:
    """Resolve DefId from DifferentialIR(IdentifierIR) or IdentifierIR."""
    if expr is None:
        return None
    if isinstance(expr, DifferentialIR):
        return _defid_from_expr(expr.operand)
    if isinstance(expr, IdentifierIR) and expr.defid is not None:
        return expr.defid
    return None


def _collect_quotient_pairs(program: ProgramIR) -> List[Tuple[DefId, DefId]]:
    """Collect (numerator_defid, denominator_defid) from BinaryOpIR(DIV, @num, @den)."""
    pairs: List[Tuple[DefId, DefId]] = []

    def walk(e: Any) -> None:
        if e is None:
            return
        if isinstance(e, BinaryOpIR):
            if e.operator == BinaryOp.DIV:
                num_defid = _defid_from_differential_or_id(e.left)
                den_defid = _defid_from_differential_or_id(e.right)
                if num_defid is not None and den_defid is not None:
                    pairs.append((num_defid, den_defid))
            walk(e.left)
            walk(e.right)
        if isinstance(e, UnaryOpIR):
            walk(e.operand)
        if isinstance(e, BlockExpressionIR):
            for s in e.statements or []:
                walk(s)
            walk(e.final_expr)
        if isinstance(e, BindingIR):
            walk(e.expr)
        if isinstance(e, ProgramIR):
            for b in e.bindings or []:
                walk(b)

    walk(program)
    return pairs


def _expand_derivative_in_expr(
    expr: ExpressionIR,
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    quotient_to_expr: Dict[Tuple[DefId, DefId], ExpressionIR],
    defid_to_expr: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
    binding_by_defid: Optional[Dict[DefId, Any]] = None,
    resolver: Optional[Any] = None,
) -> ExpressionIR:
    """Rewrite DifferentialIR -> d_ref or d(expr); expand every @xxx/@y to ∂(xxx)/∂y."""
    if isinstance(expr, DifferentialIR):
        operand = expr.operand
        t = _differential_target_from_operand(operand)
        if t is not None:
            ref = defid_to_d_ident.get(t[0])
            if ref is not None:
                return IdentifierIR(
                    ref.name, expr.location, ref.defid,
                    type_info=getattr(expr, "type_info", None),
                    shape_info=getattr(expr, "shape_info", None),
                )
        # @(expr) for compound expr: d(expr) = chain rule in terms of d_x refs
        out = _build_differential_expr(operand, defid_to_d_ident, loc)
        if getattr(expr, "type_info", None) is not None and hasattr(out, "type_info"):
            out.type_info = expr.type_info
        if getattr(expr, "shape_info", None) is not None and hasattr(out, "shape_info"):
            out.shape_info = expr.shape_info
        return out
    if isinstance(expr, BinaryOpIR):
        # Expand @xxx/@y to ∂(xxx)/∂y; raise if denominator is not a simple identifier or we cannot differentiate.
        if expr.operator == BinaryOp.DIV and isinstance(expr.left, DifferentialIR) and isinstance(expr.right, DifferentialIR):
            den_operand = expr.right.operand
            if not isinstance(den_operand, IdentifierIR) or den_operand.defid is None:
                raise ValueError(
                    "Autodiff: @num/@den quotient requires denominator to be a simple identifier (e.g. @x)"
                )
            den_defid = den_operand.defid
            num_operand = expr.left.operand
            if isinstance(num_operand, IdentifierIR) and num_operand.defid is not None:
                num_expr = defid_to_expr.get(num_operand.defid)
                if num_expr is None:
                    raise ValueError(
                        "Autodiff: numerator of @num/@den has no defining expression in this program (e.g. from function call)"
                    )
            else:
                num_expr = num_operand
            der = _diff_expr_wrt(num_expr, den_defid, expr.location or loc, binding_by_defid, resolver)
            if isinstance(der, LiteralIR) and der.value == 0 and not isinstance(num_expr, LiteralIR):
                raise ValueError(
                    "Autodiff: ∂(numerator)/∂(denominator) is not supported for this expression type (e.g. Einstein/sum)"
                )
            ti = getattr(expr, "type_info", None)
            si = getattr(expr, "shape_info", None)
            if ti is not None or si is not None:
                _set_type_info_on_expr(der, ti, si)
            return der
        new_left = _expand_derivative_in_expr(expr.left, defid_to_d_ident, quotient_to_expr, defid_to_expr, loc, binding_by_defid, resolver)
        new_right = _expand_derivative_in_expr(expr.right, defid_to_d_ident, quotient_to_expr, defid_to_expr, loc, binding_by_defid, resolver)
        return BinaryOpIR(
            expr.operator,
            new_left,
            new_right,
            expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, UnaryOpIR):
        return UnaryOpIR(
            expr.operator,
            _expand_derivative_in_expr(expr.operand, defid_to_d_ident, quotient_to_expr, defid_to_expr, loc, binding_by_defid, resolver),
            expr.location,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    if isinstance(expr, BlockExpressionIR):
        new_stmts = [_expand_derivative_in_expr(s, defid_to_d_ident, quotient_to_expr, defid_to_expr, loc, binding_by_defid, resolver) if isinstance(s, ExpressionIR) else s for s in (expr.statements or [])]
        new_final = _expand_derivative_in_expr(expr.final_expr, defid_to_d_ident, quotient_to_expr, defid_to_expr, loc, binding_by_defid, resolver) if expr.final_expr is not None else None
        return BlockExpressionIR(
            statements=new_stmts,
            location=expr.location,
            final_expr=new_final,
            type_info=getattr(expr, "type_info", None),
            shape_info=getattr(expr, "shape_info", None),
        )
    return expr


def _expand_derivative_nodes_in_program(
    program: ProgramIR,
    defid_to_d_ident: Dict[DefId, IdentifierIR],
    quotient_to_expr: Dict[Tuple[DefId, DefId], ExpressionIR],
    defid_to_expr: Dict[DefId, ExpressionIR],
    binding_by_defid: Optional[Dict[DefId, Any]] = None,
    resolver: Optional[Any] = None,
) -> None:
    """In-place: replace DifferentialIR and DIV(@.,@.) in program with d_ refs / d(expr) / derivative exprs."""
    loc = SourceLocation("", 0, 0)
    for binding in program.bindings or []:
        if not isinstance(binding, BindingIR) or binding.expr is None:
            continue
        binding.expr = _expand_derivative_in_expr(
            binding.expr, defid_to_d_ident, quotient_to_expr, defid_to_expr,
            binding.expr.location or loc, binding_by_defid, resolver,
        )


# ---------------------------------------------------------------------------
# AutodiffPass: forward diff only (dx -> dy)
# ---------------------------------------------------------------------------

class AutodiffPass(BasePass):
    """Expand @ and @y/@x into plain IR via forward diff. @(expr) -> d(expr); @(expr)/@x -> derivative."""

    requires = [
        TypeInferencePass,
        UnifiedShapeAnalysisPass,
    ]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        program = ir
        bindings = _bindings_in_block(program, program) or []
        if not bindings:
            tcx.set_analysis(AutodiffPass, {"diff_block": None, "differential_targets": set(), "differential_buffer_by_defid": {}})
            return program

        collector = _DifferentialCollector()
        for b in bindings:
            if b.expr is not None:
                b.expr.accept(collector)
        differential_targets = list(collector.targets)
        quotient_pairs = _collect_quotient_pairs(program)

        binding_by_defid: Dict[DefId, BindingIR] = {}
        for b in bindings:
            if b.defid is not None:
                binding_by_defid[b.defid] = b

        binding_to_deps: Dict[DefId, Set[DefId]] = {}
        for b in bindings:
            if b.defid is None:
                continue
            binding_to_deps[b.defid] = _collect_defids_in_expr(b.expr)

        target_defids: Set[DefId] = set()
        for did, _ in differential_targets:
            target_defids.add(did)
        for num, den in quotient_pairs:
            target_defids.add(num)
            target_defids.add(den)

        reachable: Set[DefId] = set()
        work = list(target_defids)
        while work:
            did = work.pop()
            if did in reachable:
                continue
            reachable.add(did)
            b = binding_by_defid.get(did)
            if b is None:
                continue
            for dep in binding_to_deps.get(b.defid) or []:
                if dep not in reachable:
                    work.append(dep)

        forward_order: List[BindingIR] = []
        seen: Set[DefId] = set()

        def visit(did: DefId) -> None:
            if did in seen or did not in reachable:
                return
            seen.add(did)
            b = binding_by_defid.get(did)
            if b is None:
                return
            for dep in binding_to_deps.get(b.defid) or []:
                visit(dep)
            forward_order.append(b)

        for did in target_defids:
            visit(did)
        # forward_order is deps-first (e.g. a then b for b=a*a); do not reverse or d_* uses zeros for deps.

        resolver = getattr(tcx, "resolver", None)
        if resolver is None:
            tcx.set_analysis(AutodiffPass, {"diff_block": None, "differential_targets": set(differential_targets), "differential_buffer_by_defid": {}})
            return program

        defid_to_d_ident: Dict[DefId, IdentifierIR] = {}
        defid_to_d_binding: Dict[DefId, BindingIR] = {}
        seed_value: Dict[DefId, int] = {}
        quotient_denominators = {den for _, den in quotient_pairs}
        leaves = {did for did in reachable if not (binding_to_deps.get(did) or set())}

        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            d_name = "d_" + (b.name or "")
            d_defid = resolver.allocate_for_local()
            d_ref = IdentifierIR(d_name, b.location or SourceLocation("", 0, 0), d_defid)
            defid_to_d_ident[b.defid] = d_ref
            # Seed values only for single-run fallback; per-quotient runs use backend-set env (AUTODIFF_ALGORITHM §4.2).
            if b.defid in quotient_denominators:
                seed_value[b.defid] = 1
            elif b.defid in leaves and b.defid in target_defids:
                seed_value[b.defid] = 1
            else:
                seed_value[b.defid] = 0

        defid_to_d_ref_expr: Dict[DefId, ExpressionIR] = {}
        for did, ref in defid_to_d_ident.items():
            defid_to_d_ref_expr[did] = ref

        # When there are quotient pairs, leaf d_* get RHS 0 and backend sets env per quotient (AUTODIFF_ALGORITHM §4.2).
        # Otherwise use seed_value so a single run gives total differential / single partial.
        d_rhs_by_defid: Dict[DefId, ExpressionIR] = {}
        loc0 = SourceLocation("", 0, 0)
        use_per_quotient_seeds = len(quotient_pairs) > 0
        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            bloc = b.location or loc0
            if use_per_quotient_seeds and b.defid in leaves:
                d_rhs_by_defid[b.defid] = LiteralIR(0, bloc)
            elif b.defid in seed_value and seed_value[b.defid] == 1:
                d_rhs_by_defid[b.defid] = LiteralIR(1, bloc)
            else:
                for dep in binding_to_deps.get(b.defid) or []:
                    if dep not in defid_to_d_ref_expr:
                        defid_to_d_ref_expr[dep] = LiteralIR(0, bloc)
                rhs = _forward_d_y_expr(b, defid_to_d_ref_expr, binding_by_defid, binding_to_deps, bloc, resolver)
                if rhs is not None:
                    d_rhs_by_defid[b.defid] = rhs
                else:
                    d_rhs_by_defid[b.defid] = LiteralIR(0, bloc)

        type_info = None
        shape_info = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                type_info = getattr(b, "type_info", None) or getattr(b.expr, "type_info", None)
                shape_info = getattr(b, "shape_info", None) or getattr(b.expr, "shape_info", None)
                if type_info is not None:
                    break
        for b in forward_order:
            if b.defid is None or b.defid not in reachable:
                continue
            rhs = d_rhs_by_defid.get(b.defid) or LiteralIR(0, b.location or SourceLocation("", 0, 0))
            ti = getattr(b, "type_info", None) or (
                getattr(b.expr, "type_info", None) if b.expr is not None else None
            ) or type_info
            si = getattr(b, "shape_info", None) or (
                getattr(b.expr, "shape_info", None) if b.expr is not None else None
            ) or shape_info
            _set_type_info_on_expr(rhs, ti, si)
            d_ref = defid_to_d_ident[b.defid]
            d_binding = BindingIR(
                name=d_ref.name,
                expr=rhs,
                location=b.location,
                defid=d_ref.defid,
                type_info=ti,
            )
            defid_to_d_binding[b.defid] = d_binding

        new_bindings: List[BindingIR] = []
        for b in bindings:
            new_bindings.append(b)
            d_b = defid_to_d_binding.get(b.defid)
            if d_b is not None:
                new_bindings.append(d_b)

        program.bindings = new_bindings
        # Preserve non-binding statements (e.g. assert) so they still run
        non_binding_stmts = [s for s in (program.statements or []) if not isinstance(s, BindingIR)]
        program.statements = new_bindings + non_binding_stmts

        quotient_to_expr = {}
        defid_to_expr = {
            b.defid: b.expr
            for b in (program.bindings or [])
            if isinstance(b, BindingIR) and getattr(b, "defid", None) is not None and b.expr is not None
        }
        _expand_derivative_nodes_in_program(program, defid_to_d_ident, quotient_to_expr, defid_to_expr, binding_by_defid, resolver)

        # Diff block: d_* bindings in forward order so backend can run them with per-quotient seeds.
        diff_block_list: List[BindingIR] = [
            defid_to_d_binding[b.defid]
            for b in forward_order
            if b.defid in defid_to_d_binding
        ]
        diff_block: Optional[List[BindingIR]] = diff_block_list if diff_block_list else None
        autodiff_differential_map: Dict[DefId, DefId] = {
            primal: d_binding.defid
            for primal, d_binding in defid_to_d_binding.items()
        }
        differential_leaves: Set[DefId] = leaves

        # Only nodes we created (d_* RHS and d_* bindings) are typed above; we do not
        # fill type_info on existing program nodes (see AUTODIFF_PIPELINE.md §2 and §5).

        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": diff_block,
                "differential_targets": set(differential_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": autodiff_differential_map,
                "differential_leaves": differential_leaves,
            },
        )
        # Clear custom_diff_body from all functions; no longer needed after this pass.
        for b in (program.functions or []):
            if isinstance(b.expr, FunctionValueIR) and getattr(b.expr, 'custom_diff_body', None) is not None:
                object.__setattr__(b.expr, 'custom_diff_body', None)
        return program

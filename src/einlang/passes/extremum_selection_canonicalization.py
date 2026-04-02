"""
Canonicalize generic extremum-selection patterns into SelectAtArgmaxIR.

This pass lifts backend-only argmax/argmin body matching into IR by rewriting
generic patterns such as:

- ``let max_val = max[i](body); let indices = [payload | i in R, body == max_val]; indices[0]``
- ``let weighted[..outer, i] = if body == max_val[..outer] { payload } else { sentinel };
   let result[..outer] = min[i](weighted[..outer, i]); result``

into ``SelectAtArgmaxIR`` / ``SelectAtArgminIR`` so later lowering and the
generic NumPy execution path can handle them uniformly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..ir.nodes import (
    ArrayComprehensionIR,
    BindingIR,
    BinaryOpIR,
    BlockExpressionIR,
    EinsteinIR,
    ExpressionIR,
    FunctionValueIR,
    IdentifierIR,
    IfExpressionIR,
    IndexRestIR,
    IndexVarIR,
    IRNode,
    LiteralIR,
    ModuleIR,
    ProgramIR,
    RectangularAccessIR,
    ReductionExpressionIR,
    SelectAtArgmaxIR,
    is_function_binding,
)
from ..passes.base import BasePass, TyCtxt
from ..shared.defid import DefId
from ..shared.types import BinaryOp, ReductionOp


@dataclass
class _RewriteMatch:
    expr: ExpressionIR
    helper_defids: Set[DefId]
    kind: str


@dataclass
class _ScopeEnv:
    by_defid: Dict[DefId, BindingIR]
    by_name: Dict[str, BindingIR]

    @classmethod
    def empty(cls) -> "_ScopeEnv":
        return cls(by_defid={}, by_name={})

    def fork(self) -> "_ScopeEnv":
        return _ScopeEnv(dict(self.by_defid), dict(self.by_name))

    def add(self, binding: BindingIR) -> None:
        if binding.defid is not None:
            self.by_defid[binding.defid] = binding
        if binding.name:
            self.by_name[binding.name] = binding

    def lookup_identifier(self, expr: ExpressionIR) -> Optional[BindingIR]:
        if not isinstance(expr, IdentifierIR):
            return None
        if expr.defid is not None and expr.defid in self.by_defid:
            return self.by_defid[expr.defid]
        if expr.name:
            return self.by_name.get(expr.name)
        return None


class ExtremumSelectionCanonicalizationPass(BasePass):
    """Rewrite generic extremum-selection patterns to SelectAtArgmaxIR."""

    requires: List[type[BasePass]] = []

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        rewriter = _ExtremumSelectionRewriter()
        rewriter.rewrite_program(ir)
        function_ir_map = getattr(tcx, "function_ir_map", None) or {}
        for binding in function_ir_map.values():
            if isinstance(binding, BindingIR):
                rewriter.rewrite_binding(binding)
        tcx.set_analysis(
            ExtremumSelectionCanonicalizationPass,
            {
                "implemented": True,
                "rewritten_count": rewriter.rewritten_count,
                "direct_match_count": rewriter.direct_match_count,
                "weighted_match_count": rewriter.weighted_match_count,
            },
        )
        return ir


class _ExtremumSelectionRewriter:
    def __init__(self) -> None:
        self.rewritten_count = 0
        self.direct_match_count = 0
        self.weighted_match_count = 0

    def rewrite_program(self, program: ProgramIR) -> None:
        statements, _ = self._rewrite_statement_list(list(program.statements or []), None, _ScopeEnv.empty())
        program.statements = statements
        program.bindings = [stmt for stmt in statements if isinstance(stmt, BindingIR)]
        for mod in program.modules or []:
            self._rewrite_module(mod)

    def rewrite_binding(self, binding: BindingIR) -> None:
        rewritten, _ = self._rewrite_statement(binding, _ScopeEnv.empty())
        if rewritten is not binding:
            raise AssertionError("binding rewrite should mutate in place")

    def _rewrite_module(self, module: ModuleIR) -> None:
        functions, _ = self._rewrite_statement_list(list(module.functions or []), None, _ScopeEnv.empty())
        constants, _ = self._rewrite_statement_list(list(module.constants or []), None, _ScopeEnv.empty())
        module.functions = tuple(stmt for stmt in functions if isinstance(stmt, BindingIR))
        module.constants = tuple(stmt for stmt in constants if isinstance(stmt, BindingIR))
        for sub in module.submodules or []:
            self._rewrite_module(sub)

    def _rewrite_statement_list(
        self,
        statements: List[Any],
        tail_expr: Optional[ExpressionIR],
        outer_env: _ScopeEnv,
    ) -> Tuple[List[Any], Optional[ExpressionIR]]:
        env = outer_env.fork()
        rewritten: List[Any] = []
        removable_helper_defids: Set[DefId] = set()
        for stmt in statements:
            new_stmt, helper_defids = self._rewrite_statement(stmt, env)
            rewritten.append(new_stmt)
            removable_helper_defids |= helper_defids
            if isinstance(new_stmt, BindingIR):
                env.add(new_stmt)
        new_tail = tail_expr
        if new_tail is not None:
            new_tail, tail_helpers = self._rewrite_expr(new_tail, env)
            removable_helper_defids |= tail_helpers
        if removable_helper_defids:
            rewritten = self._prune_dead_helper_bindings(rewritten, new_tail, removable_helper_defids)
        return rewritten, new_tail

    def _rewrite_statement(self, stmt: Any, env: _ScopeEnv) -> Tuple[Any, Set[DefId]]:
        if isinstance(stmt, BindingIR):
            expr = stmt.expr
            if isinstance(expr, FunctionValueIR):
                body, _ = self._rewrite_expr(expr.body, _ScopeEnv.empty())
                expr.body = body
                if expr.custom_diff_body is not None:
                    expr.custom_diff_body, _ = self._rewrite_expr(expr.custom_diff_body, _ScopeEnv.empty())
                return stmt, set()
            if expr is not None:
                stmt.expr, helper_defids = self._rewrite_expr(expr, env)
                return stmt, helper_defids
            return stmt, set()
        if isinstance(stmt, ExpressionIR):
            return self._rewrite_expr(stmt, env)
        return stmt, set()

    def _rewrite_expr(self, expr: ExpressionIR, env: _ScopeEnv) -> Tuple[ExpressionIR, Set[DefId]]:
        if expr is None:
            return expr, set()
        if isinstance(expr, BlockExpressionIR):
            block_env = env.fork()
            statements, final_expr = self._rewrite_statement_list(
                list(expr.statements or []),
                expr.final_expr,
                block_env,
            )
            expr.statements = tuple(statements)
            expr.final_expr = final_expr
            return expr, set()

        helper_defids: Set[DefId] = set()
        for slot in _iter_slots(expr):
            value = getattr(expr, slot, None)
            new_value, nested_helpers = self._rewrite_value(value, env)
            if new_value is not value:
                setattr(expr, slot, new_value)
            helper_defids |= nested_helpers

        match = self._try_rewrite_expr(expr, env)
        if match is not None:
            self.rewritten_count += 1
            if match.kind == "direct":
                self.direct_match_count += 1
            elif match.kind == "weighted":
                self.weighted_match_count += 1
            return match.expr, helper_defids | match.helper_defids
        return expr, helper_defids

    def _rewrite_value(self, value: Any, env: _ScopeEnv) -> Tuple[Any, Set[DefId]]:
        if value is None:
            return None, set()
        if isinstance(value, ExpressionIR):
            return self._rewrite_expr(value, env)
        if isinstance(value, list):
            items: List[Any] = []
            helper_defids: Set[DefId] = set()
            changed = False
            for item in value:
                new_item, nested = self._rewrite_value(item, env)
                items.append(new_item)
                helper_defids |= nested
                changed = changed or (new_item is not item)
            return (items if changed else value), helper_defids
        if isinstance(value, tuple):
            items: List[Any] = []
            helper_defids: Set[DefId] = set()
            changed = False
            for item in value:
                new_item, nested = self._rewrite_value(item, env)
                items.append(new_item)
                helper_defids |= nested
                changed = changed or (new_item is not item)
            return (tuple(items) if changed else value), helper_defids
        if isinstance(value, dict):
            changed = False
            helper_defids: Set[DefId] = set()
            rewritten_dict: Dict[Any, Any] = {}
            for key, item in value.items():
                new_key, key_helpers = self._rewrite_value(key, env)
                new_item, item_helpers = self._rewrite_value(item, env)
                helper_defids |= key_helpers
                helper_defids |= item_helpers
                rewritten_dict[new_key] = new_item
                changed = changed or (new_key is not key) or (new_item is not item)
            return (rewritten_dict if changed else value), helper_defids
        if isinstance(value, IRNode):
            helper_defids: Set[DefId] = set()
            for slot in _iter_slots(value):
                child = getattr(value, slot, None)
                new_child, nested = self._rewrite_value(child, env)
                if new_child is not child:
                    setattr(value, slot, new_child)
                helper_defids |= nested
            return value, helper_defids
        return value, set()

    def _try_rewrite_expr(self, expr: ExpressionIR, env: _ScopeEnv) -> Optional[_RewriteMatch]:
        if isinstance(expr, RectangularAccessIR):
            return self._try_rewrite_direct_first_match(expr, env)
        if isinstance(expr, ReductionExpressionIR):
            return self._try_rewrite_weighted_extremum(expr, env)
        return None

    def _try_rewrite_direct_first_match(
        self,
        expr: RectangularAccessIR,
        env: _ScopeEnv,
    ) -> Optional[_RewriteMatch]:
        if len(expr.indices or []) != 1 or not _is_literal_zero(expr.indices[0]):
            return None
        indices_binding = env.lookup_identifier(expr.array)
        if indices_binding is None or not isinstance(indices_binding.expr, ArrayComprehensionIR):
            return None
        comp = indices_binding.expr
        if len(comp.loop_vars or []) != 1 or len(comp.constraints or []) != 1:
            return None

        eq = _split_equality(comp.constraints[0])
        if eq is None:
            return None
        reduction_binding: Optional[BindingIR] = None
        reduction_expr: Optional[ReductionExpressionIR] = None
        primal_expr: Optional[ExpressionIR] = None
        for lhs, rhs in (eq, (eq[1], eq[0])):
            candidate_binding = env.lookup_identifier(rhs)
            if candidate_binding is None or not isinstance(candidate_binding.expr, ReductionExpressionIR):
                continue
            candidate_reduction = candidate_binding.expr
            if candidate_reduction.operation not in (ReductionOp.MAX, ReductionOp.MIN):
                continue
            reduction_binding = candidate_binding
            reduction_expr = candidate_reduction
            primal_expr = lhs
            break
        if reduction_binding is None or reduction_expr is None or primal_expr is None:
            return None
        if len(reduction_expr.loop_vars or []) != 1:
            return None

        comp_loop = comp.loop_vars[0]
        reduction_loop = reduction_expr.loop_vars[0]
        substituted_primal = _clone_with_symbol_subst(
            primal_expr,
            {comp_loop.name: reduction_loop},
        )
        if _expr_signature(substituted_primal) != _expr_signature(reduction_expr.body):
            return None

        diff_body = _clone_with_symbol_subst(
            comp.body,
            {comp_loop.name: reduction_loop},
        )
        helper_defids = {
            did
            for did in (indices_binding.defid, reduction_binding.defid)
            if did is not None
        }
        select = SelectAtArgmaxIR(
            primal_body=_clone_with_symbol_subst(reduction_expr.body, {}),
            diff_body=diff_body,
            loop_vars=list(reduction_expr.loop_vars or []),
            loop_var_ranges=dict(reduction_expr.loop_var_ranges or {}),
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info,
            use_argmin=(reduction_expr.operation == ReductionOp.MIN),
        )
        return _RewriteMatch(expr=select, helper_defids=helper_defids, kind="direct")

    def _try_rewrite_weighted_extremum(
        self,
        expr: ReductionExpressionIR,
        env: _ScopeEnv,
    ) -> Optional[_RewriteMatch]:
        if expr.operation != ReductionOp.MIN or len(expr.loop_vars or []) != 1:
            return None
        weighted_access = _unwrap_trivial_block(expr.body)
        if not isinstance(weighted_access, RectangularAccessIR):
            return None
        weighted_binding = env.lookup_identifier(weighted_access.array)
        if weighted_binding is None or not isinstance(weighted_binding.expr, EinsteinIR):
            return None
        weighted_clause = _single_clause(weighted_binding.expr)
        if weighted_clause is None:
            return None
        weighted_if = _unwrap_trivial_block(weighted_clause.value)
        if not isinstance(weighted_if, IfExpressionIR):
            return None
        if len(weighted_clause.indices or []) != len(weighted_access.indices or []):
            return None

        symbol_subst: Dict[str, ExpressionIR] = {}
        for clause_idx, access_idx in zip(weighted_clause.indices or [], weighted_access.indices or []):
            clause_name = _symbol_name(clause_idx)
            access_name = _symbol_name(access_idx)
            if clause_name is None or access_name is None:
                return None
            symbol_subst[clause_name] = access_idx

        eq = _split_equality(weighted_if.condition)
        if eq is None:
            return None

        extremum_binding: Optional[BindingIR] = None
        extremum_access: Optional[ExpressionIR] = None
        weighted_primal: Optional[ExpressionIR] = None
        for lhs, rhs in (eq, (eq[1], eq[0])):
            candidate_binding = env.lookup_identifier(_access_root_expr(rhs))
            if candidate_binding is None:
                continue
            extracted = _extract_extremum_reduction(candidate_binding.expr)
            if extracted is None:
                continue
            extremum_binding = candidate_binding
            extremum_access = rhs
            weighted_primal = lhs
            break
        if extremum_binding is None or extremum_access is None or weighted_primal is None:
            return None

        extracted = _extract_extremum_reduction(extremum_binding.expr)
        if extracted is None:
            return None
        _, extremum_reduction = extracted
        if extremum_reduction.operation not in (ReductionOp.MAX, ReductionOp.MIN):
            return None

        current_primal = _clone_with_symbol_subst(weighted_primal, symbol_subst)

        extremum_subst = dict(symbol_subst)
        extremum_access_indices = []
        if isinstance(extremum_access, RectangularAccessIR):
            extremum_access_indices = list(extremum_access.indices or [])
        extremum_outer_indices, _ = extracted
        if len(extremum_outer_indices) != len(extremum_access_indices):
            return None
        for clause_idx, access_idx in zip(extremum_outer_indices, extremum_access_indices):
            clause_name = _symbol_name(clause_idx)
            if clause_name is None:
                return None
            extremum_subst[clause_name] = access_idx
        if len(extremum_reduction.loop_vars or []) != len(expr.loop_vars or []):
            return None
        for src_var, dst_var in zip(extremum_reduction.loop_vars or [], expr.loop_vars or []):
            src_name = _symbol_name(src_var)
            if src_name is None:
                return None
            extremum_subst[src_name] = dst_var

        canonical_primal = _clone_with_symbol_subst(extremum_reduction.body, extremum_subst)
        if _expr_signature(current_primal) != _expr_signature(canonical_primal):
            return None

        diff_body = _clone_with_symbol_subst(_unwrap_trivial_block(weighted_if.then_expr), symbol_subst)
        helper_defids = {
            did
            for did in (weighted_binding.defid, extremum_binding.defid)
            if did is not None
        }
        else_root = _access_root_expr(_unwrap_trivial_block(weighted_if.else_expr))
        if isinstance(else_root, IdentifierIR):
            sentinel_binding = env.lookup_identifier(else_root)
            if sentinel_binding is not None and sentinel_binding.defid is not None:
                helper_defids.add(sentinel_binding.defid)

        select = SelectAtArgmaxIR(
            primal_body=current_primal,
            diff_body=diff_body,
            loop_vars=list(expr.loop_vars or []),
            loop_var_ranges=dict(expr.loop_var_ranges or {}),
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info,
            use_argmin=(extremum_reduction.operation == ReductionOp.MIN),
        )
        return _RewriteMatch(expr=select, helper_defids=helper_defids, kind="weighted")

    def _prune_dead_helper_bindings(
        self,
        statements: List[Any],
        tail_expr: Optional[ExpressionIR],
        helper_defids: Set[DefId],
    ) -> List[Any]:
        live: Set[DefId] = _collect_all_defids_ir(tail_expr)
        kept_reversed: List[Any] = []
        for stmt in reversed(statements):
            if isinstance(stmt, BindingIR):
                stmt_defid = stmt.defid
                if stmt_defid in helper_defids and stmt_defid not in live:
                    continue
                if stmt_defid is not None and stmt_defid in live:
                    live.discard(stmt_defid)
                if stmt.expr is not None:
                    live |= _collect_all_defids_ir(stmt.expr)
            else:
                live |= _collect_all_defids_ir(stmt)
            kept_reversed.append(stmt)
        kept_reversed.reverse()
        return kept_reversed


def _iter_slots(node: IRNode) -> Iterable[str]:
    seen: Set[str] = set()
    for cls in type(node).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if slot not in seen:
                seen.add(slot)
                yield slot


def _split_equality(expr: ExpressionIR) -> Optional[Tuple[ExpressionIR, ExpressionIR]]:
    if isinstance(expr, BinaryOpIR) and expr.operator == BinaryOp.EQ:
        return expr.left, expr.right
    return None


def _is_literal_zero(expr: ExpressionIR) -> bool:
    return isinstance(expr, LiteralIR) and isinstance(expr.value, (int, float)) and int(expr.value) == 0


def _symbol_name(expr: Any) -> Optional[str]:
    if isinstance(expr, (IdentifierIR, IndexVarIR, IndexRestIR)):
        return expr.name
    return None


def _single_clause(expr: EinsteinIR) -> Optional[Any]:
    clauses = expr.clauses or ()
    if len(clauses) != 1:
        return None
    return clauses[0]


def _extract_extremum_reduction(expr: ExpressionIR) -> Optional[Tuple[List[ExpressionIR], ReductionExpressionIR]]:
    if isinstance(expr, ReductionExpressionIR):
        return [], expr
    if isinstance(expr, EinsteinIR):
        clause = _single_clause(expr)
        if clause is None or not isinstance(clause.value, ReductionExpressionIR):
            return None
        return list(clause.indices or []), clause.value
    return None


def _unwrap_trivial_block(expr: ExpressionIR) -> ExpressionIR:
    while isinstance(expr, BlockExpressionIR) and not (expr.statements or ()) and expr.final_expr is not None:
        expr = expr.final_expr
    return expr


def _access_root_expr(expr: ExpressionIR) -> ExpressionIR:
    expr = _unwrap_trivial_block(expr)
    if isinstance(expr, RectangularAccessIR):
        return expr.array
    return expr


def _clone_with_symbol_subst(node: Any, subst: Dict[str, ExpressionIR]) -> Any:
    if not subst:
        return copy.deepcopy(node)
    if node is None:
        return None
    if isinstance(node, (str, int, float, bool, range)):
        return copy.deepcopy(node)
    if isinstance(node, (IdentifierIR, IndexVarIR, IndexRestIR)):
        replacement = subst.get(node.name)
        if replacement is not None:
            return copy.deepcopy(replacement)
        return copy.deepcopy(node)
    if isinstance(node, list):
        return [_clone_with_symbol_subst(item, subst) for item in node]
    if isinstance(node, tuple):
        return tuple(_clone_with_symbol_subst(item, subst) for item in node)
    if isinstance(node, dict):
        return {
            _clone_with_symbol_subst(key, subst): _clone_with_symbol_subst(value, subst)
            for key, value in node.items()
        }
    if isinstance(node, IRNode):
        clone = copy.copy(node)
        for slot in _iter_slots(node):
            setattr(clone, slot, _clone_with_symbol_subst(getattr(node, slot, None), subst))
        return clone
    return copy.deepcopy(node)


def _expr_signature(expr: Any) -> Any:
    if expr is None:
        return None
    if isinstance(expr, (str, int, float, bool)):
        return expr
    if isinstance(expr, (IdentifierIR, IndexVarIR, IndexRestIR)):
        return ("sym", expr.name)
    if isinstance(expr, LiteralIR):
        return ("lit", expr.value)
    if isinstance(expr, list):
        return tuple(_expr_signature(item) for item in expr)
    if isinstance(expr, tuple):
        return tuple(_expr_signature(item) for item in expr)
    if isinstance(expr, dict):
        items = sorted(
            ((_expr_signature(key), _expr_signature(value)) for key, value in expr.items()),
            key=repr,
        )
        return ("dict", tuple(items))
    if isinstance(expr, IRNode):
        parts: List[Any] = [type(expr).__name__]
        for slot in _iter_slots(expr):
            if slot in {"location", "type_info", "shape_info", "defid"}:
                continue
            parts.append((slot, _expr_signature(getattr(expr, slot, None))))
        return tuple(parts)
    return repr(expr)


def _collect_all_defids_ir(node: object) -> Set[DefId]:
    out: Set[DefId] = set()
    seen: Set[int] = set()
    stack: List[object] = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(cur, (IdentifierIR, IndexVarIR, IndexRestIR)):
            did = getattr(cur, "defid", None)
            if did is not None:
                out.add(did)
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for slot in _iter_slots(cur):
                stack.append(getattr(cur, slot, None))
    return out

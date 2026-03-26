"""
Pre-Autodiff Pruning Pass.

Purpose:
- Provide a dedicated hook for safe pruning before autodiff.
- Currently only prunes compile-time constant `if` branches.

Important:
- Enabled in the pass pipeline before AutodiffPass.
- Intentionally conservative to avoid changing autodiff-sensitive IR shape.
"""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from ..ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BlockExpressionIR,
    BuiltinCallIR,
    BindingIR,
    DiffRuleIR,
    DifferentialIR,
    EinsteinClauseIR,
    ExpressionIR,
    IdentifierIR,
    IfExpressionIR,
    IRNode,
    LiteralIR,
    MemberAccessIR,
    ModuleIR,
    ProgramIR,
    RectangularAccessIR,
    TupleExpressionIR,
    WhereClauseIR,
)
from ..passes.base import BasePass, TyCtxt
from ..shared.types import BinaryOp, RectangularType

logger = logging.getLogger("einlang.passes.pre_autodiff_pruning")


class PreAutodiffPruningPass(BasePass):
    """Conservative pre-autodiff pruning pass."""

    requires = []

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        pruner = _IfBranchPruner()
        pruner.rewrite_program(ir)
        tcx.set_analysis(
            PreAutodiffPruningPass,
            {
                "enabled": True,
                "implemented": True,
                "pruned_if_count": pruner.pruned_if_count,
            },
        )
        logger.debug("PreAutodiffPruningPass complete: pruned_if=%d", pruner.pruned_if_count)
        return ir


class PostAutodiffPruningPass(PreAutodiffPruningPass):
    """Same pruning logic, scheduled after autodiff-generated IR exists."""


class _IfBranchPruner:
    def __init__(self) -> None:
        self.pruned_if_count = 0
        self._contains_differential_cache: dict[int, bool] = {}

    @staticmethod
    def _is_container_constant(value: Any) -> bool:
        return isinstance(value, (list, tuple, dict, set, range))

    @staticmethod
    def _looks_like_metadata_name(name: str | None) -> bool:
        if not name:
            return False
        lowered = name.lower()
        return lowered in {"rank", "ndim", "dims"} or "rank" in lowered or "shape" in lowered

    def rewrite_program(self, program: ProgramIR) -> None:
        for stmt in program.statements or []:
            self._rewrite_node(stmt)
        for mod in program.modules or []:
            self._rewrite_node(mod)

    def _rewrite_node(self, node: Any) -> Any:
        if node is None:
            return None
        if isinstance(node, ExpressionIR):
            return self._rewrite_expr(node)
        if isinstance(node, BindingIR):
            if node.expr is not None:
                node.expr = self._rewrite_expr(node.expr)
            return node
        if isinstance(node, DiffRuleIR):
            if node.body is not None:
                node.body = self._rewrite_expr(node.body)
            return node
        if isinstance(node, EinsteinClauseIR):
            node.value = self._rewrite_expr(node.value)
            if node.where_clause is not None:
                self._rewrite_node(node.where_clause)
            return node
        if isinstance(node, WhereClauseIR):
            node.constraints = tuple(
                self._rewrite_expr(c) if isinstance(c, ExpressionIR) else c
                for c in (node.constraints or [])
            )
            return node
        if isinstance(node, ModuleIR):
            for fn in node.functions or []:
                self._rewrite_node(fn)
            for const in node.constants or []:
                self._rewrite_node(const)
            for sub in node.submodules or []:
                self._rewrite_node(sub)
            return node
        if isinstance(node, IRNode):
            self._rewrite_slots(node)
            return node
        if isinstance(node, tuple):
            return tuple(self._rewrite_node(v) for v in node)
        if isinstance(node, list):
            return [self._rewrite_node(v) for v in node]
        if isinstance(node, dict):
            return {k: self._rewrite_node(v) for k, v in node.items()}
        return node

    def _rewrite_expr(self, expr: ExpressionIR, env: dict[Any, Any] | None = None) -> ExpressionIR:
        env = env or {}
        if isinstance(expr, BlockExpressionIR):
            return self._rewrite_block(expr, env)
        if isinstance(expr, DifferentialIR):
            return expr
        if isinstance(expr, IfExpressionIR):
            expr.condition = self._rewrite_expr(expr.condition, env)
            if expr.then_expr is not None:
                if isinstance(expr.then_expr, BlockExpressionIR):
                    expr.then_expr = self._rewrite_block(expr.then_expr, env, preserve_wrapper=True)
                else:
                    expr.then_expr = self._rewrite_expr(expr.then_expr, env)
            if expr.else_expr is not None:
                if isinstance(expr.else_expr, BlockExpressionIR):
                    expr.else_expr = self._rewrite_block(expr.else_expr, env, preserve_wrapper=True)
                else:
                    expr.else_expr = self._rewrite_expr(expr.else_expr, env)
            cond_value = self._eval_if_condition(expr.condition, env)
            if isinstance(cond_value, (bool, int)):
                self.pruned_if_count += 1
                if bool(cond_value):
                    return expr.then_expr
                if expr.else_expr is not None:
                    return expr.else_expr
            return expr
        self._rewrite_slots(expr, env)
        return expr

    def _rewrite_block(
        self,
        expr: BlockExpressionIR,
        env: dict[Any, Any],
        preserve_wrapper: bool = False,
    ) -> ExpressionIR:
        preserve_diff_block = self._contains_differential(expr)
        local_env = {} if preserve_diff_block else dict(env)
        new_statements = []
        for stmt in expr.statements or []:
            if isinstance(stmt, BindingIR):
                if stmt.expr is not None:
                    stmt.expr = self._rewrite_expr(stmt.expr, local_env)
                    const_value = self._eval_metadata_constant(stmt.expr, local_env)
                    if (
                        const_value is not None
                        and not preserve_diff_block
                        and self._should_remember_binding(stmt, stmt.expr, local_env)
                    ):
                        self._remember_binding_const(stmt, const_value, local_env)
                new_statements.append(stmt)
            else:
                rewritten = self._rewrite_node_with_env(stmt, local_env)
                new_statements.append(rewritten)
        expr.statements = tuple(self._prune_dead_constant_bindings(new_statements, expr.final_expr))
        if expr.final_expr is not None:
            expr.final_expr = self._rewrite_expr(expr.final_expr, local_env)
        if preserve_wrapper:
            return expr
        return self._simplify_block(expr)

    def _contains_differential(self, node: Any) -> bool:
        if node is None:
            return False
        if isinstance(node, (str, int, float, bool)):
            return False
        cache_key = id(node)
        cached = self._contains_differential_cache.get(cache_key)
        if cached is not None:
            return cached
        if isinstance(node, DifferentialIR):
            self._contains_differential_cache[cache_key] = True
            return True
        if isinstance(node, (list, tuple)):
            result = any(self._contains_differential(item) for item in node)
            self._contains_differential_cache[cache_key] = result
            return result
        if isinstance(node, dict):
            result = any(self._contains_differential(item) for item in node.values())
            self._contains_differential_cache[cache_key] = result
            return result
        if isinstance(node, BindingIR):
            result = self._contains_differential(node.expr)
            self._contains_differential_cache[cache_key] = result
            return result
        if isinstance(node, IRNode):
            for slot in _iter_slots(type(node)):
                if slot in {"location", "type_info", "shape_info", "name", "member", "defid"}:
                    continue
                if self._contains_differential(getattr(node, slot, None)):
                    self._contains_differential_cache[cache_key] = True
                    return True
        self._contains_differential_cache[cache_key] = False
        return False

    def _rewrite_slots(self, node: Any, env: dict[Any, Any] | None = None) -> None:
        env = env or {}
        for slot in _iter_slots(type(node)):
            if slot in {"location", "type_info", "shape_info", "name", "member", "defid"}:
                continue
            value = getattr(node, slot, None)
            new_value = self._rewrite_node_with_env(value, env)
            if new_value is not value:
                setattr(node, slot, new_value)

    def _rewrite_node_with_env(self, node: Any, env: dict[Any, Any]) -> Any:
        if node is None:
            return None
        if isinstance(node, ExpressionIR):
            return self._rewrite_expr(node, env)
        if isinstance(node, EinsteinClauseIR):
            node.value = self._rewrite_expr(node.value, env)
            if node.where_clause is not None:
                self._rewrite_where_clause(node.where_clause, env)
            return node
        if isinstance(node, WhereClauseIR):
            self._rewrite_where_clause(node, env)
            return node
        if isinstance(node, tuple):
            return tuple(self._rewrite_node_with_env(v, env) for v in node)
        if isinstance(node, list):
            return [self._rewrite_node_with_env(v, env) for v in node]
        if isinstance(node, dict):
            return {k: self._rewrite_node_with_env(v, env) for k, v in node.items()}
        if isinstance(node, IRNode):
            self._rewrite_slots(node, env)
        return node

    def _rewrite_where_clause(self, node: WhereClauseIR, env: dict[Any, Any]) -> None:
        node.constraints = tuple(
            self._rewrite_expr(c, env) if isinstance(c, ExpressionIR) else c
            for c in (node.constraints or [])
        )

    def _remember_binding_const(self, binding: BindingIR, value: Any, env: dict[Any, Any]) -> None:
        if binding.defid is not None:
            env[binding.defid] = value
        if binding.name:
            env[binding.name] = value

    def _shape_tuple(self, expr: ExpressionIR) -> tuple[Any, ...] | None:
        shape_info = getattr(expr, "shape_info", None)
        if isinstance(shape_info, tuple):
            return shape_info
        if isinstance(shape_info, list):
            return tuple(shape_info)
        type_info = getattr(expr, "type_info", None)
        if isinstance(type_info, RectangularType) and type_info.shape is not None:
            return tuple(type_info.shape)
        return None

    def _is_metadata_derived_expr(self, expr: ExpressionIR, env: dict[Any, Any]) -> bool:
        if isinstance(expr, IdentifierIR):
            return (
                (expr.defid is not None and expr.defid in env)
                or (bool(expr.name) and expr.name in env)
            )
        if isinstance(expr, BuiltinCallIR):
            return expr.builtin_name in {"len", "shape"} or any(
                self._is_metadata_derived_expr(arg, env) for arg in (expr.args or [])
            )
        if isinstance(expr, MemberAccessIR):
            return expr.member == "shape" or self._is_metadata_derived_expr(expr.object, env)
        if isinstance(expr, RectangularAccessIR):
            return self._is_metadata_derived_expr(expr.array, env) or any(
                self._is_metadata_derived_expr(idx, env) for idx in (expr.indices or [])
            )
        if isinstance(expr, BinaryOpIR):
            return self._is_metadata_derived_expr(expr.left, env) or self._is_metadata_derived_expr(expr.right, env)
        if isinstance(expr, ArrayLiteralIR):
            return any(self._is_metadata_derived_expr(e, env) for e in (expr.elements or []))
        if isinstance(expr, TupleExpressionIR):
            return any(self._is_metadata_derived_expr(e, env) for e in (expr.elements or []))
        return False

    def _should_remember_binding(
        self,
        binding: BindingIR,
        expr: ExpressionIR,
        env: dict[Any, Any],
    ) -> bool:
        if self._looks_like_metadata_name(binding.name):
            return True
        return self._is_metadata_derived_expr(expr, env)

    def _eval_if_condition(self, expr: ExpressionIR, env: dict[Any, Any]) -> Any | None:
        if isinstance(expr, LiteralIR) and isinstance(expr.value, (bool, int)):
            return expr.value
        return self._eval_metadata_constant(expr, env)

    def _eval_metadata_constant(self, expr: ExpressionIR, env: dict[Any, Any]) -> Any | None:
        if isinstance(expr, LiteralIR):
            return expr.value if isinstance(expr.value, (bool, int, float, str)) else None
        if isinstance(expr, IdentifierIR):
            if expr.defid is not None and expr.defid in env:
                return env[expr.defid]
            if expr.name in env:
                return env[expr.name]
            return None
        if isinstance(expr, ArrayLiteralIR):
            vals = [self._eval_metadata_constant(e, env) for e in (expr.elements or [])]
            if any(v is None for v in vals):
                return None
            return vals
        if isinstance(expr, TupleExpressionIR):
            vals = [self._eval_metadata_constant(e, env) for e in (expr.elements or [])]
            if any(v is None for v in vals):
                return None
            return tuple(vals)
        if isinstance(expr, MemberAccessIR) and expr.member == "shape":
            return self._shape_tuple(expr.object)
        if isinstance(expr, RectangularAccessIR) and len(expr.indices or []) == 1:
            base = self._eval_metadata_constant(expr.array, env)
            idx = self._eval_metadata_constant(expr.indices[0], env)
            if isinstance(base, (list, tuple)) and isinstance(idx, int) and 0 <= idx < len(base):
                return base[idx]
            return None
        if isinstance(expr, BuiltinCallIR) and expr.builtin_name == "len" and len(expr.args or []) == 1:
            arg_val = self._eval_metadata_constant(expr.args[0], env)
            if isinstance(arg_val, (list, tuple, str)):
                return len(arg_val)
            return None
        if isinstance(expr, BinaryOpIR):
            lhs = self._eval_metadata_constant(expr.left, env)
            rhs = self._eval_metadata_constant(expr.right, env)
            if lhs is None or rhs is None:
                return None
            if self._is_container_constant(lhs) or self._is_container_constant(rhs):
                return None
            try:
                if expr.operator == BinaryOp.EQ:
                    return lhs == rhs
                if expr.operator == BinaryOp.NE:
                    return lhs != rhs
                if expr.operator == BinaryOp.LT:
                    return lhs < rhs
                if expr.operator == BinaryOp.LE:
                    return lhs <= rhs
                if expr.operator == BinaryOp.GT:
                    return lhs > rhs
                if expr.operator == BinaryOp.GE:
                    return lhs >= rhs
                if expr.operator == BinaryOp.AND:
                    return bool(lhs) and bool(rhs)
                if expr.operator == BinaryOp.OR:
                    return bool(lhs) or bool(rhs)
                if expr.operator == BinaryOp.ADD:
                    return lhs + rhs
                if expr.operator == BinaryOp.SUB:
                    return lhs - rhs
                if expr.operator == BinaryOp.MUL:
                    return lhs * rhs
                if expr.operator == BinaryOp.DIV and rhs != 0:
                    return lhs / rhs
                if expr.operator == BinaryOp.MOD and rhs != 0:
                    return lhs % rhs
            except Exception:
                return None
        return None

    def _prune_dead_constant_bindings(
        self, statements: list[Any], final_expr: ExpressionIR | None
    ) -> list[Any]:
        live_defids, live_names = self._collect_live_refs(final_expr)
        kept_rev: list[Any] = []
        for stmt in reversed(statements):
            if not isinstance(stmt, BindingIR) or stmt.expr is None:
                kept_rev.append(stmt)
                s_defids, s_names = self._collect_live_refs(stmt)
                live_defids |= s_defids
                live_names |= s_names
                continue
            used = (
                stmt.defid is not None and stmt.defid in live_defids
            ) or (stmt.defid is None and bool(stmt.name) and stmt.name in live_names)
            if (
                not used
                and self._eval_metadata_constant(stmt.expr, {}) is not None
                and self._looks_like_metadata_name(stmt.name)
            ):
                continue
            kept_rev.append(stmt)
            s_defids, s_names = self._collect_live_refs(stmt.expr)
            live_defids |= s_defids
            live_names |= s_names
        kept_rev.reverse()
        return kept_rev

    def _collect_live_refs(self, node: Any) -> tuple[set[Any], set[str]]:
        seen: set[int] = set()
        defids: set[Any] = set()
        names: set[str] = set()

        def walk(cur: Any) -> None:
            if cur is None:
                return
            if isinstance(cur, (str, int, float, bool)):
                return
            oid = id(cur)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(cur, IdentifierIR):
                if cur.defid is not None:
                    defids.add(cur.defid)
                elif cur.name:
                    names.add(cur.name)
                return
            if isinstance(cur, BindingIR):
                if cur.expr is not None:
                    walk(cur.expr)
                return
            if isinstance(cur, (list, tuple)):
                for item in cur:
                    walk(item)
                return
            if isinstance(cur, dict):
                for item in cur.values():
                    walk(item)
                return
            if isinstance(cur, IRNode):
                for slot in _iter_slots(type(cur)):
                    if slot in {"location", "type_info", "shape_info", "name", "member", "defid"}:
                        continue
                    walk(getattr(cur, slot, None))
                return

        walk(node)
        return defids, names

    def _simplify_block(self, expr: BlockExpressionIR) -> ExpressionIR:
        if expr.final_expr is None:
            return expr
        if not expr.statements:
            inner = expr.final_expr
            if isinstance(inner, BlockExpressionIR):
                return self._simplify_block(inner)
            return inner
        if (
            len(expr.statements) == 1
            and isinstance(expr.statements[0], BindingIR)
            and expr.final_expr is not None
        ):
            stmt = expr.statements[0]
            if (
                stmt.expr is not None
                and self._eval_metadata_constant(stmt.expr, {}) is not None
                and self._looks_like_metadata_name(stmt.name)
            ):
                live_defids, live_names = self._collect_live_refs(expr.final_expr)
                used = (
                    stmt.defid is not None and stmt.defid in live_defids
                ) or (stmt.defid is None and bool(stmt.name) and stmt.name in live_names)
                if not used:
                    return expr.final_expr
        return expr


@lru_cache(maxsize=None)
def _iter_slots(cls: type) -> tuple[str, ...]:
    out: list[str] = []
    for c in cls.__mro__:
        slots = getattr(c, "__slots__", ())
        if isinstance(slots, str):
            out.append(slots)
        else:
            out.extend(slots)
    # Preserve order and de-duplicate.
    seen = set()
    ordered: list[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)

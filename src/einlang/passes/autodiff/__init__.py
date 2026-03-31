"""Autodiff pass — expand ``@expr`` and ``@y/@x`` into plain IR.

Design: ``docs/AUTODIFF_DESIGN.md`` · overview: ``docs/AUTODIFF_HIGHLIGHTS.md``

Forward mode propagates differentials (tangents). For ``y = f(x₁, x₂, …)``
we emit ``d(y) = Σᵢ (∂f/∂xᵢ) · d(xᵢ)`` in execution order.

Implementation phases in this module:
1. collect derivative / quotient requests
2. build the binding dependency context
3. compute the reachable primal subgraph
4. allocate differential refs and seed leaves
5. build differential bindings
6. splice ``@x`` bindings into the program
7. expand derivative references in-place
8. record autodiff analysis for the backend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ..base import BasePass, TyCtxt
from ..shape_analysis import UnifiedShapeAnalysisPass
from ..type_inference import TypeInferencePass
from ._cleanup import clear_custom_diff_body_everywhere
from ._core import DIFF_PREFIX, USER_DIFF_PREFIX, _LOC0, _fl, _si, _ti, _z
from ._expand import (
    _binding_shape_info,
    _binding_type_info,
    _bindings_in,
    _expand_program,
    _fold_shape_accesses_program,
    _fwd_expr,
    _inline_drhs,
    _propagate_ti,
)
from ._graph import (
    _DependencyQueryCache,
    _autodiff_primal_data_defids,
    _collect_targets,
    _collect_targets_expr,
)
from ...ir.nodes import BindingIR, BinaryOpIR, DifferentialIR, ExpressionIR, FunctionValueIR, IRNode, IdentifierIR, IndexRestIR, IndexVarIR, ProgramIR, RectangularAccessIR, is_function_binding
from ...shared.defid import DefId
from ...shared.types import BinaryOp, RectangularType


@dataclass
class _AutodiffTargets:
    """What the pass must differentiate in this program."""

    diff_targets: List[Tuple[DefId, str]] = field(default_factory=list)
    standalone_diff_target_defids: Set[DefId] = field(default_factory=set)
    quotient_pairs: List[Tuple[DefId, DefId]] = field(default_factory=list)
    target_binding_defids: Set[DefId] = field(default_factory=set)
    target_statement_ids: Set[int] = field(default_factory=set)


@dataclass
class _AutodiffBindingContext:
    """Dependency queries over the binding graph used by autodiff."""

    bindings: Dict[DefId, BindingIR]
    dep_cache: _DependencyQueryCache
    binding_deps: Dict[DefId, Set[DefId]] = field(default_factory=dict)

    def deps_for(self, did: DefId) -> Set[DefId]:
        cached = self.binding_deps.get(did)
        if cached is not None:
            return cached
        binding = self.bindings.get(did)
        if binding is None or is_function_binding(binding):
            out: Set[DefId] = set()
        else:
            out = _autodiff_primal_data_defids(binding.expr, self.bindings, self.dep_cache)
        self.binding_deps[did] = out
        return out


@dataclass
class _AutodiffReachability:
    """The top-level bindings that participate in a derivative request."""

    top_targets: Set[DefId]
    reachable: Set[DefId]
    ordered: List[BindingIR]


@dataclass
class _AutodiffSeedPlan:
    """Allocated differential refs, leaves, and computed RHS bindings."""

    diff_refs: Dict[DefId, ExpressionIR]
    diff_bindings: Dict[DefId, BindingIR]
    differential_leaves: Set[DefId]


class AutodiffPass(BasePass):
    """Expand ``@expr`` and ``@y/@x`` into plain IR."""

    requires = [TypeInferencePass, UnifiedShapeAnalysisPass]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        try:
            return self._core(ir, tcx)
        finally:
            from ...ir.nodes import clear_autodiff_only_fields

            clear_autodiff_only_fields(ir)

    def _core(self, program: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        bindings = _bindings_in(program, program) or []
        if not bindings:
            tcx.set_analysis(
                AutodiffPass,
                {
                    "diff_block": None,
                    "differential_targets": set(),
                    "differential_buffer_by_defid": {},
                },
            )
            return program

        targets = self._collect_requested_targets(program, bindings)
        binding_ctx = self._build_binding_context(bindings, tcx)

        resolver = getattr(tcx, "resolver", None)
        if resolver is None:
            self._record_empty_analysis(tcx, targets.diff_targets)
            return program

        seed_plan = _AutodiffSeedPlan(diff_refs={}, diff_bindings={}, differential_leaves=set())
        if targets.diff_targets:
            reachability = self._compute_reachability(bindings, targets, binding_ctx)
            seed_plan = self._build_seed_plan(
                bindings,
                targets,
                reachability,
                binding_ctx,
                resolver,
            )
            self._splice_diff_bindings(program, bindings, seed_plan.diff_bindings)

        _expand_program(
            program,
            seed_plan.diff_refs,
            _LOC0,
            resolver,
            init_B=binding_ctx.bindings,
            target_binding_defids=targets.target_binding_defids,
            target_statement_ids=targets.target_statement_ids,
        )
        self._splice_diff_bindings(program, bindings, seed_plan.diff_bindings)
        autodiff_defids = {b.defid for b in seed_plan.diff_bindings.values() if b.defid is not None}
        preserve_primal_targets = set(targets.standalone_diff_target_defids)
        autodiff_defid_to_primal = {
            b.defid: primal_did
            for primal_did, b in seed_plan.diff_bindings.items()
            if b.defid is not None
        }
        self._prune_unreferenced_autodiff_bindings(
            program,
            autodiff_defids,
            autodiff_defid_to_primal=autodiff_defid_to_primal,
            preserve_primal_targets=preserve_primal_targets,
        )
        if autodiff_defids:
            _fold_shape_accesses_program(program, _LOC0, init_B=binding_ctx.bindings)

        self._record_analysis(
            tcx,
            targets.diff_targets,
            seed_plan.diff_bindings,
            seed_plan.differential_leaves,
        )
        return program

    def _record_empty_analysis(
        self,
        tcx: TyCtxt,
        diff_targets: List[Tuple[DefId, str]],
    ) -> None:
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": None,
                "differential_targets": set(diff_targets),
                "differential_buffer_by_defid": {},
            },
        )

    def _collect_requested_targets(
        self,
        program: ProgramIR,
        bindings: List[BindingIR],
    ) -> _AutodiffTargets:
        targets = _AutodiffTargets()
        for b in bindings:
            if b.expr is None:
                continue
            bt, bq = _collect_targets_expr(b.expr)
            targets.standalone_diff_target_defids.update(_collect_standalone_diff_targets_expr(b.expr))
            if bt or bq:
                if b.defid is not None:
                    targets.target_binding_defids.add(b.defid)
                targets.diff_targets.extend(bt)
                targets.quotient_pairs.extend(bq)
        for s in program.statements or []:
            if isinstance(s, BindingIR) or not isinstance(s, ExpressionIR):
                continue
            st, sq = _collect_targets_expr(s)
            targets.standalone_diff_target_defids.update(_collect_standalone_diff_targets_expr(s))
            if st or sq:
                targets.target_statement_ids.add(id(s))
                targets.diff_targets.extend(st)
                targets.quotient_pairs.extend(sq)
        return targets

    def _build_binding_context(
        self,
        bindings: List[BindingIR],
        tcx: TyCtxt,
    ) -> _AutodiffBindingContext:
        binding_map: Dict[DefId, BindingIR] = {}
        for b in bindings:
            if b.defid is not None:
                binding_map[b.defid] = b
        fim = getattr(tcx, "function_ir_map", None) or {}
        for did, fn in fim.items():
            if did is not None and did not in binding_map and isinstance(fn, BindingIR) and is_function_binding(fn):
                binding_map[did] = fn
        return _AutodiffBindingContext(
            bindings=binding_map,
            dep_cache=_DependencyQueryCache(binding_map),
        )

    def _compute_reachability(
        self,
        bindings: List[BindingIR],
        targets: _AutodiffTargets,
        binding_ctx: _AutodiffBindingContext,
    ) -> _AutodiffReachability:
        td: Set[DefId] = set()
        for did, _ in targets.diff_targets:
            td.add(did)
        for n, d in targets.quotient_pairs:
            td.add(n)
            td.add(d)
        top = {b.defid for b in bindings if b.defid is not None}
        td_top = td & top
        reach: Set[DefId] = set()
        wk = list(td_top)
        while wk:
            did = wk.pop()
            if did in reach:
                continue
            reach.add(did)
            b = binding_ctx.bindings.get(did)
            if b is None:
                continue
            for dep in binding_ctx.deps_for(did):
                if dep not in reach:
                    wk.append(dep)

        fwd: List[BindingIR] = []
        seen: Set[DefId] = set()

        def _vis(did: DefId) -> None:
            if did in seen or did not in reach:
                return
            seen.add(did)
            b = binding_ctx.bindings.get(did)
            if b is None:
                return
            for dep in binding_ctx.deps_for(did):
                _vis(dep)
            fwd.append(b)

        for did in td_top:
            _vis(did)
        return _AutodiffReachability(top_targets=td_top, reachable=reach, ordered=fwd)

    def _global_binding_type_shape(
        self,
        bindings: List[BindingIR],
    ) -> Tuple[object, object]:
        gti = gsi = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                gti = _ti(b) or _ti(b.expr)
                gsi = (_si(b) or _si(b.expr)) if isinstance(gti, RectangularType) else None
                if gti is not None:
                    break
        return gti, gsi

    def _build_seed_plan(
        self,
        bindings: List[BindingIR],
        targets: _AutodiffTargets,
        reachability: _AutodiffReachability,
        binding_ctx: _AutodiffBindingContext,
        resolver: object,
    ) -> _AutodiffSeedPlan:
        gti, gsi = self._global_binding_type_shape(bindings)
        D: Dict[DefId, ExpressionIR] = {}
        d2b: Dict[DefId, BindingIR] = {}
        sv: Dict[DefId, int] = {}
        qd = {d for _, d in targets.quotient_pairs}
        lvs = {
            did for did in reachability.reachable if not (binding_ctx.binding_deps.get(did) or binding_ctx.deps_for(did))
        }
        for b in reachability.ordered:
            if b.defid is None or b.defid not in reachability.reachable:
                continue
            dn = USER_DIFF_PREFIX + (b.name or "")
            dd = resolver.allocate_for_local()
            ti0 = _binding_type_info(b, binding_ctx.bindings)
            si0 = _binding_shape_info(b, binding_ctx.bindings)
            if not isinstance(b.expr, RectangularAccessIR):
                ti0 = ti0 or gti
                if isinstance(ti0, RectangularType):
                    si0 = si0 or gsi
            from ...ir.nodes import IdentifierIR

            D[b.defid] = IdentifierIR(dn, b.location or _LOC0, dd, type_info=ti0, shape_info=si0)
            if b.defid in qd:
                sv[b.defid] = 1
            elif b.defid in lvs and (
                b.defid in reachability.top_targets or not targets.quotient_pairs
            ):
                sv[b.defid] = 1
            else:
                sv[b.defid] = 0

        dre = {did: ref for did, ref in D.items()}
        drhs_map = {}
        upq = len(targets.quotient_pairs) > 0
        for b in reachability.ordered:
            if b.defid is None or b.defid not in reachability.reachable:
                continue
            bl = b.location or _LOC0
            if b.defid in sv and sv[b.defid] == 1:
                drhs_map[b.defid] = _fl(1, bl)
            elif upq and b.defid in lvs:
                drhs_map[b.defid] = _z(bl)
            else:
                for dep in binding_ctx.deps_for(b.defid) or []:
                    if dep not in dre:
                        dre[dep] = _z(bl)
                rhs = _fwd_expr(b, dre, binding_ctx.bindings, binding_ctx.binding_deps, bl, resolver)
                drhs_map[b.defid] = _inline_drhs(rhs if rhs is not None else _z(bl), bl)

        for b in reachability.ordered:
            if b.defid is None or b.defid not in reachability.reachable:
                continue
            rhs = drhs_map.get(b.defid) or _z(b.location or _LOC0)
            ti = _binding_type_info(b, binding_ctx.bindings)
            si = _binding_shape_info(b, binding_ctx.bindings)
            if not isinstance(b.expr, RectangularAccessIR):
                ti = ti or gti
                if isinstance(ti, RectangularType):
                    si = si or gsi
                else:
                    si = None
            _propagate_ti(rhs, ti, si)
            ref = D[b.defid]
            d2b[b.defid] = BindingIR(name=ref.name, expr=rhs, location=b.location, defid=ref.defid, type_info=ti)
        return _AutodiffSeedPlan(
            diff_refs=D,
            diff_bindings=d2b,
            differential_leaves=lvs,
        )

    def _splice_diff_bindings(
        self,
        program: ProgramIR,
        bindings: List[BindingIR],
        diff_bindings: Dict[DefId, BindingIR],
    ) -> None:
        nb: List[BindingIR] = []
        for b in bindings:
            nb.append(b)
            db = diff_bindings.get(b.defid)
            if db is not None:
                nb.append(db)
        program.bindings = nb
        non_b = [s for s in (program.statements or []) if not isinstance(s, BindingIR)]
        program.statements = nb + non_b

    def _prune_unreferenced_autodiff_bindings(
        self,
        program: ProgramIR,
        autodiff_binding_defids: Set[DefId],
        *,
        autodiff_defid_to_primal: Dict[DefId, DefId] = None,
        preserve_primal_targets: Set[DefId] = None,
    ) -> None:
        autodiff_defid_to_primal = autodiff_defid_to_primal or {}
        preserve_primal_targets = preserve_primal_targets or set()
        bindings = [b for b in (program.bindings or []) if isinstance(b, BindingIR)]
        binding_map: Dict[DefId, BindingIR] = {b.defid: b for b in bindings if b.defid is not None}
        live: Set[DefId] = set()
        work: List[ExpressionIR] = []

        for stmt in program.statements or []:
            if isinstance(stmt, BindingIR):
                if stmt.defid in autodiff_binding_defids:
                    continue
                if stmt.defid is not None:
                    live.add(stmt.defid)
                if stmt.expr is not None:
                    work.append(stmt.expr)
            elif isinstance(stmt, ExpressionIR):
                work.append(stmt)

        while work:
            expr = work.pop()
            for did in _collect_all_defids_ir(expr):
                if did in live:
                    continue
                live.add(did)
                binding = binding_map.get(did)
                if binding is not None and binding.expr is not None:
                    work.append(binding.expr)

        new_statements: List[object] = []
        for stmt in program.statements or []:
            if isinstance(stmt, BindingIR) and stmt.defid in autodiff_binding_defids:
                primal_did = autodiff_defid_to_primal.get(stmt.defid)
                if primal_did in preserve_primal_targets:
                    new_statements.append(stmt)
                    continue
                if stmt.defid is None or stmt.defid not in live:
                    continue
            new_statements.append(stmt)
        program.statements = new_statements
        program.bindings = [s for s in new_statements if isinstance(s, BindingIR)]

    def _record_analysis(
        self,
        tcx: TyCtxt,
        diff_targets: List[Tuple[DefId, str]],
        diff_bindings: Dict[DefId, BindingIR],
        differential_leaves: Set[DefId],
    ) -> None:
        dbl = list(diff_bindings.values())
        adm: Dict[DefId, DefId] = {p: diff_bindings[p].defid for p in diff_bindings}
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": dbl or None,
                "differential_targets": set(diff_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": adm,
                "differential_leaves": differential_leaves,
            },
        )


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
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


def _collect_standalone_diff_targets_expr(expr: ExpressionIR) -> Set[DefId]:
    out: Set[DefId] = set()
    seen: Set[int] = set()
    stack: List[object] = [expr]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        if (
            isinstance(cur, BinaryOpIR)
            and cur.operator == BinaryOp.DIV
            and isinstance(cur.left, DifferentialIR)
            and isinstance(cur.right, DifferentialIR)
        ):
            continue
        if isinstance(cur, DifferentialIR):
            op = cur.operand
            if isinstance(op, IdentifierIR) and op.defid is not None:
                out.add(op.defid)
            stack.append(op)
            continue
        if isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
            continue
        if isinstance(cur, (list, tuple)):
            stack.extend(cur)
            continue
        if isinstance(cur, IRNode):
            for cls in type(cur).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    stack.append(getattr(cur, slot, None))
    return out


_collect_autodiff_targets = _collect_targets

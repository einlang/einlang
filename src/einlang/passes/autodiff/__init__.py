"""Autodiff pass — expand ``@expr`` and ``@y/@x`` into plain IR.

Design: ``docs/AUTODIFF_DESIGN.md`` · overview: ``docs/AUTODIFF_HIGHLIGHTS.md``

Forward mode propagates differentials (tangents). For ``y = f(x₁, x₂, …)``
we emit ``d(y) = Σᵢ (∂f/∂xᵢ) · d(xᵢ)`` in execution order.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..base import BasePass, TyCtxt
from ..shape_analysis import UnifiedShapeAnalysisPass
from ..type_inference import TypeInferencePass
from ._cleanup import clear_custom_diff_body_everywhere
from ._core import DIFF_PREFIX, USER_DIFF_PREFIX, _LOC0, _fl, _si, _ti, _z
from ._expand import _bindings_in, _expand_program, _fwd_expr, _inline_drhs, _propagate_ti
from ._graph import (
    _DependencyQueryCache,
    _autodiff_primal_data_defids,
    _collect_targets,
    _collect_targets_expr,
)
from ...ir.nodes import BindingIR, ExpressionIR, FunctionValueIR, ProgramIR, is_function_binding
from ...shared.defid import DefId


class AutodiffPass(BasePass):
    """Expand ``@expr`` and ``@y/@x`` into plain IR via forward-mode autodiff."""

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

        diff_targets: List[Tuple[DefId, str]] = []
        q_pairs: List[Tuple[DefId, DefId]] = []
        target_binding_defids: Set[DefId] = set()
        target_statement_ids: Set[int] = set()
        for b in bindings:
            if b.expr is None:
                continue
            bt, bq = _collect_targets_expr(b.expr)
            if bt or bq:
                if b.defid is not None:
                    target_binding_defids.add(b.defid)
                diff_targets.extend(bt)
                q_pairs.extend(bq)
        for s in program.statements or []:
            if isinstance(s, BindingIR) or not isinstance(s, ExpressionIR):
                continue
            st, sq = _collect_targets_expr(s)
            if st or sq:
                target_statement_ids.add(id(s))
                diff_targets.extend(st)
                q_pairs.extend(sq)

        B: Dict[DefId, BindingIR] = {}
        for b in bindings:
            if b.defid is not None:
                B[b.defid] = b
        fim = getattr(tcx, "function_ir_map", None) or {}
        for did, fn in fim.items():
            if did is not None and did not in B and isinstance(fn, BindingIR) and is_function_binding(fn):
                B[did] = fn

        dep_cache = _DependencyQueryCache(B)
        b2d: Dict[DefId, Set[DefId]] = {}

        def _binding_deps(did: DefId) -> Set[DefId]:
            cached = b2d.get(did)
            if cached is not None:
                return cached
            b = B.get(did)
            if b is None or is_function_binding(b):
                out: Set[DefId] = set()
            else:
                out = _autodiff_primal_data_defids(b.expr, B, dep_cache)
            b2d[did] = out
            return out

        td: Set[DefId] = set()
        for did, _ in diff_targets:
            td.add(did)
        for n, d in q_pairs:
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
            b = B.get(did)
            if b is None:
                continue
            for dep in _binding_deps(did):
                if dep not in reach:
                    wk.append(dep)

        fwd: List[BindingIR] = []
        seen: Set[DefId] = set()

        def _vis(did: DefId) -> None:
            if did in seen or did not in reach:
                return
            seen.add(did)
            b = B.get(did)
            if b is None:
                return
            for dep in _binding_deps(did):
                _vis(dep)
            fwd.append(b)

        for did in td_top:
            _vis(did)

        R = getattr(tcx, "resolver", None)
        if R is None:
            tcx.set_analysis(
                AutodiffPass,
                {
                    "diff_block": None,
                    "differential_targets": set(diff_targets),
                    "differential_buffer_by_defid": {},
                },
            )
            return program

        gti = gsi = None
        for b in bindings:
            if b.expr is not None and not isinstance(b.expr, FunctionValueIR):
                gti = _ti(b) or _ti(b.expr)
                gsi = _si(b) or _si(b.expr)
                if gti is not None:
                    break

        D = {}
        d2b: Dict[DefId, BindingIR] = {}
        sv: Dict[DefId, int] = {}
        qd = {d for _, d in q_pairs}
        lvs = {did for did in reach if not (b2d.get(did) or set())}
        for b in fwd:
            if b.defid is None or b.defid not in reach:
                continue
            dn = USER_DIFF_PREFIX + (b.name or "")
            dd = R.allocate_for_local()
            ti0 = _ti(b) or (_ti(b.expr) if b.expr else None) or gti
            si0 = _si(b) or (_si(b.expr) if b.expr else None) or gsi
            from ...ir.nodes import IdentifierIR

            D[b.defid] = IdentifierIR(dn, b.location or _LOC0, dd, type_info=ti0, shape_info=si0)
            if b.defid in qd:
                sv[b.defid] = 1
            elif b.defid in lvs and (b.defid in td or not q_pairs):
                sv[b.defid] = 1
            else:
                sv[b.defid] = 0

        dre = {did: ref for did, ref in D.items()}
        drhs_map = {}
        upq = len(q_pairs) > 0
        for b in fwd:
            if b.defid is None or b.defid not in reach:
                continue
            bl = b.location or _LOC0
            if b.defid in sv and sv[b.defid] == 1:
                drhs_map[b.defid] = _fl(1, bl)
            elif upq and b.defid in lvs:
                drhs_map[b.defid] = _z(bl)
            else:
                for dep in b2d.get(b.defid) or []:
                    if dep not in dre:
                        dre[dep] = _z(bl)
                rhs = _fwd_expr(b, dre, B, b2d, bl, R)
                drhs_map[b.defid] = _inline_drhs(rhs if rhs is not None else _z(bl), bl)

        for b in fwd:
            if b.defid is None or b.defid not in reach:
                continue
            rhs = drhs_map.get(b.defid) or _z(b.location or _LOC0)
            ti = _ti(b) or (_ti(b.expr) if b.expr else None) or gti
            si = _si(b) or (_si(b.expr) if b.expr else None) or gsi
            _propagate_ti(rhs, ti, si)
            ref = D[b.defid]
            d2b[b.defid] = BindingIR(name=ref.name, expr=rhs, location=b.location, defid=ref.defid, type_info=ti)

        nb: List[BindingIR] = []
        for b in bindings:
            nb.append(b)
            db = d2b.get(b.defid)
            if db is not None:
                nb.append(db)
        program.bindings = nb
        non_b = [s for s in (program.statements or []) if not isinstance(s, BindingIR)]
        program.statements = nb + non_b

        _expand_program(
            program,
            D,
            _LOC0,
            R,
            init_B=B,
            target_binding_defids=target_binding_defids,
            target_statement_ids=target_statement_ids,
        )

        dbl = [d2b[b.defid] for b in fwd if b.defid in d2b]
        adm: Dict[DefId, DefId] = {p: d2b[p].defid for p in d2b}
        tcx.set_analysis(
            AutodiffPass,
            {
                "diff_block": dbl or None,
                "differential_targets": set(diff_targets),
                "differential_buffer_by_defid": {},
                "autodiff_differential_map": adm,
                "differential_leaves": lvs,
            },
        )
        return program


_collect_autodiff_targets = _collect_targets

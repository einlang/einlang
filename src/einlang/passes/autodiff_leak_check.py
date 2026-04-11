"""
Autodiff Leak Check Pass

Ensures no autodiff-only IR artifacts reach later passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from ..passes.autodiff import AutodiffPass
from ..passes.base import BasePass, TyCtxt
from ..passes.pre_autodiff_pruning import PostAutodiffPruningPass
from ..ir.nodes import (
    DiffRuleIR,
    DifferentialIR,
    FunctionValueIR,
    IRNode,
    ProgramIR,
)
from ..shared.types import DifferentialType, Type, strip_differential_types_deep

logger = logging.getLogger("einlang.passes.autodiff_leak_check")


@dataclass
class _Leak:
    kind: str
    origin: str
    location: Optional[Any]


class _LeakFinder:
    def __init__(self) -> None:
        self._seen: set[int] = set()
        self.leaks: List[_Leak] = []

    def scan(self, obj: Any, origin: str) -> None:
        self._walk(obj, origin)

    def _record(self, kind: str, origin: str, location: Optional[Any]) -> None:
        if len(self.leaks) >= 5:
            return
        self.leaks.append(_Leak(kind=kind, origin=origin, location=location))

    def _type_has_differential(self, ty: Any) -> bool:
        if ty is None:
            return False
        if isinstance(ty, DifferentialType):
            return True
        if isinstance(ty, Type):
            stripped = strip_differential_types_deep(ty)
            return stripped != ty
        return False

    def _walk(self, obj: Any, origin: str) -> None:
        if obj is None:
            return
        if isinstance(obj, (str, int, float, bool, bytes)):
            return
        oid = id(obj)
        if oid in self._seen:
            return
        self._seen.add(oid)

        if isinstance(obj, DifferentialIR):
            self._record("DifferentialIR", origin, getattr(obj, "location", None))
        elif isinstance(obj, DiffRuleIR):
            self._record("DiffRuleIR", origin, getattr(obj, "location", None))
        elif isinstance(obj, FunctionValueIR):
            if getattr(obj, "custom_diff_body", None) is not None:
                self._record("custom_diff_body", origin, getattr(obj, "location", None))

        if isinstance(obj, DifferentialType):
            self._record("DifferentialType", origin, None)
            return
        if isinstance(obj, Type):
            if self._type_has_differential(obj):
                self._record("DifferentialType", origin, None)
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                self._walk(k, origin)
                self._walk(v, origin)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                self._walk(item, origin)
            return
        if isinstance(obj, IRNode):
            for cls in type(obj).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    if slot == "location":
                        continue
                    self._walk(getattr(obj, slot, None), origin)
            return


class AutodiffLeakCheckPass(BasePass):
    """Fail-fast if autodiff-only IR survives past AutodiffPass."""

    requires = [PostAutodiffPruningPass, AutodiffPass]

    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        finder = _LeakFinder()
        finder.scan(ir, "program")

        # Also scan function IR map entries not in program (later passes may touch them).
        func_map = getattr(tcx, "function_ir_map", None) or {}
        for binding in func_map.values():
            if binding is None:
                continue
            finder.scan(binding, "function_ir_map")

        # Scan autodiff diff block, if present.
        try:
            analysis = tcx.get_analysis(AutodiffPass)
        except RuntimeError:
            analysis = {}
        diff_block = analysis.get("diff_block") if isinstance(analysis, dict) else None
        if diff_block:
            finder.scan(diff_block, "diff_block")

        if finder.leaks:
            first = finder.leaks[0]
            tcx.reporter.report_error(
                message=(
                    f"Autodiff cleanup leaked {first.kind} into `{first.origin}`; "
                    "autodiff-only IR must be fully expanded/cleared before later passes."
                ),
                location=first.location,
                code="E0999",
                help="This is a compiler bug. Ensure AutodiffPass expands @ expressions and clears autodiff-only fields.",
            )
            raise RuntimeError("Autodiff cleanup leak detected")

        logger.debug("Autodiff leak check passed")
        return ir


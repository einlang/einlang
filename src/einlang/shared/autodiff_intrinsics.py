"""Structural runtime intrinsics shared by autodiff compiler and runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .defid import BUILTIN_CRATE, DefId


_AUTODIFF_INTERNAL_BUILTIN_BASE = 10000


class AutodiffBuiltinKind(Enum):
    TANGENT = 0
    JACOBIAN = 1
    SYMBOLIC_TANGENT = 2
    SYMBOLIC_JACOBIAN = 3


@dataclass(frozen=True)
class AutodiffBuiltinSpec:
    kind: AutodiffBuiltinKind
    defid: DefId
    builtin_name: str


_AUTODIFF_BUILTIN_SPECS = (
    AutodiffBuiltinSpec(
        AutodiffBuiltinKind.TANGENT,
        DefId(BUILTIN_CRATE, _AUTODIFF_INTERNAL_BUILTIN_BASE + 0),
        "__autodiff_tangent",
    ),
    AutodiffBuiltinSpec(
        AutodiffBuiltinKind.JACOBIAN,
        DefId(BUILTIN_CRATE, _AUTODIFF_INTERNAL_BUILTIN_BASE + 1),
        "__autodiff_jacobian",
    ),
    AutodiffBuiltinSpec(
        AutodiffBuiltinKind.SYMBOLIC_TANGENT,
        DefId(BUILTIN_CRATE, _AUTODIFF_INTERNAL_BUILTIN_BASE + 2),
        "__autodiff_symbolic_tangent",
    ),
    AutodiffBuiltinSpec(
        AutodiffBuiltinKind.SYMBOLIC_JACOBIAN,
        DefId(BUILTIN_CRATE, _AUTODIFF_INTERNAL_BUILTIN_BASE + 3),
        "__autodiff_symbolic_jacobian",
    ),
)

_AUTODIFF_BUILTIN_SPEC_BY_KIND: Dict[AutodiffBuiltinKind, AutodiffBuiltinSpec] = {
    spec.kind: spec for spec in _AUTODIFF_BUILTIN_SPECS
}
_AUTODIFF_BUILTIN_KIND_BY_DEFID: Dict[DefId, AutodiffBuiltinKind] = {
    spec.defid: spec.kind for spec in _AUTODIFF_BUILTIN_SPECS
}


def autodiff_builtin_spec(kind: AutodiffBuiltinKind) -> AutodiffBuiltinSpec:
    return _AUTODIFF_BUILTIN_SPEC_BY_KIND[kind]


def autodiff_builtin_defid(kind: AutodiffBuiltinKind) -> DefId:
    return autodiff_builtin_spec(kind).defid


def autodiff_builtin_name(kind: AutodiffBuiltinKind) -> str:
    return autodiff_builtin_spec(kind).builtin_name


def autodiff_builtin_kind(defid: Optional[DefId]) -> Optional[AutodiffBuiltinKind]:
    if defid is None:
        return None
    return _AUTODIFF_BUILTIN_KIND_BY_DEFID.get(defid)

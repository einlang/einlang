"""Shared helpers for coordinate-aware signature instantiation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from ..shared.types import RectangularType


def is_pack(dim: Any) -> bool:
    return isinstance(dim, str) and dim.startswith("..")


def pack_name(dim: str) -> str:
    return dim[2:]


def is_coord_pack_param(param: Any) -> bool:
    return isinstance(param, str) and param.startswith("..")


def coord_param_name(param: str) -> str:
    return param[2:] if is_coord_pack_param(param) else param


def coordinate_arg_names(arg: Any) -> Tuple[str, ...]:
    if arg is None:
        return ()
    elements = getattr(arg, "elements", None)
    if elements is not None:
        return tuple(
            str(getattr(item, "name", ""))
            for item in elements
            if getattr(item, "name", None)
        )
    name = getattr(arg, "name", None)
    return (str(name),) if name else ()


def layout_from_type(ty: Any) -> Optional[Tuple[str, ...]]:
    if not isinstance(ty, RectangularType) or ty.shape is None:
        return None
    axes = [str(dim) for dim in ty.shape if isinstance(dim, str)]
    return tuple(axes) if axes else None


def match_symbolic_sequence(
    formal: Sequence[Any],
    actual: Sequence[Any],
    *,
    axes: Optional[Dict[str, str]] = None,
    axis_packs: Optional[Dict[str, Tuple[str, ...]]] = None,
    dims: Optional[Dict[str, Any]] = None,
    packs: Optional[Dict[str, Tuple[Any, ...]]] = None,
    actual_layout: Optional[Sequence[str]] = None,
) -> bool:
    """Match a formal shape/layout sequence with named dims and ``..packs``.

    ``axes`` anchors coordinate parameters, e.g. ``{"j": "class"}``; when an
    actual coordinate layout is available, the formal ``j`` must land on the
    actual ``class`` position. ``dims`` captures ordinary symbolic dimensions,
    while ``packs`` captures ``..left``/``..right`` slices.
    """

    axes = axes if axes is not None else {}
    axis_packs = axis_packs if axis_packs is not None else {}
    dims = dims if dims is not None else {}
    packs = packs if packs is not None else {}
    formal = tuple(formal)
    actual = tuple(actual)
    actual_layout_t = tuple(actual_layout) if actual_layout is not None else None

    def rec(fi: int, ai: int) -> bool:
        if fi == len(formal):
            return ai == len(actual)

        dim = formal[fi]
        if is_pack(dim):
            name = pack_name(dim)
            axis_pack = axis_packs.get(name)
            bound = packs.get(name)
            if bound is not None:
                end = ai + len(bound)
                if axis_pack is not None and actual_layout_t is not None:
                    if tuple(actual_layout_t[ai:end]) != axis_pack:
                        return False
                return tuple(actual[ai:end]) == bound and rec(fi + 1, end)
            for end in range(ai, len(actual) + 1):
                if axis_pack is not None and actual_layout_t is not None:
                    if tuple(actual_layout_t[ai:end]) != axis_pack:
                        continue
                if axis_pack is not None and actual_layout_t is None:
                    if tuple(str(item) for item in actual[ai:end]) != axis_pack:
                        continue
                packs[name] = tuple(actual[ai:end])
                if rec(fi + 1, end):
                    return True
            packs.pop(name, None)
            return False

        if ai >= len(actual):
            return False

        if isinstance(dim, str):
            axis = axes.get(dim)
            if axis is not None:
                if actual_layout_t is not None:
                    if ai >= len(actual_layout_t) or actual_layout_t[ai] != axis:
                        return False
                elif str(actual[ai]) != axis:
                    return False

            bound = dims.get(dim)
            if bound is not None and bound != actual[ai]:
                return False
            dims[dim] = actual[ai]
            return rec(fi + 1, ai + 1)

        if dim is not None and actual[ai] is not None and dim != actual[ai]:
            return False
        return rec(fi + 1, ai + 1)

    return rec(0, 0)


def instantiate_symbolic_sequence(
    formal: Sequence[Any],
    *,
    axes: Optional[Dict[str, str]] = None,
    dims: Optional[Dict[str, Any]] = None,
    packs: Optional[Dict[str, Tuple[Any, ...]]] = None,
) -> Optional[Tuple[Any, ...]]:
    axes = axes or {}
    dims = dims or {}
    packs = packs or {}
    result = []
    for dim in formal:
        if is_pack(dim):
            result.extend(packs.get(pack_name(dim), ()))
        elif isinstance(dim, str):
            if dim in axes:
                result.append(axes[dim])
            elif dim in dims:
                result.append(dims[dim])
            else:
                return None
        else:
            result.append(dim)
    return tuple(result)

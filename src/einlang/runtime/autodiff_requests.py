"""Thin runtime execution for internal autodiff requests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir.nodes import BindingIR, IdentifierIR
from ..passes.autodiff import AutodiffPass
from ..passes.autodiff.compiler import binding_for_defid, compile_autodiff_graph
from ..shared.defid import DefId


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _shape_of_value(value: Any) -> Tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _size_of_shape(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 1
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _basis(shape: Tuple[int, ...], flat_index: int) -> np.ndarray:
    out = np.zeros(shape or (), dtype=np.float64)
    out.reshape(-1)[flat_index] = 1.0
    return out


def _zeros_like(value: Any) -> np.ndarray:
    return np.zeros_like(_as_array(value))


def _ones_like(value: Any) -> np.ndarray:
    return np.ones_like(_as_array(value))

class LazyJacobianValue:
    """Lazy Jacobian materialized on demand via runtime JVP/VJP."""

    preserve_runtime_output_lazy = True

    def __init__(self, engine: "AutodiffRuntimeEngine", target_defid: DefId, wrt_defid: DefId):
        self._engine = engine
        self._target_defid = target_defid
        self._wrt_defid = wrt_defid
        self._materialized: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        target_shape = self._engine.target_shape(self._target_defid)
        wrt_shape = self._engine.target_shape(self._wrt_defid)
        return tuple(target_shape) + tuple(wrt_shape)

    def mode(self) -> str:
        out_shape = self._engine.target_shape(self._target_defid)
        wrt_shape = self._engine.target_shape(self._wrt_defid)
        return "jvp" if _size_of_shape(wrt_shape) <= _size_of_shape(out_shape) else "vjp"

    def materialize(self) -> np.ndarray:
        if self._materialized is not None:
            return self._materialized
        binding = self._engine._binding(self._target_defid)
        if binding is None or binding.expr is None:
            return np.zeros(self.shape)
        original_wrt = self._engine._current_value(self._wrt_defid)
        original_target = self._engine._current_value(self._target_defid)
        jacobian = np.zeros(self.shape, dtype=np.float64)
        epsilon = 1e-6
        wrt_shape = self._engine.target_shape(self._wrt_defid)
        target_shape = self._engine.target_shape(self._target_defid)
        for wrt_index in np.ndindex(*wrt_shape):
            perturbed_wrt = original_wrt.copy()
            perturbed_wrt[wrt_index] += epsilon
            perturbed_target = self._engine._evaluate_expr(binding.expr, {self._wrt_defid: perturbed_wrt})
            diff = (_as_array(perturbed_target) - _as_array(original_target)) / epsilon
            for target_index in np.ndindex(*target_shape):
                jacobian[target_index + wrt_index] = diff[target_index]
        self._materialized = jacobian
        return self._materialized

    def row(self, output_index: Tuple[int, ...]) -> Any:
        out_shape = self._engine.target_shape(self._target_defid)
        cotangent = _basis(out_shape, np.ravel_multi_index(output_index, out_shape) if out_shape else 0)
        return self._engine.eval_vjp(self._target_defid, self._wrt_defid, cotangent)

    def column(self, wrt_index: Tuple[int, ...]) -> Any:
        wrt_shape = self._engine.target_shape(self._wrt_defid)
        tangent = _basis(wrt_shape, np.ravel_multi_index(wrt_index, wrt_shape) if wrt_shape else 0)
        return self._engine.eval_jvp(self._target_defid, self._wrt_defid, tangent)

    def entry(self, output_index: Tuple[int, ...], wrt_index: Tuple[int, ...]) -> float:
        if self.mode() == "jvp":
            return float(np.asarray(self.column(wrt_index))[output_index])
        return float(np.asarray(self.row(output_index))[wrt_index])

    def __array__(self, dtype=None):
        arr = self.materialize()
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if Ellipsis in item or len(item) != len(self.shape):
            return self.materialize()[item]

        out_shape = self._engine.target_shape(self._target_defid)
        wrt_shape = self._engine.target_shape(self._wrt_defid)
        out_rank = len(out_shape)
        out_key = item[:out_rank]
        wrt_key = item[out_rank:]
        out_all_int = all(isinstance(k, (int, np.integer)) for k in out_key)
        wrt_all_int = all(isinstance(k, (int, np.integer)) for k in wrt_key)

        if out_all_int and wrt_all_int:
            return self.entry(tuple(int(k) for k in out_key), tuple(int(k) for k in wrt_key))
        if out_all_int:
            return np.asarray(self.row(tuple(int(k) for k in out_key)))[wrt_key]
        if wrt_all_int:
            return np.asarray(self.column(tuple(int(k) for k in wrt_key)))[out_key]
        return self.materialize()[item]

    def tolist(self):
        return self.materialize().tolist()

    def __repr__(self) -> str:
        return f"LazyJacobianValue(shape={self.shape}, mode={self.mode()})"


class AutodiffRuntimeEngine:
    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._compiled: Optional[Dict[str, Any]] = None
        self._scalar_vjp_cache: Dict[DefId, Dict[DefId, Any]] = {}
        self._generic_vjp_templates: Dict[Tuple[DefId, DefId], Dict[str, Any]] = {}
        self._generic_reachable_order: Dict[DefId, List[DefId]] = {}
        self._generic_template_counter = 0

    def _analysis(self) -> Dict[str, Any]:
        tcx = getattr(self._backend, "_tcx", None)
        if tcx is None:
            raise RuntimeError("runtime autodiff requires compiler TyCtxt")
        return tcx.get_analysis(AutodiffPass)

    def _compiled_graph(self) -> Dict[str, Any]:
        if self._compiled is None:
            self._compiled = compile_autodiff_graph(self._analysis())
        return self._compiled

    def _binding(self, defid: DefId) -> Optional[BindingIR]:
        return binding_for_defid(self._compiled_graph(), defid)

    def _current_value(self, defid: DefId) -> Any:
        value = self._backend.env.get_value(defid)
        if value is None:
            raise RuntimeError(f"runtime autodiff value not found for {defid}")
        return value

    def target_shape(self, defid: DefId) -> Tuple[int, ...]:
        return _shape_of_value(self._current_value(defid))

    def _evaluate_expr(
        self,
        expr: Any,
        extra_env: Optional[Dict[DefId, Any]] = None,
        *,
        context_defid: Optional[DefId] = None,
    ) -> Any:
        context_bindings = (self._compiled_graph().get("local_contexts_by_defid") or {}).get(context_defid) or {}
        with self._backend.env.scope():
            for defid, value in (extra_env or {}).items():
                self._backend.env.set_value(defid, value)
            for local_defid, binding in context_bindings.items():
                if local_defid is None or binding is None or getattr(binding, "expr", None) is None:
                    continue
                value = binding.expr.accept(self._backend)
                self._backend.env.set_value(local_defid, value, name=getattr(binding, "name", None))
            return expr.accept(self._backend)

    def _reachable_bindings(self, target_defid: DefId) -> List[DefId]:
        compiled = self._compiled_graph()
        by_target = compiled.get("runtime_vjp_templates_by_target") or {}
        target_entry = by_target.get(target_defid) or {}
        order = list(target_entry.get("order") or ())
        if order:
            return order
        cached = self._generic_reachable_order.get(target_defid)
        if cached is not None:
            return list(cached)
        bindings = {}
        bindings.update(compiled.get("bindings") or {})
        bindings.update(compiled.get("functions") or {})
        from ..passes.autodiff.compiletime import _AutodiffBindingContext, _DependencyQueryCache

        binding_ctx = _AutodiffBindingContext(bindings=bindings, dep_cache=_DependencyQueryCache(bindings))
        seen: set[DefId] = set()
        order_out: List[DefId] = []

        def dfs(did: DefId) -> None:
            if did in seen:
                return
            seen.add(did)
            for dep in binding_ctx.deps_for(did):
                if dep in bindings:
                    dfs(dep)
            order_out.append(did)

        dfs(target_defid)
        self._generic_reachable_order[target_defid] = order_out
        return list(order_out)

    def _generic_vjp_template(self, source_defid: DefId, dep_defid: DefId) -> Dict[str, Any]:
        key = (source_defid, dep_defid)
        cached = self._generic_vjp_templates.get(key)
        if cached is not None:
            return cached
        bindings = {}
        bindings.update(self._compiled_graph().get("bindings") or {})
        bindings.update(self._compiled_graph().get("functions") or {})
        local_contexts = dict(self._compiled_graph().get("local_contexts_by_defid") or {})
        source_binding = bindings.get(source_defid)
        dep_binding = bindings.get(dep_defid)
        if source_binding is None or dep_binding is None:
            raise RuntimeError(f"runtime generic VJP template missing binding for edge {source_defid}->{dep_defid}")
        from ..passes.autodiff.runtime_requests import _PlainRequestLowerer
        from ..ir.nodes import VjpIR

        lowerer = _PlainRequestLowerer(bindings, local_contexts, getattr(getattr(self._backend, "_tcx", None), "resolver", None))
        loc = getattr(source_binding, "location", None)
        seed_defid = DefId(-991, self._generic_template_counter)
        self._generic_template_counter += 1
        cot_ident = IdentifierIR(
            "__autodiff_runtime_cotangent",
            loc,
            defid=seed_defid,
            type_info=getattr(getattr(source_binding, "expr", None), "type_info", None),
            shape_info=getattr(getattr(source_binding, "expr", None), "shape_info", None),
        )
        expr = lowerer._lower_vjp(
            VjpIR(
                target=IdentifierIR(
                    getattr(source_binding, "name", "?"),
                    loc,
                    defid=source_defid,
                    type_info=getattr(getattr(source_binding, "expr", None), "type_info", None),
                    shape_info=getattr(getattr(source_binding, "expr", None), "shape_info", None),
                ),
                wrt=IdentifierIR(
                    getattr(dep_binding, "name", "?"),
                    loc,
                    defid=dep_defid,
                    type_info=getattr(getattr(dep_binding, "expr", None), "type_info", None),
                    shape_info=getattr(getattr(dep_binding, "expr", None), "shape_info", None),
                ),
                location=loc,
                type_info=getattr(getattr(dep_binding, "expr", None), "type_info", None),
                shape_info=getattr(getattr(dep_binding, "expr", None), "shape_info", None),
                cotangent=cot_ident,
            ),
            cotangent_expr=cot_ident,
        )
        template = {"expr": expr, "seed_defid": seed_defid, "context_defid": source_defid}
        self._generic_vjp_templates[key] = template
        return template

    def _accumulate(self, existing: Optional[Any], contribution: Any) -> Any:
        if existing is None:
            return contribution
        return _as_array(existing) + _as_array(contribution)

    def _edge_contribution_via_jvp(self, edge: Dict[str, Any], current_bar: Any, dep_defid: DefId) -> Any:
        dep_shape = self.target_shape(dep_defid)
        dep_size = _size_of_shape(dep_shape)
        current_bar_arr = _as_array(current_bar)
        contribution = np.zeros(dep_shape or (), dtype=np.float64)
        for flat_index in range(dep_size):
            tangent = _basis(dep_shape, flat_index)
            jvp_val = _as_array(
                self._evaluate_expr(
                    edge["jvp_expr"],
                    {edge["jvp_seed_defid"]: tangent},
                    context_defid=edge.get("context_defid"),
                )
            )
            coeff = float(np.sum(current_bar_arr * jvp_val))
            contribution.reshape(-1)[flat_index] = coeff
        return contribution

    def _edge_vjp_template(self, target_defid: DefId, dep_defid: DefId) -> Dict[str, Any]:
        compiled = self._compiled_graph()
        by_target = compiled.get("runtime_vjp_templates_by_target") or {}
        target_entry = by_target.get(target_defid) or {}
        edge = (target_entry.get("edges") or {}).get((target_defid, dep_defid))
        if edge is None:
            edge = (target_entry.get("edges") or {}).get((target_defid, dep_defid))
        if edge is None:
            raise RuntimeError(f"runtime VJP template missing for edge {target_defid}->{dep_defid}")
        return edge

    def _jvp_template(self, target_defid: DefId, wrt_defid: DefId) -> Dict[str, Any]:
        compiled = self._compiled_graph()
        by_pair = compiled.get("runtime_jvp_templates_by_pair") or {}
        template = by_pair.get((target_defid, wrt_defid))
        if template is None:
            raise RuntimeError(f"runtime JVP template missing for pair {(target_defid, wrt_defid)}")
        return template

    def _reverse_vjp_all(self, target_defid: DefId, cotangent_value: Any) -> Dict[DefId, Any]:
        bars: Dict[DefId, Any] = {target_defid: cotangent_value}
        compiled = self._compiled_graph()
        runtime_templates = (compiled.get("runtime_vjp_templates_by_target") or {}).get(target_defid) or {}
        runtime_edges = runtime_templates.get("edges") or {}
        reachable = self._reachable_bindings(target_defid)
        bindings = {}
        bindings.update(compiled.get("bindings") or {})
        bindings.update(compiled.get("functions") or {})
        for did in reversed(reachable):
            current_bar = bars.get(did)
            if current_bar is None:
                continue
            if runtime_edges:
                for edge_key, edge in runtime_edges.items():
                    source_did, dep_did = edge_key
                    if source_did != did:
                        continue
                    if edge.get("jvp_expr") is not None:
                        contribution = self._edge_contribution_via_jvp(edge, current_bar, dep_did)
                    else:
                        contribution = self._evaluate_expr(
                            edge["expr"],
                            {edge["seed_defid"]: current_bar},
                            context_defid=edge.get("context_defid"),
                        )
                    bars[dep_did] = self._accumulate(bars.get(dep_did), contribution)
                continue
            for dep_did in [dep for dep in reachable if dep in bindings and dep != did]:
                if dep_did not in (self._generic_reachable_order.get(target_defid) or []):
                    continue
            from ..passes.autodiff.compiletime import _AutodiffBindingContext, _DependencyQueryCache
            binding_ctx = _AutodiffBindingContext(bindings=bindings, dep_cache=_DependencyQueryCache(bindings))
            for dep_did in binding_ctx.deps_for(did):
                if dep_did not in bindings:
                    continue
                edge = self._generic_vjp_template(did, dep_did)
                contribution = self._evaluate_expr(
                    edge["expr"],
                    {edge["seed_defid"]: current_bar},
                    context_defid=edge.get("context_defid"),
                )
                bars[dep_did] = self._accumulate(bars.get(dep_did), contribution)
        return bars

    def eval_vjp(self, target_defid: DefId, wrt_defid: DefId, cotangent_value: Optional[Any] = None) -> Any:
        target_value = self._current_value(target_defid)
        if cotangent_value is None:
            cotangent_value = _ones_like(target_value)
            if _shape_of_value(target_value) == ():
                cotangent_value = float(np.asarray(cotangent_value).reshape(-1)[0])
        if _shape_of_value(target_value) == () and cotangent_value is not None and np.asarray(cotangent_value).shape == ():
            cached = self._scalar_vjp_cache.get(target_defid)
            if cached is None:
                cached = self._reverse_vjp_all(target_defid, cotangent_value)
                self._scalar_vjp_cache[target_defid] = cached
            if wrt_defid in cached:
                return cached[wrt_defid]
        bars = self._reverse_vjp_all(target_defid, cotangent_value)
        if wrt_defid in bars:
            return bars[wrt_defid]
        return _zeros_like(self._current_value(wrt_defid))

    def eval_jvp(self, target_defid: DefId, wrt_defid: DefId, tangent_value: Optional[Any] = None) -> Any:
        if tangent_value is None:
            tangent_value = _ones_like(self._current_value(wrt_defid))
            if _shape_of_value(self._current_value(wrt_defid)) == ():
                tangent_value = float(np.asarray(tangent_value).reshape(-1)[0])
        template = self._jvp_template(target_defid, wrt_defid)
        return self._evaluate_expr(
            template["expr"],
            {template["seed_defid"]: tangent_value},
            context_defid=template.get("context_defid"),
        )

    def execute_builtin(self, kind: Any, expr: Any) -> Any:
        args = list(getattr(expr, "args", ()) or ())
        if len(args) < 2:
            raise RuntimeError(f"{getattr(expr, 'builtin_name', '?')} requires target and wrt identifiers")
        target = args[0]
        wrt = args[1]
        if not isinstance(target, IdentifierIR) or target.defid is None:
            raise RuntimeError("autodiff target must be an identifier")
        if not isinstance(wrt, IdentifierIR) or wrt.defid is None:
            raise RuntimeError("autodiff wrt must be an identifier")
        if str(kind.name) == "JVP":
            tangent_value = args[2].accept(self._backend) if len(args) > 2 else None
            return self.eval_jvp(target.defid, wrt.defid, tangent_value)
        if str(kind.name) == "VJP":
            cotangent_value = args[2].accept(self._backend) if len(args) > 2 else None
            return self.eval_vjp(target.defid, wrt.defid, cotangent_value)
        if str(kind.name) == "LAZY_JACOBIAN":
            return LazyJacobianValue(self, target.defid, wrt.defid)
        raise RuntimeError(f"unsupported autodiff builtin kind: {kind}")

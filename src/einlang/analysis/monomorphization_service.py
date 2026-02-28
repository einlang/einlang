"""
Monomorphization service (DefId-based IR passes).

DefId-oriented design:
- Lookup: generic_defid + param_types -> specialized_defid
- Registry: (generic_defid, normalized_param_types) -> specialized BindingIR
- incremental_monomorphize -> monomorphize_if_needed -> _fully_specialize / _partially_specialize*
- _create_and_analyze: clone, register, run passes, add to pending

IR (BindingIR, FunctionCallIR), TyCtxt, tcx.function_ir_map, tcx.resolver.
    Mono runs passes up to type only (range, shape, type). Einstein_lowering runs in the
    main pipeline after all mono calls.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..shared.defid import DefId, DefType
from ..ir.nodes import BindingIR, FunctionCallIR, IRNode, ParameterIR, is_function_binding
from ..shared.types import Type
from ..passes.base import TyCtxt

logger = logging.getLogger("einlang.analysis.monomorphization_service")


# ---------------------------------------------------------------------------
# Instance (specialization key = generic_defid + param_types)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    generic_defid: DefId
    param_types: Tuple[Type, ...]

    def __str__(self) -> str:
        return f"Instance({self.generic_defid}, ({', '.join(str(t) for t in self.param_types)}))"


# ---------------------------------------------------------------------------
# MonomorphizationService
# ---------------------------------------------------------------------------

class MonomorphizationService:
    def __init__(self, tcx: TyCtxt) -> None:
        self.tcx = tcx
        self._registry: Dict[Instance, BindingIR] = {}
        self._by_defid: Dict[DefId, BindingIR] = {}
        self._analysis_cache: Dict[Tuple[DefId, Tuple[str, ...]], Any] = {}
        self._specialization_complexity: Dict[DefId, int] = {}
        self._max_complexity_per_function = 100
        self._pending_partial_specializations: Dict[DefId, List[BindingIR]] = {}
        self._monomorphizing: Set[DefId] = set()
        self._pending_specialized_functions: List[BindingIR] = []

    # ---------- Public API (aligned) ----------

    def can_incrementally_specialize(
        self, call: FunctionCallIR, arg_types: Tuple[Type, ...]
    ) -> bool:
        if not getattr(call, "function_defid", None):
            return False
        if not self._is_generic_function(call.function_defid):
            return False
        return (
            self._has_precision_and_rank(arg_types)
            or self._has_precision_only(arg_types)
            or self._has_rank_only(arg_types)
        )

    def incremental_monomorphize(
        self,
        call: FunctionCallIR,
        arg_types: Tuple[Type, ...],
        current_pass_name: str,
        required_passes: Optional[List[str]] = None,
    ) -> Optional[BindingIR]:
        if not self.can_incrementally_specialize(call, arg_types):
            return None
        generic_defid = call.function_defid
        if not generic_defid:
            return None
        if not self._track_complexity(generic_defid):
            return None
        normalized = self._normalize_types_for_instance(arg_types)
        instance = Instance(generic_defid=generic_defid, param_types=normalized)
        if instance in self._registry:
            existing = self._registry[instance]
            if required_passes:
                self._run_required_passes(existing, required_passes)
            if call is not None and existing and getattr(existing, "defid", None):
                call.set_callee_defid(existing.defid)
            return existing
        if generic_defid in self._monomorphizing:
            return None
        self._monomorphizing.add(generic_defid)
        try:
            out = self.monomorphize_if_needed(call, arg_types, required_passes)
            if out and (
                self._has_precision_only(arg_types) or self._has_rank_only(arg_types)
            ):
                if generic_defid not in self._pending_partial_specializations:
                    self._pending_partial_specializations[generic_defid] = []
                self._pending_partial_specializations[generic_defid].append(out)
            if call is not None and out and getattr(out, "defid", None):
                call.set_callee_defid(out.defid)
            return out
        finally:
            self._monomorphizing.discard(generic_defid)

    def monomorphize_if_needed(
        self,
        call: FunctionCallIR,
        arg_types: Tuple[Type, ...],
        required_passes: Optional[List[str]] = None,
    ) -> Optional[BindingIR]:
        if not call.function_defid or not self._is_generic_function(call.function_defid):
            return None
        generic_defid = call.function_defid
        if self._has_precision_and_rank(arg_types):
            return self._fully_specialize(call, generic_defid, arg_types, required_passes)
        if self._has_precision_only(arg_types):
            return self._partially_specialize(
                call, generic_defid, arg_types, required_passes
            )
        if self._has_rank_only(arg_types):
            return self._partially_specialize_rank_only(
                call, generic_defid, arg_types, required_passes
            )
        return None

    def get_pending_specialized_functions(self) -> List[BindingIR]:
        return self._pending_specialized_functions.copy()

    def clear_pending_specialized_functions(self) -> None:
        self._pending_specialized_functions.clear()

    def rewrite_calls_in_specialized_bodies(self) -> None:
        specialized_list = getattr(self.tcx, "specialized_functions", [])
        for func in specialized_list:
            body = getattr(func, "body", None)
            if body is None:
                continue
            self._rewrite_calls_in_node(body, set(), enclosing_function=func)

    def rewrite_calls_in_statements(self, statements: List[Any]) -> None:
        for stmt in statements or []:
            self._rewrite_calls_in_node(stmt, set(), enclosing_function=None)

    def get_specialized_defid_for_call(
        self,
        call: FunctionCallIR,
        enclosing_function: Optional[BindingIR] = None,
    ) -> Optional[DefId]:
        if not getattr(call, "function_defid", None):
            return None
        generic_defid = call.function_defid
        if not self._is_generic_function(generic_defid):
            return None
        from ..shared.types import UNKNOWN
        args_list = getattr(call, "arguments", []) or []
        params_list = (
            getattr(enclosing_function, "parameters", []) or []
            if enclosing_function
            else []
        )
        arg_types: List[Any] = []
        for i, a in enumerate(args_list):
            t = getattr(a, "type_info", None)
            if (t is None or t is UNKNOWN) and enclosing_function and i < len(
                params_list
            ):
                p = params_list[i]
                t = getattr(p, "param_type", None)
            arg_types.append(t)
        if not arg_types or any(t is None for t in arg_types):
            return None
        return self.get_specialized_defid(generic_defid, tuple(arg_types))

    def get_specialized_defid(
        self, generic_defid: DefId, param_types: Tuple[Type, ...]
    ) -> Optional[DefId]:
        normalized = self._normalize_types_for_instance(param_types)
        instance = Instance(generic_defid=generic_defid, param_types=normalized)
        spec = self._registry.get(instance)
        return getattr(spec, "defid", None) if spec else None

    def get_by_defid(self, defid: DefId) -> Optional[BindingIR]:
        return self._by_defid.get(defid)

    def get_generic_defid_for_specialized(self, specialized_defid: DefId) -> Optional[DefId]:
        for inst, fn in self._registry.items():
            if getattr(fn, "defid", None) == specialized_defid:
                return inst.generic_defid
        return None

    def complete_partial_specializations(
        self, improved_arg_types: Dict[DefId, Tuple[Type, ...]]
    ) -> int:
        completed = 0
        for generic_defid, complete_arg_types in improved_arg_types.items():
            if generic_defid not in self._pending_partial_specializations:
                continue
            if not self._has_precision_and_rank(complete_arg_types):
                continue
            self._create_and_analyze(
                None,
                generic_defid,
                complete_arg_types,
                ["type", "range"],
                is_partial=False,
            )
            completed += 1
            del self._pending_partial_specializations[generic_defid]
        return completed

    def unify_local_var_defids_in_program(self, program: "ProgramIR") -> None:
        pass

    def _create_specialized_function(
        self,
        generic_func: BindingIR,
        generic_defid: DefId,
        arg_types: Tuple[Type, ...],
        current_pass_name: str,
        required_passes: Optional[List[str]] = None,
    ) -> Optional[BindingIR]:
        return self._create_and_analyze(
            None, generic_defid, arg_types, required_passes or [], is_partial=False
        )

    # ---------- Generic check (_is_generic_call; resolver + IR) ----------

    def _is_generic_function(self, defid: DefId) -> bool:
        if not self.tcx.resolver:
            return False
        result = self.tcx.resolver.query(defid)
        if not result:
            return False
        def_type, definition = result
        if def_type != DefType.FUNCTION:
            return False
        from ..shared.nodes import FunctionDefinition
        if is_function_binding(definition) or isinstance(definition, FunctionDefinition):
            for param in definition.parameters:
                param_type = getattr(param, "param_type", None) or getattr(
                    param, "type_annotation", None
                )
                if param_type is None:
                    return True
                from ..shared.types import RectangularType
                if isinstance(param_type, RectangularType):
                    if getattr(param_type, "is_dynamic_rank", False):
                        return True
        return False

    # ---------- Specialize dispatch (_fully_specialize, _partially_specialize*) ----------

    def _fully_specialize(
        self,
        call: FunctionCallIR,
        generic_defid: DefId,
        arg_types: Tuple[Type, ...],
        required_passes: Optional[List[str]],
    ) -> Optional[BindingIR]:
        sid = self.get_specialized_defid(generic_defid, arg_types)
        if sid:
            spec = self.get_by_defid(sid)
            if spec and required_passes:
                self._run_required_passes(spec, required_passes)
            return spec
        return self._create_and_analyze(
            call, generic_defid, arg_types, required_passes, is_partial=False
        )

    def _partially_specialize(
        self,
        call: FunctionCallIR,
        generic_defid: DefId,
        arg_types: Tuple[Type, ...],
        required_passes: Optional[List[str]],
    ) -> Optional[BindingIR]:
        from ..shared.types import TypeKind, RectangularType
        partial: List[Type] = []
        for t in arg_types:
            if t is not None and getattr(t, "kind", None) == TypeKind.RECTANGULAR:
                el = getattr(t, "element_type", None)
                partial.append(
                    RectangularType(
                        element_type=el, shape=None, is_dynamic_rank=True
                    )
                )
            else:
                partial.append(t)
        pt = tuple(partial)
        sid = self.get_specialized_defid(generic_defid, pt)
        if sid:
            spec = self.get_by_defid(sid)
            if spec and required_passes:
                self._run_required_passes(spec, required_passes)
            return spec
        return self._create_and_analyze(
            call, generic_defid, pt, required_passes, is_partial=True
        )

    def _partially_specialize_rank_only(
        self,
        call: FunctionCallIR,
        generic_defid: DefId,
        arg_types: Tuple[Type, ...],
        required_passes: Optional[List[str]],
    ) -> Optional[BindingIR]:
        from ..shared.types import TypeKind, RectangularType, UNKNOWN
        partial: List[Type] = []
        for t in arg_types:
            if t is not None and getattr(t, "kind", None) == TypeKind.RECTANGULAR:
                shape = getattr(t, "shape", None)
                partial.append(
                    RectangularType(
                        element_type=UNKNOWN, shape=shape, is_dynamic_rank=False
                    )
                )
            else:
                partial.append(t)
        pt = tuple(partial)
        sid = self.get_specialized_defid(generic_defid, pt)
        if sid:
            spec = self.get_by_defid(sid)
            if spec and required_passes:
                self._run_required_passes(spec, required_passes)
            return spec
        return self._create_and_analyze(
            call, generic_defid, pt, required_passes, is_partial=True
        )

    # ---------- Create and analyze (_create_and_analyze) ----------

    def _create_and_analyze(
        self,
        call: Optional[FunctionCallIR],
        generic_defid: DefId,
        arg_types: Tuple[Type, ...],
        required_passes: Optional[List[str]],
        is_partial: bool,
    ) -> Optional[BindingIR]:
        function_ir_map = getattr(self.tcx, "function_ir_map", None) or {}
        generic_func = function_ir_map.get(generic_defid)
        if not generic_func or not is_function_binding(generic_func):
            return None
        module_path = getattr(generic_func, "module_path", ("__specialized",))
        if not isinstance(module_path, tuple):
            module_path = ("__specialized",)
        type_strs = [str(t) for t in arg_types]
        display_name = f"{generic_func.name}_{'_'.join(type_strs)}"
        specialized_defid = self.tcx.resolver.allocate_for_item(
            module_path, display_name, None, DefType.FUNCTION
        )
        specialized_func = self._clone_and_specialize(
            generic_func, arg_types, specialized_defid
        )
        if not specialized_func:
            return None
        self.tcx.def_registry[specialized_defid] = (DefType.FUNCTION, specialized_func)
        normalized = self._normalize_types_for_instance(arg_types)
        instance = Instance(generic_defid=generic_defid, param_types=normalized)
        self._registry[instance] = specialized_func
        self._by_defid[specialized_defid] = specialized_func
        if specialized_func and getattr(specialized_func, "body", None):
            self._substitute_call_targets_in_body(
                specialized_func.body,
                generic_defid,
                specialized_func.defid,
                set(),
            )
        if required_passes:
            specialized_func = self._run_passes(specialized_func, required_passes)
        if is_partial:
            object.__setattr__(specialized_func, "_is_partially_specialized", True)
            object.__setattr__(specialized_func, "_generic_defid", generic_defid)
        self._pending_specialized_functions.append(specialized_func)
        if not hasattr(self.tcx, "specialized_functions"):
            self.tcx.specialized_functions = []
        self.tcx.specialized_functions.append(specialized_func)
        if specialized_func.defid and hasattr(self.tcx, "function_ir_map"):
            self.tcx.function_ir_map[specialized_func.defid] = specialized_func
        return specialized_func

    def _run_required_passes(
        self, specialized_func: BindingIR, required_passes: Optional[List[str]]
    ) -> None:
        if not required_passes:
            return
        already: Set[str] = set()
        for cache_key in self._analysis_cache:
            if cache_key[0] == specialized_func.defid:
                already.update(cache_key[1])
        missing = [p for p in required_passes if p not in already]
        if missing:
            self._run_passes(specialized_func, missing)

    def _clone_and_specialize(
        self,
        generic_func: BindingIR,
        arg_types: Tuple[Type, ...],
        specialized_defid: DefId,
    ) -> Optional[BindingIR]:
        spec = copy.deepcopy(generic_func)
        normalized = self._normalize_types_for_instance(arg_types)
        spec.name = f"{generic_func.name}_{'_'.join(str(t) for t in normalized)}"
        if len(spec.parameters) != len(arg_types):
            return None
        param_types = self._create_dynamic_shape_types(arg_types)
        for param, pt in zip(spec.parameters, param_types):
            if isinstance(param, ParameterIR):
                object.__setattr__(param, "param_type", pt)
        object.__setattr__(spec, "defid", specialized_defid)
        return spec

    # ---------- DCE on specialized bodies ----------

    def _dce_specialized_body(self, func: BindingIR) -> None:
        """Dead code elimination on a specialized function body.

        Uses the visitor pattern (DCEVisitor) to evaluate len(param.shape) → rank,
        fold constant if-conditions, and prune dead branches.
        Only mutates the already-cloned specialized copy.

        The visitor derives parameter ranks from param_type on ParameterIR
        nodes (set during monomorphization) — no external map needed.  Scoped
        constant tracking handles nested blocks correctly.
        """
        from .dce_visitor import DCEVisitor

        try:
            visitor = DCEVisitor()
            func.accept(visitor)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"DCE failed for {func.name}: {e}")

    # ---------- Run passes (_run_passes; ProgramIR + pass instances) ----------

    def _run_passes(
        self, specialized_func: BindingIR, passes: List[str]
    ) -> BindingIR:
        from ..ir.nodes import ProgramIR
        from ..shared.source_location import SourceLocation
        cache_key = (specialized_func.defid, tuple(passes))
        if cache_key in self._analysis_cache:
            if self._analysis_cache[cache_key] == "running":
                return specialized_func
            return specialized_func
        self._analysis_cache[cache_key] = "running"
        mini = ProgramIR(
            statements=[specialized_func],
            modules=[],
            source_files={},
            defid_to_name=None,
        )
        isolated = TyCtxt()
        isolated.resolver = self.tcx.resolver
        isolated.path_resolver = self.tcx.path_resolver
        isolated.module_loader = self.tcx.module_loader
        isolated.symbol_linker = self.tcx.symbol_linker
        if getattr(self.tcx, "reporter", None):
            isolated.reporter = self.tcx.reporter
        isolated.monomorphization_service = self
        if getattr(self.tcx, "function_ir_map", None) is not None:
            isolated.function_ir_map = self.tcx.function_ir_map
        isolated.specialized_functions = [specialized_func]
        pass_map = {
            "einstein_grouping": self._get_einstein_grouping_pass,
            "constraint_classifier": self._get_constraint_classifier_pass,
            "rest_pattern": self._get_rest_pattern_pass,
            "range": self._get_range_analysis_pass,
            "shape": self._get_shape_analysis_pass,
            "type": self._get_type_inference_pass,
            "einstein_lowering": self._get_einstein_lowering_pass,
        }
        full_order = [
            "einstein_grouping",
            "constraint_classifier",
            "rest_pattern",
            "range",
            "shape",
            "type",
            "einstein_lowering",
        ]
        # DCE: prune dead if-else branches on the specialized copy.
        # Evaluates len(param.shape) → constant rank, then folds constant conditions.
        if getattr(specialized_func, 'body', None):
            self._dce_specialized_body(specialized_func)

        if len(passes) == 1 and passes[0] in ("range", "shape"):
            pass_order = passes
        elif passes:
            indices = [full_order.index(p) for p in passes if p in full_order]
            pass_order = full_order[: max(indices) + 1] if indices else []
        else:
            pass_order = []
        for pass_name in pass_order:
            factory = pass_map.get(pass_name)
            if not factory:
                continue
            try:
                instance = factory()
                result_ir = instance.run(mini, isolated)
                if result_ir.functions:
                    new_func = result_ir.functions[0]
                    if new_func is not specialized_func and getattr(new_func, "defid", None) == getattr(specialized_func, "defid", None):
                        object.__setattr__(specialized_func.expr, "body", getattr(new_func, "body", specialized_func.body))
                        if hasattr(specialized_func, "return_type"):
                            object.__setattr__(specialized_func.expr, "return_type", getattr(new_func, "return_type", specialized_func.return_type))
                        object.__setattr__(mini, 'statements', [specialized_func])
                        object.__setattr__(mini, '_bindings', [specialized_func])
                    else:
                        if new_func is not specialized_func and new_func.defid:
                            for inst, fn in list(self._registry.items()):
                                if getattr(fn, "defid", None) == new_func.defid:
                                    self._registry[inst] = new_func
                                    break
                            self._by_defid[new_func.defid] = new_func
                            if hasattr(self.tcx, "function_ir_map") and self.tcx.function_ir_map:
                                self.tcx.function_ir_map[new_func.defid] = new_func
                        specialized_func = new_func
                    for extra in result_ir.functions[1:]:
                        self._pending_specialized_functions.append(extra)
                        if hasattr(self.tcx, "specialized_functions"):
                            self.tcx.specialized_functions.append(extra)
                        if extra.defid and hasattr(self.tcx, "function_ir_map"):
                            self.tcx.function_ir_map[extra.defid] = extra
                    if pass_name == "type":
                        iso_mono = getattr(
                            isolated, "monomorphization_service", None
                        )
                        if iso_mono:
                            for extra in iso_mono.get_pending_specialized_functions():
                                self._pending_specialized_functions.append(extra)
                                if hasattr(self.tcx, "specialized_functions"):
                                    self.tcx.specialized_functions.append(extra)
                                if extra.defid and hasattr(
                                    self.tcx, "function_ir_map"
                                ):
                                    self.tcx.function_ir_map[extra.defid] = extra
                            iso_mono.clear_pending_specialized_functions()
                    object.__setattr__(mini, 'statements', [specialized_func])
                    object.__setattr__(mini, '_bindings', [specialized_func])
            except Exception as e:
                logger.warning("Pass %s failed for %s: %s", pass_name, specialized_func.name, e)
        self._analysis_cache[cache_key] = True
        if specialized_func.defid:
            self._by_defid[specialized_func.defid] = specialized_func
            if hasattr(self.tcx, "function_ir_map") and self.tcx.function_ir_map is not None:
                self.tcx.function_ir_map[specialized_func.defid] = specialized_func
        return specialized_func

    def _get_einstein_grouping_pass(self):
        from ..passes.einstein_grouping import EinsteinDeclarationGroupingPass
        return EinsteinDeclarationGroupingPass()

    def _get_constraint_classifier_pass(self):
        from ..passes.constraint_classifier import ConstraintClassifierPass
        return ConstraintClassifierPass()

    def _get_rest_pattern_pass(self):
        from ..passes.rest_pattern_preprocessing import RestPatternPreprocessingPass
        return RestPatternPreprocessingPass()

    def _get_range_analysis_pass(self):
        from ..passes.range_analysis import RangeAnalysisPass
        return RangeAnalysisPass()

    def _get_shape_analysis_pass(self):
        from ..passes.shape_analysis import UnifiedShapeAnalysisPass
        return UnifiedShapeAnalysisPass()

    def _get_type_inference_pass(self):
        from ..passes.type_inference import TypeInferencePass
        return TypeInferencePass()

    def _get_einstein_lowering_pass(self):
        from ..passes.einstein_lowering import EinsteinLoweringPass
        return EinsteinLoweringPass()

    # ---------- Type helpers (_has_*, normalize) ----------

    def _track_complexity(self, generic_defid: DefId) -> bool:
        n = self._specialization_complexity.get(generic_defid, 0) + 1
        self._specialization_complexity[generic_defid] = n
        if n > self._max_complexity_per_function:
            logger.warning(
                "Complexity limit exceeded for %s: %s specializations",
                generic_defid,
                n,
            )
            return False
        return True

    def _has_precision_and_rank(self, arg_types: Tuple[Type, ...]) -> bool:
        from ..shared.types import TypeKind, UNKNOWN, RectangularType
        for t in arg_types:
            if t is None or t is UNKNOWN:
                return False
            kind = getattr(t, "kind", None)
            if kind == TypeKind.RECTANGULAR:
                if not isinstance(t, RectangularType) or t.element_type is None:
                    return False
                if t.element_type is UNKNOWN:
                    return False
            elif kind != TypeKind.PRIMITIVE:
                return False
        return True

    def _has_precision_only(self, arg_types: Tuple[Type, ...]) -> bool:
        from ..shared.types import TypeKind, UNKNOWN, RectangularType
        for t in arg_types:
            if t is None or t is UNKNOWN:
                return False
            if t.kind == TypeKind.RECTANGULAR:
                if not isinstance(t, RectangularType):
                    return False
                if t.element_type is None or t.element_type is UNKNOWN:
                    return False
                if t.shape is not None and len(t.shape) > 0:
                    return False
            elif t.kind == TypeKind.PRIMITIVE:
                return False
        return True

    def _has_rank_only(self, arg_types: Tuple[Type, ...]) -> bool:
        from ..shared.types import TypeKind, UNKNOWN, RectangularType
        for t in arg_types:
            if t is None or t is UNKNOWN:
                continue
            if getattr(t, "kind", None) == TypeKind.RECTANGULAR and isinstance(t, RectangularType):
                has_rank = (
                    t.shape is not None and len(t.shape) > 0
                )
                has_precision = (
                    t.element_type is not None and t.element_type != UNKNOWN
                )
                if has_rank and not has_precision:
                    return True
        return False

    def _compute_rank(self, type_info: Any) -> int:
        from ..shared.types import RectangularType
        if not isinstance(type_info, RectangularType):
            return 0
        if getattr(type_info, "shape", None) is not None:
            return len(type_info.shape)
        rank = 0
        current = type_info
        while isinstance(current, RectangularType):
            rank += 1
            current = getattr(current, "element_type", None)
        return rank

    def _create_dynamic_shape_types(
        self, arg_types: Tuple[Type, ...]
    ) -> Tuple[Type, ...]:
        from ..shared.types import TypeKind, RectangularType
        result: List[Type] = []
        for t in arg_types:
            if t is None:
                result.append(t)
                continue
            if getattr(t, "kind", None) == TypeKind.RECTANGULAR and isinstance(t, RectangularType):
                rank = self._compute_rank(t)
                shape = tuple(None for _ in range(rank)) if rank > 0 else None
                el = self._create_dynamic_shape_types((t.element_type,))[0]
                result.append(
                    RectangularType(
                        element_type=el,
                        shape=shape,
                        is_dynamic_rank=False,
                    )
                )
            else:
                result.append(t)
        return tuple(result)

    def _normalize_types_for_instance(
        self, arg_types: Tuple[Type, ...]
    ) -> Tuple[Type, ...]:
        from ..shared.types import TypeKind, RectangularType
        out: List[Type] = []
        for t in arg_types:
            if t is None:
                out.append(t)
                continue
            if getattr(t, "kind", None) == TypeKind.RECTANGULAR and isinstance(t, RectangularType):
                el_norm = self._normalize_types_for_instance((t.element_type,))[0]
                shape = (
                    tuple(None for _ in t.shape) if t.shape else None
                )
                out.append(
                    RectangularType(
                        element_type=el_norm,
                        shape=shape,
                        is_dynamic_rank=getattr(t, "is_dynamic_rank", False),
                    )
                )
            else:
                out.append(t)
        return tuple(out)

    # ---------- IR helpers: unify, substitute, rewrite ----------

    def _collect_functions_from_module(
        self, mod: Any, out: List[BindingIR]
    ) -> None:
        from ..ir.nodes import ModuleIR
        if not isinstance(mod, ModuleIR):
            return
        out.extend(getattr(mod, "functions", None) or [])
        for sub in getattr(mod, "submodules", None) or []:
            self._collect_functions_from_module(sub, out)

    def _substitute_call_targets_in_body(
        self,
        node: Any,
        generic_defid: DefId,
        specialized_defid: DefId,
        visited: Set[int],
    ) -> None:
        from ..ir.nodes import FunctionCallIR, IRNode
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, FunctionCallIR) and getattr(
            node, "function_defid", None
        ) == generic_defid:
            node.set_callee_defid(specialized_defid)
        if isinstance(node, IRNode):
            for attr in getattr(node, "__slots__", ()) or []:
                if hasattr(node, attr):
                    self._substitute_call_targets_in_body(
                        getattr(node, attr),
                        generic_defid,
                        specialized_defid,
                        visited,
                    )
        elif hasattr(node, "__slots__") and not isinstance(
            node, (list, tuple, dict)
        ):
            for attr in node.__slots__:
                if hasattr(node, attr):
                    self._substitute_call_targets_in_body(
                        getattr(node, attr),
                        generic_defid,
                        specialized_defid,
                        visited,
                    )
        elif isinstance(node, (list, tuple)):
            for x in node:
                self._substitute_call_targets_in_body(
                    x, generic_defid, specialized_defid, visited
                )
        elif isinstance(node, dict):
            for x in node.values():
                self._substitute_call_targets_in_body(
                    x, generic_defid, specialized_defid, visited
                )
        elif hasattr(node, "__dict__"):
            for x in node.__dict__.values():
                self._substitute_call_targets_in_body(
                    x, generic_defid, specialized_defid, visited
                )

    def _rewrite_calls_in_node(
        self,
        node: Any,
        visited: Set[int],
        enclosing_function: Optional[BindingIR] = None,
    ) -> None:
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, FunctionCallIR):
            fd = getattr(node, "function_defid", None)
            sid = self.get_specialized_defid_for_call(node, enclosing_function)
            if sid is not None:
                node.set_callee_defid(sid)
            else:
                from ..shared.types import UNKNOWN
                args_list = getattr(node, "arguments", []) or []
                arg_types_list: List[Any] = []
                for a in args_list:
                    t = getattr(a, "type_info", None)
                    arg_types_list.append(t)
                all_known = bool(arg_types_list and all(t is not None and t is not UNKNOWN for t in arg_types_list))
                if not all_known and len(arg_types_list) == 2 and fd and self._is_generic_function(fd):
                    from ..shared.types import RectangularType, PrimitiveType
                    from ..shared.types import F32
                    if arg_types_list[1] is not None and arg_types_list[1] is not UNKNOWN and getattr(arg_types_list[1], "name", None) == "f32":
                        fill = RectangularType(element_type=F32, shape=None)
                        arg_types_list = [fill, arg_types_list[1]]
                        all_known = True
                is_generic = bool(fd and self._is_generic_function(fd))
                spec = None
                if all_known and is_generic:
                    spec = self.incremental_monomorphize(
                        node, tuple(arg_types_list), "rewrite", required_passes=["range", "type"]
                    )
                    if spec and getattr(spec, "defid", None):
                        node.set_callee_defid(spec.defid)
            for arg in getattr(node, "arguments", []) or []:
                self._rewrite_calls_in_node(arg, visited, enclosing_function)
            return
        if isinstance(node, IRNode):
            for cls in type(node).__mro__:
                for attr in getattr(cls, "__slots__", ()):
                    if hasattr(node, attr):
                        self._rewrite_calls_in_node(
                            getattr(node, attr), visited, enclosing_function
                        )
        elif hasattr(node, "__slots__") and not isinstance(
            node, (list, tuple, dict)
        ):
            for cls in type(node).__mro__:
                for attr in getattr(cls, "__slots__", ()):
                    if hasattr(node, attr):
                        self._rewrite_calls_in_node(
                            getattr(node, attr), visited, enclosing_function
                        )
        elif isinstance(node, (list, tuple)):
            for x in node:
                self._rewrite_calls_in_node(x, visited, enclosing_function)
        elif isinstance(node, dict):
            for x in node.values():
                self._rewrite_calls_in_node(x, visited, enclosing_function)
        elif hasattr(node, "__dict__"):
            for x in node.__dict__.values():
                self._rewrite_calls_in_node(x, visited, enclosing_function)

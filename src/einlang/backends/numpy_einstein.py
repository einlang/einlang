"""NumPy backend Einstein execution: variable decl, lowered einstein/clause; env only."""

from typing import Any, List, Optional

import numpy as np

from ..ir.nodes import (
    EinsteinDeclarationIR, LiteralIR, RangeIR, LoweredEinsteinIR, LoweredEinsteinClauseIR,
    LoweredReductionIR,
)
from ..shared.defid import DefId
from .numpy_helpers import _reject_non_lowered


class EinsteinExecutionMixin:
    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> Any:
        _reject_non_lowered(type(node).__name__)

    def visit_variable_declaration(self, node: Any) -> Any:
        if not (hasattr(node, "value") and node.value):
            return None
        stack = getattr(self, "_variable_decl_stack", None)
        if stack is None:
            self._variable_decl_stack = []
            stack = self._variable_decl_stack
        stack.append(node)
        try:
            return node.value.accept(self)
        finally:
            stack.pop()

    def visit_lowered_einstein_clause(self, node: LoweredEinsteinClauseIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein_clause(node, variable_decl)

    def visit_lowered_einstein(self, node: LoweredEinsteinIR) -> Any:
        stack = getattr(self, "_variable_decl_stack", None)
        variable_decl = stack[-1] if stack else None
        return self._execute_lowered_einstein(node, variable_decl)

    def visit_binding(self, node: Any) -> Any:
        return None

    def _primitive_type_to_numpy_dtype(self, type_obj: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType
        if not isinstance(type_obj, PrimitiveType):
            return None
        type_name = type_obj.name.lower()
        dtype_map = {
            "i32": np.int32, "i64": np.int64, "f32": np.float32, "f64": np.float64,
            "bool": np.bool_, "int": np.int32, "float": np.float32,
        }
        return dtype_map.get(type_name)

    def _type_info_to_numpy_dtype(self, type_info: Any) -> Optional[Any]:
        from ..shared.types import PrimitiveType, RectangularType, Type, TypeKind
        if type_info is None:
            return None
        if isinstance(type_info, Type) and hasattr(type_info, "kind") and type_info.kind == TypeKind.UNKNOWN:
            return None
        if isinstance(type_info, PrimitiveType):
            return self._primitive_type_to_numpy_dtype(type_info)
        if isinstance(type_info, RectangularType):
            return self._type_info_to_numpy_dtype(type_info.element_type)
        return None

    def _body_implies_float(self, body: Any) -> bool:
        from ..ir.nodes import FunctionCallIR, BinaryOpIR
        from ..shared.types import PrimitiveType
        if body is None:
            return False
        if isinstance(body, LoweredReductionIR):
            return self._body_implies_float(body.body)
        if isinstance(body, FunctionCallIR):
            name = (body.function_name or "").lower()
            if name in ("exp", "ln", "log", "sqrt", "sigmoid", "tanh"):
                return True
        if isinstance(body, BinaryOpIR):
            op = str(getattr(body, "operator", ""))
            if op in ("/", "DIV", "**"):
                return True
            for operand in (body.left, body.right):
                if operand is not None:
                    t = getattr(operand, "type_info", None)
                    if isinstance(t, PrimitiveType) and (t.name or "").lower() in ("f32", "f64", "float"):
                        return True
            if body.left and self._body_implies_float(body.left):
                return True
            if body.right and self._body_implies_float(body.right):
                return True
        return False

    def _get_defid_for_pattern_var(self, var_name: str, pattern: Any) -> Optional[DefId]:
        if hasattr(pattern, "name") and pattern.name == var_name:
            return pattern.defid
        if hasattr(pattern, "inner_pattern"):
            return self._get_defid_for_pattern_var(var_name, pattern.inner_pattern)
        if hasattr(pattern, "patterns"):
            for nested in pattern.patterns:
                defid = self._get_defid_for_pattern_var(var_name, nested)
                if defid is not None:
                    return defid
        return None

    def _execute_lowered_einstein(self, lowered_einstein: LoweredEinsteinIR, variable_decl: Any) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        binding = getattr(variable_decl, "_binding", None)
        variable_key = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)
        tensor_shape = lowered_einstein.shape
        tensor_element_type = lowered_einstein.element_type
        output_shape = None
        if tensor_shape:
            output_shape = []
            for shape_dim in tensor_shape:
                dim_value = shape_dim.accept(self)
                if isinstance(dim_value, (int, np.integer)):
                    output_shape.append(int(dim_value))
                elif isinstance(dim_value, np.ndarray) and dim_value.ndim == 0:
                    try:
                        output_shape.append(int(dim_value))
                    except (TypeError, ValueError):
                        output_shape = None
                        break
                else:
                    output_shape = None
                    break
        if not output_shape and lowered_einstein.items:
            output_shape = self._shape_from_all_items(lowered_einstein.items)
        elif output_shape and lowered_einstein.items:
            # Compiler shape may underestimate for multi-segment declarations
            # (e.g. _compute_shape_union picks a symbolic expr that evaluates to
            # the first clause's end, not the union).  Widen to cover all items.
            items_shape = self._shape_from_all_items(lowered_einstein.items)
            if items_shape and len(items_shape) == len(output_shape):
                output_shape = [max(a, b) for a, b in zip(output_shape, items_shape)]
        if not output_shape and lowered_einstein.items:
            raise RuntimeError(
                "Einstein declaration has no shape from compiler. "
                "Compiler must set shape (union of clause ranges) on LoweredEinsteinIR."
            )
        if not output_shape:
            output_shape = [1]
        dtype = self._type_info_to_numpy_dtype(tensor_element_type)
        if dtype is None and lowered_einstein.items:
            first = lowered_einstein.items[0]
            type_info = getattr(first.body, "type_info", None)
            if type_info is None and hasattr(first.body, "expr"):
                type_info = getattr(first.body.expr, "type_info", None)
            dtype = self._type_info_to_numpy_dtype(type_info)
        if dtype is None:
            dtype = np.int32

        # Multi-segment: reuse existing array if this variable was already
        # declared (e.g. pad's `let result[i in 0..p] = ...; let result[i in p..n] = ...;`)
        existing = self.env.get_value(variable_key) if variable_key is not None else None
        if existing is not None and isinstance(existing, np.ndarray):
            needed = tuple(output_shape)
            current = existing.shape
            if len(needed) == len(current) and any(n > c for n, c in zip(needed, current)):
                new_shape = tuple(max(n, c) for n, c in zip(needed, current))
                output = np.zeros(new_shape, dtype=existing.dtype)
                slices = tuple(slice(0, s) for s in current)
                output[slices] = existing
                self.env.set_value(variable_key, output)
            else:
                output = existing
        else:
            output = np.zeros(output_shape, dtype=dtype)
            if variable_key is not None:
                self.env.set_value(variable_key, output)

        for item in lowered_einstein.items:
            result = self._execute_lowered_einstein_clause(
                item, variable_decl,
                shape=tensor_shape, element_type=tensor_element_type,
                pre_allocated_output=output,
            )
            if result is not None and variable_key is not None:
                self.env.set_value(variable_key, result)
        return output

    def _shape_from_all_items(self, items: List) -> Optional[List[int]]:
        """Compute output shape from the max absolute end across ALL items (not just the first)."""
        if not items:
            return None
        rank = None
        max_ends: Optional[List[int]] = None
        for item in items:
            loops = getattr(item, "loops", None) or []
            if not loops:
                continue
            if rank is None:
                rank = len(loops)
                max_ends = [0] * rank
            if len(loops) != rank:
                return None
            for d, loop in enumerate(loops):
                it = getattr(loop, "iterable", None)
                if it is None:
                    return None
                if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
                    end = int(it.value.stop)
                elif isinstance(it, RangeIR):
                    try:
                        end = int(it.end.accept(self))
                    except (TypeError, ValueError):
                        return None
                else:
                    try:
                        r = it.accept(self)
                        end = len(r) if hasattr(r, "__len__") else None
                    except Exception:
                        return None
                    if end is None:
                        return None
                if end > max_ends[d]:
                    max_ends[d] = end
        return max_ends

    def _execute_lowered_einstein_clause(
        self,
        lowered: LoweredEinsteinClauseIR,
        variable_decl: Any,
        shape: Optional[List] = None,
        element_type: Optional[Any] = None,
        pre_allocated_output: Optional[Any] = None,
    ) -> Any:
        from ..runtime.compute.lowered_execution import execute_lowered_loops, execute_lowered_bindings, check_lowered_guards
        clause_indices = getattr(lowered, "indices", None) or []
        binding = getattr(variable_decl, "_binding", None)
        variable_defid = (getattr(binding, "defid", None) if binding else None) or getattr(variable_decl, "defid", None)

        def cell_index(full_context: dict) -> Optional[tuple]:
            out = []
            loop_pos = 0
            for idx in clause_indices:
                if isinstance(idx, LiteralIR):
                    try:
                        out.append(int(idx.value))
                    except (TypeError, ValueError):
                        break
                elif loop_pos < len(lowered.loops):
                    defid = getattr(lowered.loops[loop_pos].variable, "defid", None)
                    v = full_context.get(defid)
                    if v is None and defid is not None:
                        v = self.env.get_value(defid)
                    if v is None:
                        break
                    out.append(v)
                    loop_pos += 1
                else:
                    break
            return tuple(out) if len(out) == len(clause_indices) else None

        if pre_allocated_output is not None:
            output = pre_allocated_output
        else:
            output_shape = None
            if shape:
                output_shape = []
                for shape_dim in shape:
                    dim_value = shape_dim.accept(self)
                    if isinstance(dim_value, (int, np.integer)):
                        output_shape.append(int(dim_value))
                    elif isinstance(dim_value, np.ndarray) and dim_value.ndim == 0:
                        try:
                            output_shape.append(int(dim_value))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                    else:
                        output_shape = None
                        break
            if output_shape is None and lowered.loops:
                output_shape = []
                for loop in lowered.loops:
                    it = getattr(loop, "iterable", None)
                    if it is None:
                        output_shape = None
                        break
                    if isinstance(it, LiteralIR) and isinstance(getattr(it, "value", None), range):
                        output_shape.append(int(it.value.stop))
                    elif isinstance(it, RangeIR):
                        try:
                            end_val = int(it.end.accept(self))
                        except (TypeError, ValueError):
                            output_shape = None
                            break
                        output_shape.append(end_val)
                    else:
                        try:
                            r = it.accept(self)
                            output_shape.append(len(r) if hasattr(r, "__len__") else None)
                        except Exception:
                            output_shape = None
                        if output_shape and output_shape[-1] is None:
                            output_shape = None
                            break
            if output_shape is None:
                output_shape = [int(getattr(idx, "value", 0)) + 1 if isinstance(idx, LiteralIR) else 1 for idx in clause_indices] if clause_indices else [1]
            dtype = self._type_info_to_numpy_dtype(element_type)
            if dtype is None:
                dtype = getattr(lowered.body, "type_info", None) and self._type_info_to_numpy_dtype(lowered.body.type_info)
            if dtype is None:
                dtype = np.float32 if self._body_implies_float(lowered.body) else np.int32
            output = np.zeros(output_shape, dtype=dtype)

        def expr_evaluator(expr: Any) -> Any:
            return expr.accept(self)

        _loop_defid_to_name = {}
        for lp in lowered.loops:
            v = getattr(lp, "variable", None)
            if v and getattr(v, "defid", None):
                _loop_defid_to_name[v.defid] = getattr(v, "name", None)

        with self.env.scope():
            if not lowered.loops:
                if all(isinstance(idx, LiteralIR) for idx in clause_indices):
                    idx_tuple = tuple(int(idx.value) for idx in clause_indices)
                else:
                    idx_tuple = None
                if idx_tuple is not None:
                    value = lowered.body.accept(self)
                    if value is not None:
                        if isinstance(value, np.ndarray):
                            if value.ndim == 0:
                                value = value.item()
                            elif value.size == 1:
                                value = value.flatten()[0].item()
                        elif isinstance(value, np.generic):
                            value = value.item()
                        output[idx_tuple] = value
            else:
                _MAX = 1_000_000
                _n = [0]
                for loop_context in execute_lowered_loops(lowered.loops, {}, expr_evaluator):
                    _n[0] += 1
                    if _n[0] > _MAX:
                        raise RuntimeError("Einstein clause loop iterations exceeded limit.")
                    full_context = execute_lowered_bindings(lowered.bindings, loop_context, expr_evaluator)
                    for defid, val in full_context.items():
                        if defid is not None:
                            vname = _loop_defid_to_name.get(defid)
                            self.env.set_value(defid, val, name=vname)
                    if lowered.guards and not check_lowered_guards(lowered.guards, full_context, lambda e: self._to_bool(e.accept(self))):
                        continue
                    try:
                        value = lowered.body.accept(self)
                    except IndexError:
                        continue
                    idx_tuple = cell_index(full_context)
                    if idx_tuple is None and clause_indices:
                        out_idx = []
                        loop_pos = 0
                        for idx in clause_indices:
                            if isinstance(idx, LiteralIR):
                                try:
                                    out_idx.append(int(idx.value))
                                except (TypeError, ValueError):
                                    break
                            elif loop_pos < len(lowered.loops):
                                v = full_context.get(getattr(lowered.loops[loop_pos].variable, "defid", None))
                                if v is None:
                                    break
                                out_idx.append(v)
                                loop_pos += 1
                            else:
                                break
                        if len(out_idx) == len(clause_indices):
                            idx_tuple = tuple(out_idx)
                    if idx_tuple is None:
                        idx_tuple = tuple(full_context.get(getattr(loop.variable, "defid", None)) for loop in lowered.loops)
                    if idx_tuple is None or len(idx_tuple) != output.ndim:
                        continue
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            value = value.item()
                        elif value.size == 1:
                            value = value.flatten()[0].item()
                    elif isinstance(value, np.generic):
                        value = value.item()
                    if len(idx_tuple) == 1:
                        output[idx_tuple[0]] = value
                    else:
                        output[idx_tuple] = value

        if variable_defid:
            self.env.set_value(variable_defid, output)
        return output

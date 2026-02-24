"""backend helpers, env is DefId-keyed."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ..shared.types import BinaryOp
from ..ir.nodes import (
    LiteralIR, IdentifierIR, BinaryOpIR, IRVisitor,
    LiteralPatternIR, IdentifierPatternIR,
    TuplePatternIR, ArrayPatternIR, RestPatternIR, GuardPatternIR,
    ProgramIR, FunctionDefIR, ConstantDefIR, ExpressionIR,
)
from ..shared.defid import DefId

def _reject_non_lowered(node_type_name: str) -> None:
    raise RuntimeError(
        f"Non-lowered IR at runtime: {node_type_name}. "
        "Lowering passes must replace with lowered form before execution."
    )

def builtin_assert(condition: Any, message: str = "Assertion failed") -> None:
    def _all_true(v):
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return bool(v)
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return bool(v.item())
        if hasattr(v, "__iter__") and hasattr(v, "__len__") and not hasattr(v, "all"):
            return all(_all_true(e) for e in v)
        if hasattr(v, "all") and callable(v.all):
            return bool(v.all())
        return bool(v)
    if not _all_true(condition):
        raise RuntimeError(f"assertion failed: {message}")

def builtin_print(*args: Any) -> None:
    out = []
    for a in args:
        out.append(a.tolist() if hasattr(a, "tolist") else (list(a) if isinstance(a, (list, tuple)) else a))
    print(*out)

def builtin_len(collection: Any) -> int:
    if hasattr(collection, "__len__"):
        return len(collection)
    if isinstance(collection, np.ndarray):
        return int(collection.size)
    raise TypeError(f"Object of type {type(collection).__name__} has no len()")

def builtin_shape(array: Any) -> Any:
    if isinstance(array, np.ndarray):
        return np.array(array.shape, dtype=int)
    if isinstance(array, (list, tuple)):
        if not array:
            return np.array([0], dtype=int)
        if isinstance(array[0], (list, tuple, np.ndarray)):
            r = builtin_shape(array[0]) if not isinstance(array[0], np.ndarray) else array[0].shape
            return np.array([len(array)] + list(r), dtype=int)
        return np.array([len(array)], dtype=int)
    return np.array([], dtype=int)

def builtin_typeof(value: Any) -> str:
    if isinstance(value, bool): return "bool"
    if isinstance(value, (np.float32, np.float16)): return "f32"
    if isinstance(value, np.float64): return "f64"
    if isinstance(value, (int, np.integer)): return "i32"
    if isinstance(value, float): return "f32"
    if isinstance(value, str): return "str"
    if isinstance(value, np.ndarray): return "rectangular"
    if isinstance(value, (list, tuple)):
        if len(value) == 0: return "rectangular"
        first_len = None
        for e in value:
            if not isinstance(e, (list, tuple, np.ndarray)):
                return "rectangular"
            L = len(e) if not isinstance(e, np.ndarray) else (e.shape[0] if len(e.shape) > 0 else 0)
            if first_len is None: first_len = L
            elif L != first_len: return "array"
        return "rectangular"
    if value is None: return "null"
    return type(value).__name__

def builtin_sum(array: Any) -> Any:
    if isinstance(array, np.ndarray): return np.sum(array)
    if isinstance(array, (list, tuple)): return np.sum(np.array(array))
    return array

def builtin_max(*args: Any) -> Any:
    if not args: raise TypeError("max() requires at least one argument")
    if len(args) == 1 and isinstance(args[0], np.ndarray): return np.max(args[0])
    if len(args) == 2: return np.maximum(np.asarray(args[0]), np.asarray(args[1]))
    return max(*args)

def builtin_min(*args: Any) -> Any:
    if not args: raise TypeError("min() requires at least one argument")
    if len(args) == 1 and isinstance(args[0], np.ndarray): return np.min(args[0])
    if len(args) == 2: return np.minimum(np.asarray(args[0]), np.asarray(args[1]))
    return min(*args)

def builtin_array_append(array: Any, value: Any) -> Any:
    lst = array.tolist() if isinstance(array, np.ndarray) else (list(array) if isinstance(array, list) else [array])
    lst.append(value)
    return np.array(lst, dtype=array.dtype) if isinstance(array, np.ndarray) else lst


class _PatternMatcher(IRVisitor[Optional[Dict[DefId, Any]]]):
    def __init__(self, value: Any, backend: Any):
        self.value = value
        self.backend = backend

    def visit_literal_pattern(self, node: LiteralPatternIR) -> Optional[Dict[DefId, Any]]:
        return {} if self.value == node.value else None

    def visit_identifier_pattern(self, node: IdentifierPatternIR) -> Optional[Dict[DefId, Any]]:
        did = getattr(node, "defid", None)
        return {did: self.value} if did else {}

    def visit_wildcard_pattern(self, node: Any) -> Optional[Dict[DefId, Any]]:
        return {}

    def visit_tuple_pattern(self, node: TuplePatternIR) -> Optional[Dict[DefId, Any]]:
        if not isinstance(self.value, tuple):
            return None
        val_list = list(self.value)
        has_rest = any(hasattr(p, "pattern") for p in node.patterns)
        if has_rest:
            ri = next((i for i, p in enumerate(node.patterns) if hasattr(p, "pattern")), None)
            if ri is None or len(val_list) < len(node.patterns) - 1:
                return None
            bindings: Dict[DefId, Any] = {}
            for i in range(ri):
                r = node.patterns[i].accept(_PatternMatcher(val_list[i], self.backend))
                if r is None: return None
                bindings.update(r)
            end = len(val_list) - (len(node.patterns) - ri - 1)
            rp = node.patterns[ri]
            if getattr(getattr(rp, "pattern", None), "defid", None) is not None:
                bindings[rp.pattern.defid] = tuple(val_list[ri:end])
            for i in range(ri + 1, len(node.patterns)):
                r = node.patterns[i].accept(_PatternMatcher(val_list[end + (i - ri - 1)], self.backend))
                if r is None: return None
                bindings.update(r)
            return bindings
        if len(val_list) != len(node.patterns):
            return None
        bindings = {}
        for p, v in zip(node.patterns, val_list):
            r = p.accept(_PatternMatcher(v, self.backend))
            if r is None: return None
            bindings.update(r)
        return bindings

    def visit_array_pattern(self, node: ArrayPatternIR) -> Optional[Dict[DefId, Any]]:
        lst = self.value.tolist() if isinstance(self.value, np.ndarray) else list(self.value) if isinstance(self.value, (list, tuple)) else None
        if lst is None: return None
        has_rest = any(hasattr(p, "pattern") for p in node.patterns)
        if has_rest:
            ri = next((i for i, p in enumerate(node.patterns) if hasattr(p, "pattern")), None)
            if ri is None or len(lst) < len(node.patterns) - 1: return None
            bindings: Dict[DefId, Any] = {}
            for i in range(ri):
                r = node.patterns[i].accept(_PatternMatcher(lst[i], self.backend))
                if r is None: return None
                bindings.update(r)
            end = len(lst) - (len(node.patterns) - ri - 1)
            rp = node.patterns[ri]
            if getattr(getattr(rp, "pattern", None), "defid", None) is not None:
                bindings[rp.pattern.defid] = lst[ri:end]
            for i in range(ri + 1, len(node.patterns)):
                r = node.patterns[i].accept(_PatternMatcher(lst[end + (i - ri - 1)], self.backend))
                if r is None: return None
                bindings.update(r)
            return bindings
        if len(lst) != len(node.patterns): return None
        bindings = {}
        for p, v in zip(node.patterns, lst):
            r = p.accept(_PatternMatcher(v, self.backend))
            if r is None: return None
            bindings.update(r)
        return bindings

    def visit_rest_pattern(self, node: RestPatternIR) -> Optional[Dict[DefId, Any]]:
        did = getattr(node.pattern, "defid", None)
        return {did: self.value} if did else {}

    def visit_guard_pattern(self, node: GuardPatternIR) -> Optional[Dict[DefId, Any]]:
        return node.inner_pattern.accept(_PatternMatcher(self.value, self.backend))

    def visit_literal(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_identifier(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_binary_op(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_function_call(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_function_def(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_constant_def(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_rectangular_access(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_jagged_access(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_block_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_if_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_lambda(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_unary_op(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_range(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_array_comprehension(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_module(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_array_literal(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_tuple_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_tuple_access(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_interpolated_string(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_cast_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_member_access(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_try_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_match_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_reduction_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_where_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_arrow_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_pipeline_expression(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_builtin_call(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_function_ref(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_einstein_declaration(self, node: Any) -> Optional[Dict[DefId, Any]]: return None
    def visit_program(self, node: Any) -> Optional[Dict[DefId, Any]]: return None


def _extract_binding(constraint: Any) -> Optional[Tuple[DefId, Any]]:
    if isinstance(constraint, BinaryOpIR) and getattr(constraint, "operator", None) == BinaryOp.ASSIGN and isinstance(constraint.left, IdentifierIR):
        did = getattr(constraint.left, "defid", None)
        if did:
            return (did, constraint.right)
    return None


class FunctionDefRegistrar(IRVisitor[None]):
    def __init__(self, def_table: Dict[DefId, FunctionDefIR]):
        self.def_table = def_table
        self._scope_stack: List[Optional[DefId]] = []

    def visit_program(self, node: ProgramIR) -> None:
        for f in node.functions:
            f.accept(self)

    def visit_function_def(self, stmt: FunctionDefIR) -> None:
        self._scope_stack.append(getattr(stmt, "defid", None))
        try:
            if stmt.defid:
                self.def_table[stmt.defid] = stmt
            if getattr(stmt, "body", None):
                stmt.body.accept(self)
        finally:
            self._scope_stack.pop()

    def visit_constant_def(self, stmt: ConstantDefIR) -> None:
        pass

    def visit_literal(self, n: Any) -> None: pass
    def visit_identifier(self, n: Any) -> None: pass
    def visit_binary_op(self, n: Any) -> None: pass
    def visit_function_call(self, n: Any) -> None: pass
    def visit_unary_op(self, n: Any) -> None: pass
    def visit_rectangular_access(self, n: Any) -> None: pass
    def visit_jagged_access(self, n: Any) -> None: pass
    def visit_block_expression(self, n: Any) -> None:
        for stmt in getattr(n, "statements", []) or []:
            stmt.accept(self)
        if getattr(n, "final_expr", None):
            n.final_expr.accept(self)
    def visit_if_expression(self, n: Any) -> None:
        if getattr(n, "then_expr", None):
            n.then_expr.accept(self)
        if getattr(n, "else_expr", None):
            n.else_expr.accept(self)
    def visit_lambda(self, n: Any) -> None:
        if getattr(n, "body", None):
            n.body.accept(self)
    def visit_range(self, n: Any) -> None: pass
    def visit_array_comprehension(self, n: Any) -> None: pass
    def visit_array_literal(self, n: Any) -> None: pass
    def visit_tuple_expression(self, n: Any) -> None: pass
    def visit_tuple_access(self, n: Any) -> None: pass
    def visit_interpolated_string(self, n: Any) -> None: pass
    def visit_cast_expression(self, n: Any) -> None: pass
    def visit_member_access(self, n: Any) -> None: pass
    def visit_try_expression(self, n: Any) -> None: pass
    def visit_match_expression(self, n: Any) -> None: pass
    def visit_reduction_expression(self, n: Any) -> None: pass
    def visit_where_expression(self, n: Any) -> None: pass
    def visit_arrow_expression(self, n: Any) -> None: pass
    def visit_pipeline_expression(self, n: Any) -> None: pass
    def visit_builtin_call(self, n: Any) -> None: pass
    def visit_function_ref(self, n: Any) -> None: pass
    def visit_einstein_declaration(self, n: Any) -> None: pass
    def visit_variable_declaration(self, n: Any) -> None:
        if getattr(n, "value", None):
            n.value.accept(self)
    def visit_literal_pattern(self, n: Any) -> None: pass
    def visit_identifier_pattern(self, n: Any) -> None: pass
    def visit_wildcard_pattern(self, n: Any) -> None: pass
    def visit_tuple_pattern(self, n: Any) -> None: pass
    def visit_array_pattern(self, n: Any) -> None: pass
    def visit_rest_pattern(self, n: Any) -> None: pass
    def visit_guard_pattern(self, n: Any) -> None: pass
    def visit_module(self, n: Any) -> None: pass


class NameToDefIdLookup(IRVisitor[Optional[DefId]]):
    def __init__(self, name: str):
        self.name = name

    def visit_program(self, node: ProgramIR) -> Optional[DefId]:
        for f in node.functions:
            r = f.accept(self)
            if r is not None: return r
        for c in node.constants:
            r = c.accept(self)
            if r is not None: return r
        for stmt in node.statements:
            if isinstance(stmt, ExpressionIR):
                r = stmt.accept(self)
                if r is not None: return r
        return None

    def visit_function_def(self, stmt: FunctionDefIR) -> Optional[DefId]:
        return stmt.defid if stmt.name == self.name and stmt.defid else None

    def visit_constant_def(self, stmt: ConstantDefIR) -> Optional[DefId]:
        return stmt.defid if stmt.name == self.name and stmt.defid else None

    def visit_where_expression(self, node: Any) -> Optional[DefId]:
        r = node.expr.accept(self)
        if r: return r
        for c in getattr(node, "constraints", []) or []:
            r = c.accept(self)
            if r: return r
        return None

    def visit_literal(self, n: Any) -> Optional[DefId]: return None
    def visit_identifier(self, n: Any) -> Optional[DefId]: return None
    def visit_binary_op(self, n: Any) -> Optional[DefId]: return None
    def visit_function_call(self, n: Any) -> Optional[DefId]: return None
    def visit_unary_op(self, n: Any) -> Optional[DefId]: return None
    def visit_rectangular_access(self, n: Any) -> Optional[DefId]: return None
    def visit_jagged_access(self, n: Any) -> Optional[DefId]: return None
    def visit_block_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_if_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_lambda(self, n: Any) -> Optional[DefId]: return None
    def visit_range(self, n: Any) -> Optional[DefId]: return None
    def visit_array_comprehension(self, n: Any) -> Optional[DefId]: return None
    def visit_array_literal(self, n: Any) -> Optional[DefId]: return None
    def visit_tuple_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_tuple_access(self, n: Any) -> Optional[DefId]: return None
    def visit_interpolated_string(self, n: Any) -> Optional[DefId]: return None
    def visit_cast_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_member_access(self, n: Any) -> Optional[DefId]: return None
    def visit_try_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_match_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_reduction_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_arrow_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_pipeline_expression(self, n: Any) -> Optional[DefId]: return None
    def visit_builtin_call(self, n: Any) -> Optional[DefId]: return None
    def visit_function_ref(self, n: Any) -> Optional[DefId]: return None
    def visit_einstein_declaration(self, n: Any) -> Optional[DefId]: return None
    def visit_variable_declaration(self, n: Any) -> Optional[DefId]:
        return n.value.accept(self) if getattr(n, "value", None) else None
    def visit_literal_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_identifier_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_wildcard_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_tuple_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_array_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_rest_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_guard_pattern(self, n: Any) -> Optional[DefId]: return None
    def visit_module(self, n: Any) -> Optional[DefId]: return None


class DefIdToNameLookup(IRVisitor[Optional[str]]):
    def __init__(self, defid: DefId):
        self.defid = defid

    def visit_program(self, node: ProgramIR) -> Optional[str]:
        for f in node.functions:
            r = f.accept(self)
            if r is not None: return r
        for c in node.constants:
            r = c.accept(self)
            if r is not None: return r
        if getattr(node, "defid_to_name", None):
            return node.defid_to_name.get(self.defid)
        for stmt in node.statements:
            if isinstance(stmt, ExpressionIR) and getattr(stmt, "defid", None) == self.defid and getattr(node, "defid_to_name", None):
                return node.defid_to_name.get(self.defid)
        return None

    def visit_function_def(self, stmt: FunctionDefIR) -> Optional[str]:
        return stmt.name if stmt.defid == self.defid else None

    def visit_constant_def(self, stmt: ConstantDefIR) -> Optional[str]:
        return stmt.name if stmt.defid == self.defid else None

    def visit_literal(self, n: Any) -> Optional[str]: return None
    def visit_identifier(self, n: Any) -> Optional[str]: return None
    def visit_binary_op(self, n: Any) -> Optional[str]: return None
    def visit_function_call(self, n: Any) -> Optional[str]: return None
    def visit_unary_op(self, n: Any) -> Optional[str]: return None
    def visit_rectangular_access(self, n: Any) -> Optional[str]: return None
    def visit_jagged_access(self, n: Any) -> Optional[str]: return None
    def visit_block_expression(self, n: Any) -> Optional[str]: return None
    def visit_if_expression(self, n: Any) -> Optional[str]: return None
    def visit_lambda(self, n: Any) -> Optional[str]: return None
    def visit_range(self, n: Any) -> Optional[str]: return None
    def visit_array_comprehension(self, n: Any) -> Optional[str]: return None
    def visit_array_literal(self, n: Any) -> Optional[str]: return None
    def visit_tuple_expression(self, n: Any) -> Optional[str]: return None
    def visit_tuple_access(self, n: Any) -> Optional[str]: return None
    def visit_interpolated_string(self, n: Any) -> Optional[str]: return None
    def visit_cast_expression(self, n: Any) -> Optional[str]: return None
    def visit_member_access(self, n: Any) -> Optional[str]: return None
    def visit_try_expression(self, n: Any) -> Optional[str]: return None
    def visit_match_expression(self, n: Any) -> Optional[str]: return None
    def visit_reduction_expression(self, n: Any) -> Optional[str]: return None
    def visit_where_expression(self, n: Any) -> Optional[str]: return None
    def visit_arrow_expression(self, n: Any) -> Optional[str]: return None
    def visit_pipeline_expression(self, n: Any) -> Optional[str]: return None
    def visit_builtin_call(self, n: Any) -> Optional[str]: return None
    def visit_function_ref(self, n: Any) -> Optional[str]: return None
    def visit_einstein_declaration(self, n: Any) -> Optional[str]: return None
    def visit_variable_declaration(self, n: Any) -> Optional[str]:
        return n.value.accept(self) if getattr(n, "value", None) else None
    def visit_literal_pattern(self, n: Any) -> Optional[str]: return None
    def visit_identifier_pattern(self, n: Any) -> Optional[str]: return None
    def visit_wildcard_pattern(self, n: Any) -> Optional[str]: return None
    def visit_tuple_pattern(self, n: Any) -> Optional[str]: return None
    def visit_array_pattern(self, n: Any) -> Optional[str]: return None
    def visit_rest_pattern(self, n: Any) -> Optional[str]: return None
    def visit_guard_pattern(self, n: Any) -> Optional[str]: return None
    def visit_module(self, n: Any) -> Optional[str]: return None


class BindingExtractor(IRVisitor[Optional[Tuple[DefId, ExpressionIR]]]):
    def visit_binary_op(self, expr: BinaryOpIR) -> Optional[Tuple[DefId, ExpressionIR]]:
        if getattr(expr, "operator", None) == BinaryOp.ASSIGN and isinstance(expr.left, IdentifierIR):
            did = getattr(expr.left, "defid", None)
            if did:
                return (did, expr.right)
        return None

    def visit_program(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_literal(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_identifier(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_function_call(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_unary_op(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_rectangular_access(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_jagged_access(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_block_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_if_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_lambda(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_range(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_array_comprehension(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_array_literal(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_tuple_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_tuple_access(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_interpolated_string(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_cast_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_member_access(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_try_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_match_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_reduction_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_where_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_arrow_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_pipeline_expression(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_builtin_call(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_function_ref(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_einstein_declaration(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_variable_declaration(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]:
        return n.value.accept(self) if getattr(n, "value", None) else None
    def visit_literal_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_identifier_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_wildcard_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_tuple_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_array_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_rest_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_guard_pattern(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_constant_def(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_function_def(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None
    def visit_module(self, n: Any) -> Optional[Tuple[DefId, ExpressionIR]]: return None

"""
Visitor Helper Utilities

Common visitor patterns for expression analysis without isinstance/hasattr.
"""

from typing import Set, Optional, List, Any
from ..ir.nodes import (
    ExpressionIR, IdentifierIR, IndexVarIR, BinaryOpIR, LiteralIR,
    RectangularAccessIR, FunctionCallIR, IRVisitor,
    UnaryOpIR,
    BlockExpressionIR,
    IfExpressionIR,
    LambdaIR,
    RangeIR,
    ArrayComprehensionIR,
    ArrayLiteralIR,
    TupleExpressionIR,
    TupleAccessIR,
    CastExpressionIR,
    MemberAccessIR,
    TryExpressionIR,
    MatchExpressionIR,
    ReductionExpressionIR,
    WhereExpressionIR,
    PipelineExpressionIR,
    BuiltinCallIR,
    JaggedAccessIR,
    is_function_binding,
    is_einstein_binding,
    ProgramIR,
)
from ..shared.defid import DefId


class _DefIdFinder(IRVisitor[Optional[DefId]]):
    """IR visitor: find first IdentifierIR or IndexVarIR with given name and return its defid."""

    def __init__(self, target_name: str) -> None:
        self._name = target_name

    def _first(self, *nodes: Any) -> Optional[DefId]:
        for node in nodes:
            if node is not None:
                result = node.accept(self)
                if result is not None:
                    return result
        return None

    def visit_identifier(self, node: IdentifierIR) -> Optional[DefId]:
        if node.name == self._name:
            return node.defid
        return None

    def visit_index_var(self, node: IndexVarIR) -> Optional[DefId]:
        if node.name == self._name:
            return node.defid
        return None

    def visit_literal(self, node: LiteralIR) -> Optional[DefId]:
        return None

    def visit_binary_op(self, node: BinaryOpIR) -> Optional[DefId]:
        return self._first(node.left, node.right)

    def visit_unary_op(self, node: UnaryOpIR) -> Optional[DefId]:
        return self._first(node.operand)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> Optional[DefId]:
        result = self._first(node.array)
        if result is not None:
            return result
        return self._first(*(node.indices or []))

    def visit_jagged_access(self, node: JaggedAccessIR) -> Optional[DefId]:
        return self._first(node.base, *(node.index_chain or []))

    def visit_function_call(self, node: FunctionCallIR) -> Optional[DefId]:
        result = self._first(node.callee_expr)
        if result is not None:
            return result
        return self._first(*(node.arguments or []))

    def visit_block_expression(self, node: BlockExpressionIR) -> Optional[DefId]:
        result = self._first(*(node.statements or []))
        if result is not None:
            return result
        return self._first(node.final_expr)

    def visit_if_expression(self, node: IfExpressionIR) -> Optional[DefId]:
        return self._first(node.condition, node.then_expr, node.else_expr)

    def visit_lambda(self, node: LambdaIR) -> Optional[DefId]:
        return self._first(node.body)

    def visit_range(self, node: RangeIR) -> Optional[DefId]:
        return self._first(node.start, node.end)

    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> Optional[DefId]:
        return self._first(node.body)

    def visit_array_literal(self, node: ArrayLiteralIR) -> Optional[DefId]:
        return self._first(*(node.elements or []))

    def visit_tuple_expression(self, node: TupleExpressionIR) -> Optional[DefId]:
        return self._first(*(node.elements or []))

    def visit_tuple_access(self, node: TupleAccessIR) -> Optional[DefId]:
        return self._first(node.tuple_expr)

    def visit_cast_expression(self, node: CastExpressionIR) -> Optional[DefId]:
        return self._first(node.expr)

    def visit_member_access(self, node: MemberAccessIR) -> Optional[DefId]:
        return self._first(node.object)

    def visit_try_expression(self, node: TryExpressionIR) -> Optional[DefId]:
        return self._first(node.operand)

    def visit_match_expression(self, node: MatchExpressionIR) -> Optional[DefId]:
        result = self._first(node.scrutinee)
        if result is not None:
            return result
        for arm in node.arms or []:
            result = self._first(arm.body)
            if result is not None:
                return result
        return None

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> Optional[DefId]:
        return self._first(node.body)

    def visit_where_expression(self, node: WhereExpressionIR) -> Optional[DefId]:
        return self._first(node.expr)

    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> Optional[DefId]:
        return self._first(node.left, node.right)

    def visit_builtin_call(self, node: BuiltinCallIR) -> Optional[DefId]:
        return self._first(*(node.args or []))

    def visit_program(self, node: Any) -> Optional[DefId]:
        return None

    def visit_index_rest(self, node: Any) -> Optional[DefId]:
        return None

    def visit_module(self, node: Any) -> Optional[DefId]:
        return None

    def visit_interpolated_string(self, node: Any) -> Optional[DefId]:
        return None

    def visit_literal_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_identifier_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_wildcard_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_tuple_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_array_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_rest_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_guard_pattern(self, node: Any) -> Optional[DefId]:
        return None

    def visit_binding(self, node: Any) -> Optional[DefId]:
        if node.expr is not None and not (is_function_binding(node) or is_einstein_binding(node)):
            return self._first(node.expr)
        return None


def defid_of_var_in_expr(expr: Optional[ExpressionIR], name: str) -> Optional[DefId]:
    """Return defid of first IdentifierIR or IndexVarIR with given name in expr tree. Uses IR visitor (no isinstance/hasattr)."""
    if expr is None:
        return None
    return expr.accept(_DefIdFinder(name))


class VariableExtractor(IRVisitor[Set[str]]):
    """Extract variable names from expression - visitor pattern"""
    
    def visit_program(self, node) -> Set[str]:
        """Visit program - not used for variable extraction"""
        return set()
    
    def visit_identifier(self, expr: IdentifierIR) -> Set[str]:
        return {expr.name}
    
    def visit_binary_op(self, expr: BinaryOpIR) -> Set[str]:
        vars: Set[str] = set()
        vars.update(expr.left.accept(self))
        vars.update(expr.right.accept(self))
        return vars
    
    # Default: empty set
    def visit_literal(self, node) -> Set[str]:
        return set()
    
    def visit_function_call(self, node) -> Set[str]:
        return set()
    
    def visit_unary_op(self, node) -> Set[str]:
        return set()
    
    def visit_rectangular_access(self, node) -> Set[str]:
        return set()
    
    def visit_jagged_access(self, node) -> Set[str]:
        return set()
    
    def visit_block_expression(self, node) -> Set[str]:
        return set()
    
    def visit_if_expression(self, node) -> Set[str]:
        return set()
    
    def visit_lambda(self, node) -> Set[str]:
        return set()
    
    def visit_range(self, node) -> Set[str]:
        return set()
    
    def visit_array_comprehension(self, node) -> Set[str]:
        return set()
    
    def visit_array_literal(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_expression(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_access(self, node) -> Set[str]:
        return set()
    
    def visit_interpolated_string(self, node) -> Set[str]:
        return set()
    
    def visit_cast_expression(self, node) -> Set[str]:
        return set()
    
    def visit_member_access(self, node) -> Set[str]:
        return set()
    
    def visit_try_expression(self, node) -> Set[str]:
        return set()
    
    def visit_match_expression(self, node) -> Set[str]:
        return set()
    
    def visit_reduction_expression(self, node) -> Set[str]:
        return set()
    
    def visit_where_expression(self, node) -> Set[str]:
        return set()
    
    def visit_pipeline_expression(self, node) -> Set[str]:
        return set()
    
    def visit_builtin_call(self, node) -> Set[str]:
        return set()
    
    def visit_binding(self, node: Any) -> Set[str]:
        if is_function_binding(node) or is_einstein_binding(node):
            return set()
        if node.expr is not None:
            return node.expr.accept(self)
        return set()

    def visit_literal_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_identifier_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_wildcard_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_tuple_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_array_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_rest_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_guard_pattern(self, node) -> Set[str]:
        return set()
    
    def visit_module(self, node) -> Set[str]:
        return set()


class ConstantEvaluator(IRVisitor[Optional[int]]):
    """Evaluate constant expression to integer - visitor pattern"""
    
    def visit_program(self, node) -> Optional[int]:
        """Visit program - not used for constant evaluation"""
        return None
    
    def visit_literal(self, expr: LiteralIR) -> Optional[int]:
        if isinstance(expr.value, int):
            return expr.value
        return None
    
    # Default: None
    def visit_identifier(self, node) -> Optional[int]:
        return None
    
    def visit_binary_op(self, node) -> Optional[int]:
        return None
    
    def visit_function_call(self, node) -> Optional[int]:
        return None
    
    def visit_unary_op(self, node) -> Optional[int]:
        return None
    
    def visit_rectangular_access(self, node) -> Optional[int]:
        return None
    
    def visit_jagged_access(self, node) -> Optional[int]:
        return None
    
    def visit_block_expression(self, node) -> Optional[int]:
        return None
    
    def visit_if_expression(self, node) -> Optional[int]:
        return None
    
    def visit_lambda(self, node) -> Optional[int]:
        return None
    
    def visit_range(self, node) -> Optional[int]:
        return None
    
    def visit_array_comprehension(self, node) -> Optional[int]:
        return None
    
    def visit_array_literal(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_expression(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_access(self, node) -> Optional[int]:
        return None
    
    def visit_interpolated_string(self, node) -> Optional[int]:
        return None
    
    def visit_cast_expression(self, node) -> Optional[int]:
        return None
    
    def visit_member_access(self, node) -> Optional[int]:
        return None
    
    def visit_try_expression(self, node) -> Optional[int]:
        return None
    
    def visit_match_expression(self, node) -> Optional[int]:
        return None
    
    def visit_reduction_expression(self, node) -> Optional[int]:
        return None
    
    def visit_where_expression(self, node) -> Optional[int]:
        return None
    
    def visit_pipeline_expression(self, node) -> Optional[int]:
        return None
    
    def visit_builtin_call(self, node) -> Optional[int]:
        return None
    
    def visit_binding(self, node) -> Optional[int]:
        return None
    
    def visit_literal_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_identifier_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_wildcard_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_tuple_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_array_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_rest_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_guard_pattern(self, node) -> Optional[int]:
        return None
    
    def visit_module(self, node) -> Optional[int]:
        return None


class ArrayAccessCollector(IRVisitor[List[RectangularAccessIR]]):
    """Collect all array accesses from expression - visitor pattern"""
    
    def visit_program(self, node) -> List[RectangularAccessIR]:
        """Visit program - not used for array access collection"""
        return []
    
    def visit_rectangular_access(self, expr: RectangularAccessIR) -> List[RectangularAccessIR]:
        accesses = [expr]
        accesses.extend(expr.array.accept(self))
        for idx in (expr.indices or []):
            if idx is not None:
                accesses.extend(idx.accept(self))
        return accesses
    
    def visit_binary_op(self, expr: BinaryOpIR) -> List[RectangularAccessIR]:
        accesses = []
        accesses.extend(expr.left.accept(self))
        accesses.extend(expr.right.accept(self))
        return accesses
    
    def visit_function_call(self, expr: FunctionCallIR) -> List[RectangularAccessIR]:
        accesses = []
        for arg in expr.arguments:
            accesses.extend(arg.accept(self))
        return accesses
    
    # Default: empty list
    def visit_literal(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_index_var(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_index_rest(self, node) -> List[RectangularAccessIR]:
        return []

    def visit_unary_op(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_jagged_access(self, node) -> List[RectangularAccessIR]:
        accesses = []
        accesses.extend(node.base.accept(self))
        for idx in (node.index_chain or []):
            accesses.extend(idx.accept(self))
        return accesses
    
    def visit_block_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for stmt in (node.statements or []):
            accesses.extend(stmt.accept(self))
        if node.final_expr is not None:
            accesses.extend(node.final_expr.accept(self))
        return accesses

    def visit_if_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        if node.condition is not None:
            accesses.extend(node.condition.accept(self))
        if node.then_expr is not None:
            accesses.extend(node.then_expr.accept(self))
        if node.else_expr is not None:
            accesses.extend(node.else_expr.accept(self))
        return accesses

    def visit_lambda(self, node) -> List[RectangularAccessIR]:
        if node.body is not None:
            return node.body.accept(self)
        return []

    def visit_range(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_comprehension(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_literal(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for el in (node.elements or []):
            if el is not None:
                accesses.extend(el.accept(self))
        return accesses
    
    def visit_tuple_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        for el in (node.elements or []):
            accesses.extend(el.accept(self))
        return accesses
    
    def visit_tuple_access(self, node) -> List[RectangularAccessIR]:
        return node.tuple_expr.accept(self)
    
    def visit_interpolated_string(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_cast_expression(self, node) -> List[RectangularAccessIR]:
        return node.expr.accept(self)

    def visit_member_access(self, node) -> List[RectangularAccessIR]:
        return node.object.accept(self)
    
    def visit_try_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_match_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        accesses.extend(node.scrutinee.accept(self))
        for arm in (node.arms or []):
            if arm.body is not None:
                accesses.extend(arm.body.accept(self))
        return accesses

    def visit_reduction_expression(self, node) -> List[RectangularAccessIR]:
        if node.body:
            return node.body.accept(self)
        return []
    
    def visit_where_expression(self, node) -> List[RectangularAccessIR]:
        accesses = []
        accesses.extend(node.expr.accept(self))
        for c in (node.constraints or []):
            accesses.extend(c.accept(self))
        return accesses
    
    def visit_pipeline_expression(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_builtin_call(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_binding(self, node: Any) -> List[RectangularAccessIR]:
        if not (is_function_binding(node) or is_einstein_binding(node)) and node.value:
            return node.value.accept(self)
        return []
    
    def visit_literal_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_identifier_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_wildcard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_tuple_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_array_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_rest_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_guard_pattern(self, node) -> List[RectangularAccessIR]:
        return []
    
    def visit_module(self, node) -> List[RectangularAccessIR]:
        return []


class ExprInvolvesVarVisitor(IRVisitor[bool]):
    """True if the expression tree contains an IdentifierIR with the given var_name (or BinaryOpIR with such in left/right)."""

    def __init__(self, var_name: str) -> None:
        self._var_name = var_name

    def visit_identifier(self, node: IdentifierIR) -> bool:
        return node.name == self._var_name

    def visit_binary_op(self, node: BinaryOpIR) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_literal(self, node: Any) -> bool:
        return False

    def visit_index_var(self, node: Any) -> bool:
        return node.range_ir.accept(self) if node.range_ir is not None else False

    def visit_index_rest(self, node: Any) -> bool:
        return False

    def visit_rectangular_access(self, node: Any) -> bool:
        if node.array.accept(self):
            return True
        return any(idx.accept(self) for idx in node.indices)

    def visit_function_call(self, node: Any) -> bool:
        if node.callee_expr.accept(self):
            return True
        return any(a.accept(self) for a in node.arguments)

    def visit_unary_op(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_block_expression(self, node: Any) -> bool:
        for stmt in node.statements:
            if stmt.accept(self):
                return True
        return (
            node.final_expr.accept(self)
            if node.final_expr is not None
            else False
        )

    def visit_if_expression(self, node: Any) -> bool:
        return (
            node.condition.accept(self)
            or node.then_expr.accept(self)
            or (node.else_expr.accept(self) if node.else_expr is not None else False)
        )

    def visit_lambda(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_range(self, node: Any) -> bool:
        return node.start.accept(self) or node.end.accept(self)

    def visit_array_comprehension(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_module(self, node: Any) -> bool:
        return False

    def visit_array_literal(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_expression(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_tuple_access(self, node: Any) -> bool:
        return node.tuple_expr.accept(self)

    def visit_interpolated_string(self, node: Any) -> bool:
        return any(p.accept(self) for p in node.parts)

    def visit_cast_expression(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_member_access(self, node: Any) -> bool:
        return node.object.accept(self)

    def visit_try_expression(self, node: Any) -> bool:
        return node.operand.accept(self)

    def visit_match_expression(self, node: Any) -> bool:
        if node.scrutinee.accept(self):
            return True
        return any(arm.body.accept(self) for arm in node.arms)

    def visit_reduction_expression(self, node: Any) -> bool:
        return node.body.accept(self)

    def visit_where_expression(self, node: Any) -> bool:
        if node.expr.accept(self):
            return True
        return any(c.accept(self) for c in node.constraints)

    def visit_pipeline_expression(self, node: Any) -> bool:
        return node.left.accept(self) or node.right.accept(self)

    def visit_builtin_call(self, node: Any) -> bool:
        return any(a.accept(self) for a in node.args)

    def visit_jagged_access(self, node: Any) -> bool:
        if node.base.accept(self):
            return True
        return any(idx.accept(self) for idx in node.index_chain)

    def visit_literal_pattern(self, node: Any) -> bool:
        return False

    def visit_identifier_pattern(self, node: Any) -> bool:
        return False

    def visit_wildcard_pattern(self, node: Any) -> bool:
        return False

    def visit_tuple_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_array_pattern(self, node: Any) -> bool:
        return any(e.accept(self) for e in node.elements)

    def visit_rest_pattern(self, node: Any) -> bool:
        return False

    def visit_guard_pattern(self, node: Any) -> bool:
        return node.inner_pattern.accept(self) or node.guard_expr.accept(self)

    def visit_binding(self, node: Any) -> bool:
        return node.expr.accept(self)

    def visit_program(self, node: Any) -> bool:
        return any(stmt.accept(self) for stmt in node.statements)


class DefaultRecursingVisitor(IRVisitor[None]):
    """
    Base visitor that by default recurses into all sub-expressions and returns None.
    Override only the node kinds that need custom behavior (e.g. cast validation).
    """

    def _recurse(self, *nodes: Any) -> None:
        for n in nodes:
            if n is not None:
                n.accept(self)

    def _recurse_seq(self, seq: Any) -> None:
        if seq:
            for n in seq:
                if n is not None:
                    n.accept(self)

    def visit_program(self, node: ProgramIR) -> None:
        self._recurse_seq(node.statements)
        self._recurse_seq(node.functions)
        self._recurse_seq(node.constants)

    def visit_literal(self, node: LiteralIR) -> None:
        pass

    def visit_identifier(self, node: IdentifierIR) -> None:
        pass

    def visit_index_var(self, node: IndexVarIR) -> None:
        self._recurse(node.range_ir)

    def visit_index_rest(self, node: Any) -> None:
        pass

    def visit_binary_op(self, node: BinaryOpIR) -> None:
        self._recurse(node.left, node.right)

    def visit_function_call(self, node: FunctionCallIR) -> None:
        self._recurse(node.callee_expr)
        self._recurse_seq(node.arguments)

    def visit_rectangular_access(self, node: RectangularAccessIR) -> None:
        self._recurse(node.array)
        self._recurse_seq(node.indices)

    def visit_jagged_access(self, node: JaggedAccessIR) -> None:
        self._recurse(node.base)
        self._recurse_seq(node.index_chain)

    def visit_block_expression(self, node: BlockExpressionIR) -> None:
        self._recurse_seq(node.statements)
        self._recurse(node.final_expr)

    def visit_if_expression(self, node: IfExpressionIR) -> None:
        self._recurse(node.condition, node.then_expr, node.else_expr)

    def visit_lambda(self, node: LambdaIR) -> None:
        self._recurse(node.body)

    def visit_unary_op(self, node: UnaryOpIR) -> None:
        self._recurse(node.operand)

    def visit_range(self, node: RangeIR) -> None:
        self._recurse(node.start, node.end)

    def visit_array_comprehension(self, node: ArrayComprehensionIR) -> None:
        self._recurse(node.body)

    def visit_module(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'functions', None))
        self._recurse_seq(getattr(node, 'constants', None))
        self._recurse_seq(getattr(node, 'submodules', None))

    def visit_array_literal(self, node: ArrayLiteralIR) -> None:
        self._recurse_seq(node.elements)

    def visit_tuple_expression(self, node: TupleExpressionIR) -> None:
        self._recurse_seq(node.elements)

    def visit_tuple_access(self, node: TupleAccessIR) -> None:
        self._recurse(node.tuple_expr)

    def visit_interpolated_string(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'parts', None))

    def visit_cast_expression(self, node: CastExpressionIR) -> None:
        self._recurse(node.expr)

    def visit_member_access(self, node: MemberAccessIR) -> None:
        self._recurse(node.object)

    def visit_try_expression(self, node: TryExpressionIR) -> None:
        self._recurse(node.operand)

    def visit_match_expression(self, node: MatchExpressionIR) -> None:
        self._recurse(node.scrutinee)
        for arm in node.arms or []:
            self._recurse(arm.body)

    def visit_reduction_expression(self, node: ReductionExpressionIR) -> None:
        self._recurse(node.body)
        wc = getattr(node, 'where_clause', None)
        if wc is not None:
            self._recurse_seq(getattr(wc, 'constraints', None))

    def visit_where_expression(self, node: WhereExpressionIR) -> None:
        self._recurse(node.expr)
        self._recurse_seq(node.constraints)

    def visit_pipeline_expression(self, node: PipelineExpressionIR) -> None:
        self._recurse(node.left, node.right)

    def visit_builtin_call(self, node: BuiltinCallIR) -> None:
        self._recurse_seq(node.args)

    def visit_literal_pattern(self, node: Any) -> None:
        pass

    def visit_identifier_pattern(self, node: Any) -> None:
        pass

    def visit_wildcard_pattern(self, node: Any) -> None:
        pass

    def visit_tuple_pattern(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'elements', None))

    def visit_array_pattern(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'elements', None))

    def visit_rest_pattern(self, node: Any) -> None:
        pass

    def visit_guard_pattern(self, node: Any) -> None:
        self._recurse(getattr(node, 'inner_pattern', None), getattr(node, 'guard_expr', None))

    def visit_or_pattern(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'alternatives', None))

    def visit_constructor_pattern(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'patterns', None))

    def visit_binding_pattern(self, node: Any) -> None:
        self._recurse(getattr(node, 'inner_pattern', None))

    def visit_range_pattern(self, node: Any) -> None:
        pass

    def visit_function_value(self, node: Any) -> None:
        self._recurse(getattr(node, 'body', None))

    def visit_einstein(self, node: Any) -> None:
        pass

    def visit_einstein_clause(self, node: Any) -> None:
        pass

    def visit_binding(self, node: Any) -> None:
        if is_function_binding(node) or is_einstein_binding(node):
            return
        self._recurse(node.expr)

    def visit_lowered_reduction(self, node: Any) -> None:
        self._recurse(getattr(node, 'body', None))

    def visit_lowered_comprehension(self, node: Any) -> None:
        self._recurse(getattr(node, 'body', None))

    def visit_lowered_einstein_clause(self, node: Any) -> None:
        self._recurse(getattr(node, 'body', None))

    def visit_lowered_einstein(self, node: Any) -> None:
        self._recurse_seq(getattr(node, 'items', None))

    def visit_lowered_recurrence(self, node: Any) -> None:
        self._recurse(getattr(node, 'initial', None))
        rl = getattr(node, 'recurrence_loop', None)
        if rl is not None:
            self._recurse(getattr(rl, 'iterable', None))
        self._recurse(getattr(node, 'body', None))


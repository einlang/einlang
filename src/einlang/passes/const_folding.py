"""
Const Folding Pass

Rust Pattern: rustc_mir::const_eval
Reference: PASS_SYSTEM_DESIGN.md
"""

from ..passes.base import BasePass, TyCtxt
from ..passes.type_inference import TypeInferencePass
from ..passes.exhaustiveness import ExhaustivenessPass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR, BindingIR, is_einstein_binding, is_function_binding, is_constant_binding,
    LiteralIR, BinaryOpIR, UnaryOpIR, FunctionCallIR,
    BlockExpressionIR, IfExpressionIR, LambdaIR, IRVisitor,
    RectangularAccessIR, JaggedAccessIR, ArrayLiteralIR, TupleExpressionIR,
    TupleAccessIR, InterpolatedStringIR, CastExpressionIR, MemberAccessIR,
    TryExpressionIR, MatchExpressionIR, ReductionExpressionIR, WhereExpressionIR,
    ArrowExpressionIR, PipelineExpressionIR, BuiltinCallIR,
)
from typing import Optional, Any


class ConstFoldingPass(BasePass):
    """
    Constant folding pass (Rust naming: rustc_mir::const_eval).
    
    Rust Pattern: rustc_mir::const_eval (constant evaluation)
    
    Implementation Alignment: Follows Rust's const eval:
    - Evaluates constant expressions at compile-time
    - Replaces constant expressions with literals
    - Preserves type information
    - Only folds pure operations (no side effects)
    
    Reference: Rust const eval (CTFE - Compile-Time Function Execution)
    """
    requires = [TypeInferencePass, ExhaustivenessPass]  # Depends on type inference and exhaustiveness
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Fold constant expressions in IR.
        
        Rust Pattern: Constant evaluation at compile-time
        """
        folder = ConstantFolder(tcx)
        
        # Fold all expressions in IR
        # Use visitor pattern (no isinstance) - Rust pattern
        folder_visitor = ConstantFoldingVisitor(folder)
        
        # Use visitor pattern: ir.accept(visitor) modifies in place
        ir.accept(folder_visitor)
        
        # Return same IR object (modified in place)
        return ir


class ConstantFolder(IRVisitor[ExpressionIR]):
    """
    Constant folder - evaluates constant expressions (Rust pattern: rustc_mir::const_eval::ConstEval).
    
    Rust Pattern: rustc_mir::const_eval::ConstEval
    
    Implementation Alignment: Follows Rust's visitor pattern:
    - Visitor pattern for expression folding (no if/elif chains)
    - Type-safe dispatch via visit_xxx methods
    - Recursive folding via accept() calls
    
    Reference: `rustc_mir::const_eval` uses visitor pattern for constant evaluation
    
    Note: visit_function_def and visit_constant_def are required by IRVisitor interface
    but should not be called on expressions. They raise NotImplementedError.
    """
    """
    Constant folder - evaluates constant expressions (Rust pattern: rustc_mir::const_eval::ConstEval).
    
    Rust Pattern: rustc_mir::const_eval::ConstEval
    
    Implementation Alignment: Follows Rust's visitor pattern:
    - Visitor pattern for expression folding (no if/elif chains)
    - Type-safe dispatch via visit_xxx methods
    - Recursive folding via accept() calls
    
    Reference: `rustc_mir::const_eval` uses visitor pattern for constant evaluation
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.fold_count = 0
    
    def visit_program(self, node: ProgramIR) -> ExpressionIR:
        """Visit program - not used for constant folding (expressions only)"""
        # This visitor only folds expressions, not programs
        # Return a dummy expression (should not be called)
        from ..ir.nodes import LiteralIR
        from ..shared.source_location import SourceLocation
        return LiteralIR(value=None, location=SourceLocation("", 0, 0, 0, 0))
    
    def fold_expression(self, expr: ExpressionIR) -> ExpressionIR:
        """
        Fold constant expression using visitor pattern.
        
        Returns: LiteralIR if constant, otherwise original expression
        
        Rust Pattern: Visitor pattern dispatch via accept()
        """
        return expr.accept(self)  # Visitor pattern
    
    def visit_literal(self, expr: LiteralIR) -> ExpressionIR:
        """
        Literals are already constants - return as-is.
        
        Rust Pattern: Visitor pattern for literals
        """
        return expr  # Already constant
    
    def visit_identifier(self, expr) -> ExpressionIR:
        """
        Identifiers are not constants (unless they refer to const values).
        
        Rust Pattern: Visitor pattern for identifiers
        """
        # Could check if identifier refers to a constant value
        # For now, return as-is
        return expr
    
    def visit_binary_op(self, expr: BinaryOpIR) -> ExpressionIR:
        """
        Fold binary operation if operands are constant.
        
        Rust Pattern: Visitor pattern for binary operations
        """
        # Use visitor pattern: recursively fold operands
        left = expr.left.accept(self)  # Visitor pattern
        right = expr.right.accept(self)  # Visitor pattern
        
        # Use visitor pattern (no isinstance) - Rust pattern
        left_literal = left.accept(LiteralExtractor())
        right_literal = right.accept(LiteralExtractor())
        
        if left_literal is not None and right_literal is not None:
            # Evaluate at compile-time
            result = self._eval_binary_op(expr.operator, left_literal, right_literal)
            if result is not None:
                self.fold_count += 1
                return LiteralIR(
                    value=result,
                    location=expr.location,
                    type_info=expr.type_info,
                    shape_info=expr.shape_info
                )
        
        # Not constant, return (potentially partially-folded) expression
        # kind is class variable, not constructor parameter
        return BinaryOpIR(
            operator=expr.operator,
            left=left,
            right=right,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_unary_op(self, expr: UnaryOpIR) -> ExpressionIR:
        """
        Fold unary operation if operand is constant.
        
        Rust Pattern: Visitor pattern for unary operations
        """
        # Use visitor pattern: recursively fold operand
        operand = expr.operand.accept(self)  # Visitor pattern
        
        # Use visitor pattern (no isinstance) - Rust pattern
        operand_literal = operand.accept(LiteralExtractor())
        
        if operand_literal is not None:
            # Evaluate at compile-time
            result = self._eval_unary_op(expr.operator, operand_literal)
            if result is not None:
                self.fold_count += 1
                return LiteralIR(
                    value=result,
                    location=expr.location,
                    type_info=expr.type_info,
                    shape_info=expr.shape_info
                )
        
        # Not constant, return (potentially partially-folded) expression
        # kind is class variable, not constructor parameter
        return UnaryOpIR(
            operator=expr.operator,
            operand=operand,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_function_call(self, expr) -> ExpressionIR:
        """
        Fold function call if all arguments are constant and function is pure.
        
        Rust Pattern: Visitor pattern for function calls
        """
        folded_args = [arg.accept(self) for arg in expr.arguments]
        return FunctionCallIR(
            callee_expr=expr.callee_expr,
            location=expr.location,
            arguments=folded_args,
            module_path=expr.module_path,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_rectangular_access(self, expr: RectangularAccessIR) -> ExpressionIR:
        """Visit rectangular array access - fold array and indices. Never set an index slot to a list."""
        folded_array = expr.array.accept(self)
        folded_indices = []
        for idx in (expr.indices or []):
            if idx is None:
                raise ValueError("IR index slot is None")
            res = idx.accept(self) if hasattr(idx, 'accept') else idx
            if res is None:
                raise ValueError("IR index slot became None after const folding")
            folded_indices.append(res)
        return RectangularAccessIR(
            array=folded_array,
            indices=folded_indices,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_jagged_access(self, expr) -> ExpressionIR:
        """Visit jagged array access - fold base and index chain"""
        folded_base = expr.base.accept(self)
        folded_chain = []
        for idx in (expr.index_chain or []):
            if idx is None:
                raise ValueError("IR index_chain slot is None")
            res = idx.accept(self) if hasattr(idx, 'accept') else idx
            if res is None:
                raise ValueError("IR index_chain slot became None after const folding")
            folded_chain.append(res)
        return JaggedAccessIR(
            base=folded_base,
            index_chain=folded_chain,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_block_expression(self, expr) -> ExpressionIR:
        """Visit block expression - fold statements and final_expr"""
        folded_statements = [stmt.accept(self) for stmt in expr.statements]
        folded_final_expr = expr.final_expr.accept(self) if expr.final_expr is not None else None
        return BlockExpressionIR(
            statements=folded_statements,
            final_expr=folded_final_expr,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_if_expression(self, expr) -> ExpressionIR:
        """Visit if expression - fold condition and prune dead branches (DCE)."""
        folded_condition = expr.condition.accept(self)
        # DCE: if condition is a compile-time constant, eliminate dead branch
        if isinstance(folded_condition, LiteralIR) and isinstance(folded_condition.value, (bool, int)):
            is_true = bool(folded_condition.value)
            if is_true:
                return expr.then_expr.accept(self)
            else:
                if expr.else_expr is not None:
                    return expr.else_expr.accept(self)
        folded_then = expr.then_expr.accept(self)
        folded_else = expr.else_expr.accept(self) if expr.else_expr else None
        return IfExpressionIR(
            condition=folded_condition,
            then_expr=folded_then,
            else_expr=folded_else,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_lambda(self, expr) -> ExpressionIR:
        """Visit lambda - fold body"""
        folded_body = expr.body.accept(self)
        # kind is class variable, not constructor parameter
        return LambdaIR(
            parameters=expr.parameters,
            body=folded_body,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_binding(self, node: BindingIR) -> ExpressionIR:
        if is_function_binding(node):
            return node
        elif is_einstein_binding(node):
            raise NotImplementedError("ConstantFolder only handles expressions, not Einstein declarations")
        else:
            folded_value = node.expr.accept(self) if hasattr(node, 'expr') and node.expr else None
            return BindingIR(
                name=getattr(node, 'name', ''),
                expr=folded_value,
                type_info=getattr(node, 'type_info', None),
                location=getattr(node, 'location', None),
                defid=getattr(node, 'defid', None),
            )
    
    def visit_range(self, expr) -> ExpressionIR:
        """Visit range expression - cannot fold"""
        return expr
    
    def visit_array_comprehension(self, expr) -> ExpressionIR:
        """Visit array comprehension - fold body, ranges, and constraints"""
        from ..ir.nodes import ArrayComprehensionIR
        folded_body = expr.body.accept(self)
        folded_ranges = [r.accept(self) for r in expr.ranges]
        # Fold constraints as well
        folded_constraints = [c.accept(self) for c in expr.constraints] if expr.constraints else []
        return ArrayComprehensionIR(
            body=folded_body,
            loop_vars=expr.loop_vars,
            ranges=folded_ranges,
            constraints=folded_constraints,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_array_literal(self, expr: ArrayLiteralIR) -> ExpressionIR:
        """Visit array literal - fold elements"""
        folded_elements = [elem.accept(self) for elem in expr.elements]
        return ArrayLiteralIR(
            elements=folded_elements,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_tuple_expression(self, expr: TupleExpressionIR) -> ExpressionIR:
        """Visit tuple expression - fold elements"""
        folded_elements = [elem.accept(self) for elem in expr.elements]
        return TupleExpressionIR(
            elements=folded_elements,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_tuple_access(self, expr: TupleAccessIR) -> ExpressionIR:
        """Visit tuple access - fold tuple expression"""
        folded_tuple = expr.tuple_expr.accept(self)
        return TupleAccessIR(
            tuple_expr=folded_tuple,
            index=expr.index,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_interpolated_string(self, expr: InterpolatedStringIR) -> ExpressionIR:
        """Visit interpolated string - fold expression parts"""
        folded_parts = []
        for part in expr.parts:
            if isinstance(part, str):
                folded_parts.append(part)
            else:
                folded_parts.append(part.accept(self))
        return InterpolatedStringIR(
            parts=folded_parts,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_cast_expression(self, expr: CastExpressionIR) -> ExpressionIR:
        """Visit cast expression - fold operand"""
        folded_expr = expr.expr.accept(self)
        return CastExpressionIR(
            expr=folded_expr,
            target_type=expr.target_type,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_member_access(self, expr: MemberAccessIR) -> ExpressionIR:
        """Visit member access - fold object"""
        folded_object = expr.object.accept(self)
        return MemberAccessIR(
            object=folded_object,
            member=expr.member,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_try_expression(self, expr: TryExpressionIR) -> ExpressionIR:
        """Visit try expression - fold operand"""
        folded_operand = expr.operand.accept(self)
        return TryExpressionIR(
            operand=folded_operand,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_match_expression(self, expr: MatchExpressionIR) -> ExpressionIR:
        """Visit match expression - fold scrutinee and arms"""
        from ..ir.nodes import MatchArmIR
        folded_scrutinee = expr.scrutinee.accept(self)
        folded_arms = []
        for arm in expr.arms:
            folded_body = arm.body.accept(self)
            folded_arms.append(MatchArmIR(pattern=arm.pattern, body=folded_body))
        return MatchExpressionIR(
            scrutinee=folded_scrutinee,
            arms=folded_arms,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_reduction_expression(self, expr: ReductionExpressionIR) -> ExpressionIR:
        """Visit reduction expression - fold body"""
        folded_body = expr.body.accept(self)
        folded_where = None
        if expr.where_clause:
            from ..ir.nodes import WhereClauseIR
            folded_constraints = [c.accept(self) for c in expr.where_clause.constraints]
            folded_where = WhereClauseIR(constraints=folded_constraints, ranges=expr.where_clause.ranges)
        # CRITICAL: Preserve loop_var_ranges - they're needed by the backend
        # Also fold the range expressions (start/end) if they're constant
        folded_loop_var_ranges = {}
        if expr.loop_var_ranges:
            for loop_var_defid, range_ir in expr.loop_var_ranges.items():
                if range_ir:
                    folded_start = range_ir.start.accept(self)
                    folded_end = range_ir.end.accept(self)
                    from ..ir.nodes import RangeIR
                    folded_loop_var_ranges[loop_var_defid] = RangeIR(
                        start=folded_start,
                        end=folded_end,
                        location=range_ir.location
                    )
        return ReductionExpressionIR(
            operation=expr.operation,
            loop_vars=expr.loop_vars,
            body=folded_body,
            where_clause=folded_where,
            loop_var_ranges=folded_loop_var_ranges if folded_loop_var_ranges else expr.loop_var_ranges,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_where_expression(self, expr: WhereExpressionIR) -> ExpressionIR:
        """Visit where expression - fold expr and constraints"""
        from ..ir.nodes import WhereClauseIR
        folded_expr = expr.expr.accept(self)
        folded_constraints = [c.accept(self) for c in expr.constraints]
        return WhereExpressionIR(
            expr=folded_expr,
            constraints=folded_constraints,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_arrow_expression(self, expr: ArrowExpressionIR) -> ExpressionIR:
        """Visit arrow expression - fold components"""
        folded_components = [comp.accept(self) for comp in expr.components]
        return ArrowExpressionIR(
            components=folded_components,
            operator=expr.operator,
            location=expr.location,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_pipeline_expression(self, expr: PipelineExpressionIR) -> ExpressionIR:
        """Visit pipeline expression - fold left and right"""
        folded_left = expr.left.accept(self)
        folded_right = expr.right.accept(self)
        return PipelineExpressionIR(
            left=folded_left,
            right=folded_right,
            location=expr.location,
            operator=getattr(expr, 'operator', '|>'),
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    def visit_builtin_call(self, expr: BuiltinCallIR) -> ExpressionIR:
        """Visit builtin call - fold arguments"""
        folded_args = [arg.accept(self) for arg in expr.args]
        return BuiltinCallIR(
            builtin_name=expr.builtin_name,
            args=folded_args,
            location=expr.location,
            defid=expr.defid,
            type_info=expr.type_info,
            shape_info=expr.shape_info
        )
    
    # Pattern visitors (no-op, patterns don't fold)
    def visit_literal_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_identifier_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_wildcard_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_tuple_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_array_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_rest_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_guard_pattern(self, node) -> ExpressionIR:
        raise NotImplementedError("ConstantFolder only handles expressions, not patterns")
    
    def visit_module(self, node) -> ExpressionIR:
        """Modules are not expressions - should not be called"""
        raise NotImplementedError("ConstantFolder only handles expressions, not modules")
    
    def _eval_binary_op(self, op: str, left: Any, right: Any) -> Optional[Any]:
        """
        Evaluate binary operation at compile-time.
        
        Rust Pattern: Constant evaluation for binary operations
        """
        try:
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                if right == 0:
                    return None  # Division by zero
                return left / right
            # ... other operators
        except Exception:
            return None
    
    def _eval_unary_op(self, op: str, operand: Any) -> Optional[Any]:
        """
        Evaluate unary operation at compile-time.
        
        Rust Pattern: Constant evaluation for unary operations
        """
        try:
            if op == '-':
                return -operand
            elif op == '!':
                return not operand
            # ... other operators
        except Exception:
            return None


class ConstantFoldingVisitor(IRVisitor[None]):
    """
    Visitor for constant folding statements (Rust pattern: visitor for IR transformation).
    
    Rust Pattern: Visitor pattern for IR transformation (no isinstance)
    
    Implementation Alignment: Follows Rust's visitor pattern:
    - Visitor pattern for statement processing (no isinstance)
    - Type-safe dispatch
    - Modifies nodes in place
    - Uses ConstantFolder (which is also a visitor) for expression folding
    
    Reference: Rust visitor patterns for IR transformation
    """
    
    def __init__(self, folder: ConstantFolder):
        self.folder = folder  # ConstantFolder is also a visitor (IRVisitor[ExpressionIR])
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and fold all functions, constants, and statements in place."""
        # Fold all functions in place
        for func in node.functions:
            func.accept(self)
        
        # Fold all constants in place
        for const in node.constants:
            const.accept(self)
        
        # Fold top-level statements in place - use visitor pattern
        for stmt in node.statements:
            stmt.accept(self)

    def visit_binding(self, stmt: BindingIR) -> None:
        if is_einstein_binding(stmt):
            for clause in stmt.clauses or []:
                if getattr(clause, "value", None):
                    folded_value = clause.value.accept(self.folder)
                    clause.value = folded_value
        elif is_function_binding(stmt):
            folded_body = stmt.body.accept(self.folder)
            if hasattr(stmt, 'expr') and stmt.expr is not None:
                object.__setattr__(stmt.expr, 'body', folded_body)
            else:
                object.__setattr__(stmt, 'body', folded_body)
        elif is_constant_binding(stmt):
            folded_value = stmt.value.accept(self.folder)
            object.__setattr__(stmt, 'expr', folded_value)
        else:
            if hasattr(stmt, 'expr') and stmt.expr is not None:
                stmt.expr.accept(self.folder)
    
    # Required visitor methods (for IRVisitor interface) - void visitor, no-op for expressions
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node) -> None:
        pass
    
    def visit_function_call(self, node) -> None:
        pass
    
    def visit_unary_op(self, node) -> None:
        pass
    
    def visit_rectangular_access(self, node) -> None:
        pass
    
    def visit_jagged_access(self, node) -> None:
        pass
    
    def visit_block_expression(self, node) -> None:
        pass
    
    def visit_if_expression(self, node) -> None:
        pass
    
    def visit_lambda(self, node) -> None:
        pass
    
    def visit_range(self, node) -> None:
        pass
    
    def visit_array_comprehension(self, node) -> None:
        pass
    
    def visit_array_literal(self, node) -> None:
        pass
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_cast_expression(self, node) -> None:
        pass
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_reduction_expression(self, node) -> None:
        pass
    
    def visit_where_expression(self, node) -> None:
        pass
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_literal_pattern(self, node) -> None:
        pass
    
    def visit_identifier_pattern(self, node) -> None:
        pass
    
    def visit_wildcard_pattern(self, node) -> None:
        pass
    
    def visit_tuple_pattern(self, node) -> None:
        pass
    
    def visit_array_pattern(self, node) -> None:
        pass
    
    def visit_rest_pattern(self, node) -> None:
        pass
    
    def visit_guard_pattern(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass


class LiteralExtractor(IRVisitor[Optional[Any]]):
    """
    Extract literal value from expression (Rust pattern: visitor for value extraction).
    
    Rust Pattern: Visitor pattern for value extraction (no isinstance)
    
    Implementation Alignment: Follows Rust's visitor pattern:
    - Visitor pattern for value extraction (no isinstance)
    - Type-safe dispatch
    - Returns literal value if expression is a literal, None otherwise
    """
    
    def visit_program(self, node: ProgramIR) -> Optional[Any]:
        """Visit program - not used for literal extraction (expressions only)"""
        return None
    
    def visit_literal(self, expr: LiteralIR) -> Optional[Any]:
        """Extract literal value"""
        return expr.value
    
    def visit_identifier(self, expr) -> Optional[Any]:
        """Identifiers are not literals"""
        return None
    
    def visit_binary_op(self, expr) -> Optional[Any]:
        """Binary ops are not literals"""
        return None
    
    def visit_function_call(self, expr) -> Optional[Any]:
        """Function calls are not literals"""
        return None
    
    def visit_unary_op(self, expr) -> Optional[Any]:
        """Unary ops are not literals"""
        return None
    
    def visit_rectangular_access(self, expr) -> Optional[Any]:
        """Array access is not literal"""
        return None
    
    def visit_jagged_access(self, expr) -> Optional[Any]:
        """Array access is not literal"""
        return None
    
    def visit_block_expression(self, expr) -> Optional[Any]:
        """Block expressions are not literals"""
        return None
    
    def visit_if_expression(self, expr) -> Optional[Any]:
        """If expressions are not literals"""
        return None
    
    def visit_lambda(self, expr) -> Optional[Any]:
        """Lambdas are not literals"""
        return None
    
    def visit_binding(self, node) -> Optional[Any]:
        if is_function_binding(node):
            return None
        elif is_einstein_binding(node):
            return None
        else:
            if hasattr(node, 'value') and node.value:
                return node.value.accept(self)
            return None
    
    def visit_range(self, expr) -> Optional[Any]:
        """Range expressions are not literals"""
        return None
    
    def visit_array_comprehension(self, expr) -> Optional[Any]:
        """Array comprehensions are not literals"""
        return None
    
    def visit_array_literal(self, expr) -> Optional[Any]:
        """Array literals are not simple literals (need evaluation)"""
        return None
    
    def visit_tuple_expression(self, expr) -> Optional[Any]:
        """Tuple expressions are not literals"""
        return None
    
    def visit_tuple_access(self, expr) -> Optional[Any]:
        """Tuple access is not literal"""
        return None
    
    def visit_interpolated_string(self, expr) -> Optional[Any]:
        """Interpolated strings are not literals"""
        return None
    
    def visit_cast_expression(self, expr) -> Optional[Any]:
        """Cast expressions are not literals"""
        return None
    
    def visit_member_access(self, expr) -> Optional[Any]:
        """Member access is not literal"""
        return None
    
    def visit_try_expression(self, expr) -> Optional[Any]:
        """Try expressions are not literals"""
        return None
    
    def visit_match_expression(self, expr) -> Optional[Any]:
        """Match expressions are not literals"""
        return None
    
    def visit_reduction_expression(self, expr) -> Optional[Any]:
        """Reduction expressions are not literals"""
        return None
    
    def visit_where_expression(self, expr) -> Optional[Any]:
        """Where expressions are not literals"""
        return None
    
    def visit_arrow_expression(self, expr) -> Optional[Any]:
        """Arrow expressions are not literals"""
        return None
    
    def visit_pipeline_expression(self, expr) -> Optional[Any]:
        """Pipeline expressions are not literals"""
        return None
    
    def visit_builtin_call(self, expr) -> Optional[Any]:
        """Builtin calls are not literals"""
        return None
    
    def visit_literal_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_identifier_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_wildcard_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_tuple_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_array_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_rest_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_guard_pattern(self, node) -> Optional[Any]:
        """Patterns are not literals"""
        return None
    
    def visit_unary_op(self, expr) -> Optional[Any]:
        """Unary ops are not literals"""
        return None
    
    def visit_module(self, node) -> Optional[Any]:
        """Modules are not literals"""
        return None


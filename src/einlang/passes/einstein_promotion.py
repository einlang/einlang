"""
Einstein Promotion Pass

Detects VariableDeclaration nodes whose RHS is an Einstein expression
(reduction or product of rectangular accesses with free indices) and
promotes them to EinsteinDeclaration with auto-derived LHS index order.

Default rule: free indices are ordered by first occurrence when traversing
the expression body in left-to-right depth-first order, skipping
reduction/contraction indices.
"""

from typing import List, Optional, Dict, Tuple, Set
import logging

logger = logging.getLogger("einlang.passes.einstein_promotion")


def try_promote_to_einstein(var_decl, scope=None) -> Optional["EinsteinDeclaration"]:
    """
    Attempt to promote a VariableDeclaration to an EinsteinDeclaration.

    Returns an EinsteinDeclaration if promotion is possible, None otherwise.

    Handles:
    1. Explicit reduction with omitted LHS: let C = sum[k](A[i,k] * B[k,j])
    2. Element-wise Einstein expression: let C = B[i] + C[j]

    `scope` is an optional Scope object used to filter out index identifiers that
    are already declared in an enclosing scope (e.g., Einstein indices from a
    parent EinsteinDeclaration). Free indices that resolve in the scope chain
    are excluded from the promotion LHS.
    """
    value = var_decl.value
    if value is None:
        return None

    from ..shared.nodes import (
        ReductionExpression,
        EinsteinDeclaration,
        EinsteinClause,
        IndexVar,
        WhereExpression,
        CastExpression,
    )

    # Unwrap WhereExpression to find the core expression (e.g. sum[i](A[i]) where cond)
    core = value
    if isinstance(value, WhereExpression):
        core = value.expr

    # Collect all index identifiers in the expression with their positions.
    # Scan the original value (including where clause body) for index references.
    scan_results: List[Tuple[str, int]] = []  # (name, position)
    _scan_index_identifiers(value, scan_results)

    if not scan_results:
        return None

    # Build name → first position map
    first_pos: Dict[str, int] = {}
    all_names: List[str] = []
    for name, pos in scan_results:
        if name not in first_pos:
            first_pos[name] = pos
            all_names.append(name)

    if isinstance(core, ReductionExpression):
        # Explicit reduction: reduction vars from over_clause, rest are free
        contracted = _extract_reduction_var_names(core)
        free = [n for n in all_names if n not in contracted]
    elif _is_compound_einstein_expression(value):
        # Element-wise Einstein expression: a compound expression (binary,
        # unary, etc.) containing RectangularAccess with identifier indices.
        # e.g., B[i] + C[j], (B[i] + C[j]) ** 2
        #
        # Exclude any identifiers that belong to reductions nested inside
        # the expression (e.g. sum[n](A[n]) / 2 — n is a reduction var,
        # not a free Einstein index).
        contracted = _extract_reduction_var_names(value)
        free = [n for n in all_names if n not in contracted]
    else:
        # Single array access like `arr[i]` — could be a regular variable
        # index, don't promote.
        return None

    # Exclude free indices that are already declared in an enclosing scope
    # (e.g., Einstein indices from a parent EinsteinDeclaration, function
    # parameters, or previously declared variables). These are references to
    # existing bindings, not new Einstein output indices.
    if scope is not None:
        free = [n for n in free if scope.lookup(n) is None]
    if not free:
        # All indices are contracted → scalar result.
        # For explicit reductions, the reduction's own REDUCTION scope in name
        # resolution handles the contraction. For implicit contractions, the
        # backend doesn't support zero-index Einstein declarations yet.
        # In both cases, keep as a regular var_decl.
        return None

    # Sort free indices by first occurrence position
    free.sort(key=lambda n: first_pos[n])

    # Build the LHS indices
    lhs_indices: List[IndexVar] = []
    for name in free:
        lhs_indices.append(IndexVar(name=name, range_expr=None, location=var_decl.location))

    # Build the clause value
    clause_value = value

    clause = EinsteinClause(
        indices=lhs_indices,
        value=clause_value,
        location=var_decl.location,
    )

    return EinsteinDeclaration(
        array_name=var_decl.name,
        clauses=[clause],
        location=var_decl.location,
    )


def promote_inline_einstein_expressions(ast) -> None:
    """
    AST pre-pass: wrap inline Einstein expressions in implicit VariableDeclarations
    so that the name resolver can promote them to EinsteinDeclarations.

    Handles:
    - Expression statements: (B[i] + C[j]) ** 2; → let __ein_N = (B[i] + C[j]) ** 2;
    - Block final expressions: { ...; (B[i] + C[j]) ** 2 } → { ...; let __ein_N = (B[i] + C[j]) ** 2; __ein_N }
    """
    from ..shared.nodes import (
        BlockExpression,
        ExpressionStatement,
        VariableDeclaration,
        Identifier as ASTIdentifier,
    )

    _generated_counter = [0]  # mutable counter for unique names

    def _gen_name():
        name = f"__ein_{_generated_counter[0]}"
        _generated_counter[0] += 1
        return name

    def _needs_promotion(expr) -> bool:
        """Check if an expression contains Einstein patterns that need promotion."""
        scan = []
        _scan_index_identifiers(expr, scan)
        if not scan:
            return False
        # Check if there are any free indices
        from ..shared.nodes import ReductionExpression
        core = expr
        from ..shared.nodes import WhereExpression
        if isinstance(expr, WhereExpression):
            core = expr.expr
        if isinstance(core, ReductionExpression):
            contracted = _extract_reduction_var_names(core)
            all_names = list(dict.fromkeys(n for n, _ in scan))
            free = [n for n in all_names if n not in contracted]
            return len(free) > 0
        if _is_compound_einstein_expression(expr):
            contracted = _extract_reduction_var_names(expr)
            all_names = list(dict.fromkeys(n for n, _ in scan))
            free = [n for n in all_names if n not in contracted]
            return len(free) > 0
        return False

    def _rewrite_block(block) -> None:
        """Rewrite statements and final_expr in a block to promote inline Einstein expressions."""
        stmts = list(block.statements or [])
        new_stmts = []
        for stmt in stmts:
            if isinstance(stmt, ExpressionStatement) and stmt.expr is not None:
                if _needs_promotion(stmt.expr):
                    name = _gen_name()
                    loc = stmt.location
                    var_decl = VariableDeclaration(
                        pattern=name,
                        type_annotation=None,
                        value=stmt.expr,
                        location=loc,
                    )
                    new_stmts.append(var_decl)
                    # The expression statement is consumed by the var_decl
                    # (no separate reference needed since it's a statement)
                else:
                    new_stmts.append(stmt)
            elif isinstance(stmt, BlockExpression):
                _rewrite_block(stmt)
                new_stmts.append(stmt)
            else:
                new_stmts.append(stmt)

        final_expr = block.final_expr
        if final_expr is not None and _needs_promotion(final_expr):
            name = _gen_name()
            loc = final_expr.location
            var_decl = VariableDeclaration(
                pattern=name,
                type_annotation=None,
                value=final_expr,
                location=loc,
            )
            new_stmts.append(var_decl)
            final_expr = ASTIdentifier(name=name, location=loc)

        object.__setattr__(block, 'statements', new_stmts)
        if final_expr is not None:
            object.__setattr__(block, 'final_expr', final_expr)

    from ..shared.nodes import Program as ASTProgram, FunctionDefinition, IfExpression
    from ..shared.nodes import BlockExpression as ASTBlockExpression

    def _walk_stmts(stmts):
        for stmt in stmts:
            if isinstance(stmt, ASTBlockExpression):
                _rewrite_block(stmt)
            elif isinstance(stmt, ExpressionStatement) and stmt.expr is not None:
                # Top-level expression statement: wrap in implicit binding
                # Actually, for top-level, we handle this by converting to
                # a VariableDeclaration which will be promoted
                pass  # handled by _rewrite_block for block contents
            elif isinstance(stmt, FunctionDefinition):
                # Skip function bodies — coordinate function signatures
                # already define which indices are symbolic/contracted.
                pass
            elif isinstance(stmt, IfExpression):
                if isinstance(stmt.then_block, ASTBlockExpression):
                    _rewrite_block(stmt.then_block)
                if isinstance(stmt.else_block, ASTBlockExpression):
                    _rewrite_block(stmt.else_block)

    _walk_stmts(ast.statements)

    # Also handle the top-level program as a block
    new_stmts = []
    for stmt in ast.statements:
        if isinstance(stmt, ExpressionStatement) and stmt.expr is not None:
            if _needs_promotion(stmt.expr):
                name = _gen_name()
                loc = stmt.location
                var_decl = VariableDeclaration(
                    pattern=name,
                    type_annotation=None,
                    value=stmt.expr,
                    location=loc,
                )
                new_stmts.append(var_decl)
            else:
                new_stmts.append(stmt)
        else:
            new_stmts.append(stmt)
    object.__setattr__(ast, 'statements', new_stmts)


def _scan_index_identifiers(expr, results: List[Tuple[str, int]]) -> None:
    """
    Recursively scan an expression for RectangularAccess nodes and collect
    all Identifier indices with their first-occurrence positions.

    Traversal is left-to-right depth-first (the natural reading order).
    """
    from ..shared.nodes import (
        RectangularAccess,
        Identifier,
        BinaryExpression,
        UnaryExpression,
        ReductionExpression,
        FunctionCall,
        IfExpression,
        BlockExpression,
        ArrayLiteral,
        ArrayComprehension,
        CastExpression,
        LambdaExpression,
        WhereExpression,
    )

    if isinstance(expr, RectangularAccess):
        for idx in (expr.indices or []):
            if isinstance(idx, Identifier):
                results.append((idx.name, len(results)))
            else:
                _scan_index_identifiers(idx, results)
    elif isinstance(expr, BinaryExpression):
        _scan_index_identifiers(expr.left, results)
        _scan_index_identifiers(expr.right, results)
    elif isinstance(expr, UnaryExpression):
        _scan_index_identifiers(expr.operand, results)
    elif isinstance(expr, ReductionExpression):
        # Scan the body but NOT the over_clause (reduction vars are handled separately)
        _scan_index_identifiers(expr.body, results)
    elif isinstance(expr, FunctionCall):
        for arg in expr.arguments or []:
            _scan_index_identifiers(arg, results)
    elif isinstance(expr, IfExpression):
        _scan_index_identifiers(expr.condition, results)
        if expr.then_block:
            _scan_index_identifiers(expr.then_block, results)
        if expr.else_block:
            _scan_index_identifiers(expr.else_block, results)
    elif isinstance(expr, BlockExpression):
        for stmt in expr.statements or []:
            _scan_index_identifiers(stmt, results)
        if expr.final_expr:
            _scan_index_identifiers(expr.final_expr, results)
    elif isinstance(expr, ArrayLiteral):
        for elem in expr.elements or []:
            _scan_index_identifiers(elem, results)
    elif isinstance(expr, ArrayComprehension):
        _scan_index_identifiers(expr.expr, results)
    elif isinstance(expr, CastExpression):
        _scan_index_identifiers(expr.expr, results)
    elif isinstance(expr, LambdaExpression):
        _scan_index_identifiers(expr.body, results)
    elif isinstance(expr, WhereExpression):
        _scan_index_identifiers(expr.expr, results)
    # Literal, Identifier (standalone), etc. → no indices to scan


def _extract_reduction_var_names(reduction_expr) -> Set[str]:
    """Extract ALL reduction variable names from a ReductionExpression, including nested ones."""
    from ..shared.nodes import ReductionExpression
    names: Set[str] = set()
    _extract_reduction_var_names_impl(reduction_expr, names)
    return names


def _extract_reduction_var_names_impl(expr, names: Set[str]) -> None:
    """Recursively extract reduction variable names from an expression."""
    from ..shared.nodes import (
        ReductionExpression,
        BinaryExpression,
        UnaryExpression,
        WhereExpression,
        CastExpression,
    )
    if isinstance(expr, ReductionExpression):
        over = expr.over_clause
        if over and over.range_groups:
            for group in over.range_groups:
                for v in group.variables:
                    name = v if isinstance(v, str) else getattr(v, 'name', str(v))
                    if name:
                        names.add(str(name))
        # Also scan nested reductions in the body
        _extract_reduction_var_names_impl(expr.body, names)
    elif isinstance(expr, BinaryExpression):
        _extract_reduction_var_names_impl(expr.left, names)
        _extract_reduction_var_names_impl(expr.right, names)
    elif isinstance(expr, UnaryExpression):
        _extract_reduction_var_names_impl(expr.operand, names)
    elif isinstance(expr, CastExpression):
        _extract_reduction_var_names_impl(expr.expr, names)
    elif isinstance(expr, WhereExpression):
        _extract_reduction_var_names_impl(expr.expr, names)


def _is_compound_einstein_expression(expr) -> bool:
    """
    Check if an expression is an algebraic compound expression containing
    RectangularAccess with Identifier indices, indicating an element-wise
    Einstein expression.

    Returns True for expressions like B[i] + C[j], (B[i] + C[j]) ** 2, -arr[i].
    Returns False for simple expressions like arr[i], container expressions
    like [sum[j](row[j]) | row in nested], and function calls.

    Only BinaryExpression, UnaryExpression, and CastExpression are considered
    "algebraic" — container types (ArrayComprehension, BlockExpression,
    FunctionCall, IfExpression, etc.) are never promoted.
    """
    from ..shared.nodes import (
        RectangularAccess,
        BinaryExpression,
        UnaryExpression,
        CastExpression,
        WhereExpression,
    )

    core = expr
    if isinstance(expr, WhereExpression):
        core = expr.expr

    # A standalone RectangularAccess is not compound (could be var access)
    if isinstance(core, RectangularAccess):
        return False

    # Only promote algebraic expressions, not containers
    if not isinstance(core, (BinaryExpression, UnaryExpression, CastExpression)):
        return False

    # Check if the expression contains any RectangularAccess with Identifier indices
    return _contains_rectangular_access_with_identifier(core)


def _contains_rectangular_access_with_identifier(expr) -> bool:
    """Quick check: does the expression tree contain any RectangularAccess
    with at least one Identifier index?"""
    from ..shared.nodes import (
        RectangularAccess,
        Identifier,
        BinaryExpression,
        UnaryExpression,
        ReductionExpression,
        FunctionCall,
        IfExpression,
        BlockExpression,
        ArrayLiteral,
        ArrayComprehension,
        CastExpression,
        LambdaExpression,
        WhereExpression,
    )

    if isinstance(expr, RectangularAccess):
        for idx in (expr.indices or []):
            if isinstance(idx, Identifier):
                return True
        return False
    elif isinstance(expr, BinaryExpression):
        return (_contains_rectangular_access_with_identifier(expr.left) or
                _contains_rectangular_access_with_identifier(expr.right))
    elif isinstance(expr, UnaryExpression):
        return _contains_rectangular_access_with_identifier(expr.operand)
    elif isinstance(expr, ReductionExpression):
        return _contains_rectangular_access_with_identifier(expr.body)
    elif isinstance(expr, FunctionCall):
        for arg in expr.arguments or []:
            if _contains_rectangular_access_with_identifier(arg):
                return True
        return False
    elif isinstance(expr, IfExpression):
        if _contains_rectangular_access_with_identifier(expr.condition):
            return True
        if expr.then_block and _contains_rectangular_access_with_identifier(expr.then_block):
            return True
        if expr.else_block and _contains_rectangular_access_with_identifier(expr.else_block):
            return True
        return False
    elif isinstance(expr, BlockExpression):
        for stmt in expr.statements or []:
            if _contains_rectangular_access_with_identifier(stmt):
                return True
        if expr.final_expr and _contains_rectangular_access_with_identifier(expr.final_expr):
            return True
        return False
    elif isinstance(expr, ArrayLiteral):
        for elem in expr.elements or []:
            if _contains_rectangular_access_with_identifier(elem):
                return True
        return False
    elif isinstance(expr, ArrayComprehension):
        return _contains_rectangular_access_with_identifier(expr.expr)
    elif isinstance(expr, CastExpression):
        return _contains_rectangular_access_with_identifier(expr.expr)
    elif isinstance(expr, LambdaExpression):
        return _contains_rectangular_access_with_identifier(expr.body)
    elif isinstance(expr, WhereExpression):
        return _contains_rectangular_access_with_identifier(expr.expr)
    else:
        return False

"""
Merge @fn (diff rule) into the corresponding fn before name resolution.

So there is a single function node per name and a single DefId; IR passes
do not need to match @fn to fn by DefId. Runs on AST only.
"""

from typing import List, Any

from ..shared.nodes import FunctionDefinition, DiffRuleDef


def merge_diff_rules_into_functions(ast: Any) -> None:
    """
    In-place: for each DiffRuleDef (@fn name { body }), attach body to the
    FunctionDefinition with the same name that appears before it in the
    statement list, then remove the DiffRuleDef from statements.

    Must run before name resolution so one DefId is allocated per function.
    """
    statements = getattr(ast, "statements", None)
    if not statements:
        return
    new_statements: List[Any] = []
    for i, stmt in enumerate(statements):
        if isinstance(stmt, DiffRuleDef):
            # Find the most recent FunctionDefinition with this name before this @fn
            target = None
            for j in range(i - 1, -1, -1):
                prev = statements[j]
                if isinstance(prev, FunctionDefinition) and prev.name == stmt.name:
                    target = prev
                    break
            if target is not None:
                object.__setattr__(target, "custom_diff_body", stmt.body)
            # Do not append the DiffRuleDef (it is merged)
            continue
        new_statements.append(stmt)
    object.__setattr__(ast, "statements", new_statements)

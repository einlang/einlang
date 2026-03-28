from einlang.ir.nodes import (
    BinaryOpIR,
    BindingIR,
    IdentifierIR,
    IndexVarIR,
    LiteralIR,
    LoweredEinsteinClauseIR,
    LoweredEinsteinIR,
    ProgramIR,
    WhereClauseIR,
)
from einlang.shared.nodes import (
    BinaryExpression,
    BlockExpression,
    FunctionDefinition,
    Identifier,
    Literal,
    OverClause,
    Parameter,
    RangeGroup,
    ReductionExpression,
    SourceLocation as ASTSourceLocation,
    VariableDeclaration,
    WhereClause,
)
from einlang.shared.source_location import SourceLocation as IRSourceLocation
from einlang.shared.types import BinaryOp, I32


def test_ast_function_definition_str_mimics_source() -> None:
    loc = ASTSourceLocation("test.ein", 1, 1)
    body = BlockExpression(
        statements=[VariableDeclaration("y", Literal(1, loc), location=loc)],
        final_expr=Identifier("x", loc),
        location=loc,
    )
    fn = FunctionDefinition(
        name="id_plus_local",
        parameters=[Parameter("x", I32)],
        return_type=I32,
        body=body,
        is_public=True,
        location=loc,
    )

    assert str(fn) == "pub fn id_plus_local(x: i32) -> i32 { let y = 1; x }"


def test_ast_reduction_expression_str_mimics_source() -> None:
    loc = ASTSourceLocation("test.ein", 1, 1)
    where_clause = WhereClause.from_list(
        [BinaryExpression(Identifier("i", loc), BinaryOp.LT, Literal(10, loc), location=loc)]
    )
    expr = ReductionExpression(
        function_name="sum",
        body=Identifier("x", loc),
        over_clause=OverClause(range_groups=[RangeGroup(range_expr=None, variables=["i"])]),
        where_clause=where_clause,
        location=loc,
    )

    assert str(expr) == "sum[i](x) where (i < 10)"


def test_ir_lowered_einstein_str_mimics_source() -> None:
    loc = IRSourceLocation("test.ein", 1, 1)
    clause = LoweredEinsteinClauseIR(
        body=IdentifierIR("x", loc),
        indices=[IndexVarIR("i", loc)],
        location=loc,
    )
    lowered = LoweredEinsteinIR(items=[clause], location=loc)

    assert str(lowered) == "{ [i] = x }"


def test_ir_program_and_where_clause_str_mimic_source() -> None:
    loc = IRSourceLocation("test.ein", 1, 1)
    binding = BindingIR("x", LiteralIR(1, loc), location=loc)
    where_clause = WhereClauseIR(
        constraints=[
            BinaryOpIR(
                BinaryOp.LT,
                IdentifierIR("i", loc),
                LiteralIR(4, loc),
                loc,
            )
        ],
        location=loc,
    )
    program = ProgramIR(statements=[binding], location=loc)

    assert str(program) == "let x = 1;"
    assert str(where_clause) == "where i < 4"

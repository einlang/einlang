from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, LoweredEinsteinIR, LoweredReductionIR
from einlang.passes.lowered_execution_facts import LoweredExecutionFactsPass


def _compile(source: str):
    compiler = CompilerDriver()
    result = compiler.compile(
        source.strip(),
        source_file="<lowered_execution_facts>",
        stop_after_pass="LoweredExecutionFactsPass",
    )
    assert result.success, result.get_errors()
    assert result.ir is not None
    assert result.tcx is not None
    return result.ir, result.tcx


def _first_reduction(ir):
    for stmt in ir.statements or []:
        if isinstance(stmt, BindingIR) and isinstance(stmt.expr, LoweredReductionIR):
            return stmt.expr
    raise AssertionError("no lowered reduction found")


def _binding_expr(ir, name: str):
    for stmt in ir.statements or []:
        if isinstance(stmt, BindingIR) and stmt.name == name:
            return stmt.expr
    raise AssertionError(f"no binding named {name!r}")


def test_lowered_execution_facts_annotate_nested_and_if_reductions():
    ir, tcx = _compile(
        """
        let x = [1.0, 2.0, 3.0];
        let y = sum[i in 0..3](
            if x[i] > 0.0 { sum[j in 0..1](x[i]) } else { 0.0 }
        );
        """
    )
    reduction = _first_reduction(ir)
    facts = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_facts_by_id"][reduction.execution_facts_id]

    assert facts.contains_nested_reduction_or_select is True
    assert facts.contains_if_expression is True
    assert facts.contains_lowered_einstein is False


def test_lowered_execution_facts_recognize_matmul_sumprod_kernel():
    ir, tcx = _compile(
        """
        let A = [[1.0, 2.0], [3.0, 4.0]];
        let B = [[5.0, 6.0], [7.0, 8.0]];
        let C[i, j] = sum[k](A[i, k] * B[k, j]);
        """
    )

    plan_map = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_kernel_plans_by_id"]
    plans = list(plan_map.values())
    assert plans, "expected at least one reduction kernel plan"
    kinds = {plan.kind for plan in plans}
    assert "matmul_sumprod" in kinds


def test_lowered_execution_facts_annotate_clause_call_and_nested_lowered_flags():
    ir, tcx = _compile(
        """
        fn inc(v) { v + 1.0 }
        let x = [1.0, 2.0, 3.0];
        let y[i in 0..3] = inc(x[i]);
        let z[i in 0..3] = {
            let t = sum[j in 0..2](x[i]);
            inc(t)
        };
        """
    )

    y_expr = _binding_expr(ir, "y")
    z_expr = _binding_expr(ir, "z")
    assert isinstance(y_expr, LoweredEinsteinIR)
    assert isinstance(z_expr, LoweredEinsteinIR)
    y_clause = y_expr.items[0]
    z_clause = z_expr.items[0]

    facts_map = tcx.get_analysis(LoweredExecutionFactsPass)["clause_facts_by_id"]
    y_facts = facts_map[y_clause.execution_facts_id]
    z_facts = facts_map[z_clause.execution_facts_id]

    assert y_facts.has_literal_index is False
    assert len(y_facts.static_loop_ranges) == 1
    assert y_facts.static_loop_ranges[0] == (0, 3)
    assert y_facts.body_contains_call_using_loop_var is True
    assert tuple(y_facts.call_arg_loop_defids) == tuple(y_facts.loop_defids_nonnull)
    assert y_facts.body_is_elementwise_call is True
    assert y_facts.body_has_direct_nested_lowered_binding is False
    assert tuple(y_facts.loop_defids_nonnull) == tuple(y_facts.loop_names_by_defid.keys())

    assert z_facts.body_contains_call_using_loop_var is False
    assert z_facts.call_arg_loop_defids == ()
    assert len(z_facts.static_loop_ranges) == 1
    assert z_facts.static_loop_ranges[0] == (0, 3)
    assert z_facts.body_is_elementwise_call is False
    assert z_facts.body_has_direct_nested_lowered_binding is True

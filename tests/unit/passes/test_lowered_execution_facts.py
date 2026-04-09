from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, LoweredReductionIR
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

from einlang.compiler.driver import CompilerDriver
from einlang.ir.serialization import serialize_ir
from einlang.ir.nodes import BindingIR, IRNode, LoweredEinsteinIR, LoweredReductionIR
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


def _walk(node):
    if node is None:
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item)
        return
    if not isinstance(node, IRNode):
        return
    yield node
    slots = []
    for cls in type(node).__mro__:
        cls_slots = getattr(cls, "__slots__", ())
        if isinstance(cls_slots, str):
            slots.append(cls_slots)
        else:
            slots.extend(cls_slots)
    seen = set()
    for slot in slots:
        if slot in seen:
            continue
        seen.add(slot)
        yield from _walk(getattr(node, slot, None))


def _first_reduction(ir):
    for node in _walk(ir):
        if isinstance(node, LoweredReductionIR):
            return node
    raise AssertionError("no lowered reduction found")


def _reductions(ir):
    out = []
    seen = set()
    for node in _walk(ir):
        if isinstance(node, LoweredReductionIR) and id(node) not in seen:
            seen.add(id(node))
            out.append(node)
    return out


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
    assert facts.execution_strategy == "vectorized"
    assert reduction.execution_strategy == "vectorized"


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
    reduction = _first_reduction(ir)
    facts = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_facts_by_id"][reduction.execution_facts_id]
    assert reduction.execution_strategy == "matmul_sumprod"
    assert facts.execution_strategy == "matmul_sumprod"


def test_lowered_execution_facts_recognize_windowed_sumprod_kernel():
    ir, tcx = _compile(
        """
        let x[ci in 0..2, t in 0..6] = (1 + ci + t) as f32;
        let w[co in 0..3, ci in 0..2, k in 0..3] = (1 + co + ci + k) as f32;
        let y[co in 0..3, t in 0..4] =
            sum[ci in 0..2, k in 0..3](x[ci, t + k] * w[co, ci, k]);
        """
    )

    plan_map = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_kernel_plans_by_id"]
    plans = list(plan_map.values())
    assert plans, "expected at least one reduction kernel plan"
    kinds = {plan.kind for plan in plans}
    assert "windowed_sumprod" in kinds
    reduction = _first_reduction(ir)
    facts = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_facts_by_id"][reduction.execution_facts_id]
    assert reduction.execution_strategy == "windowed_sumprod"
    assert facts.execution_strategy == "windowed_sumprod"


def test_lowered_execution_facts_recognize_windowed_sumprod_with_symbolic_stride():
    ir, tcx = _compile(
        """
        fn slide(x, w, stride) {
            let y[co in 0..3, t in 0..2] =
                sum[ci in 0..2, k in 0..3](x[ci, t * stride + k] * w[co, ci, k]);
            y
        }
        let x[ci in 0..2, t in 0..6] = (1 + ci + t) as f32;
        let w[co in 0..3, ci in 0..2, k in 0..3] = (1 + co + ci + k) as f32;
        let y = slide(x, w, 3);
        """
    )

    plan_map = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_kernel_plans_by_id"]
    plans = list(plan_map.values())
    assert plans, "expected at least one reduction kernel plan"
    kinds = {plan.kind for plan in plans}
    assert "windowed_sumprod" in kinds
    reductions = _reductions(ir)
    assert any(reduction.execution_strategy == "windowed_sumprod" for reduction in reductions)


def test_lowered_execution_facts_annotate_clause_call_and_nested_lowered_flags():
    ir, tcx = _compile(
        """
        fn inc(v) { v }
        let x = [1.0, 2.0, 3.0];
        let y[i in 0..3] = inc(x[i]);
        let z[i in 0..3] = {
            let t = sum[j in 0..2](x[j]);
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
    assert z_clause.vectorization_strategy == "scalar"
    assert tuple(z_clause.vectorization_scalar_loop_dims or ()) == (0,)
    assert z_facts.vectorization_strategy == "scalar"
    assert z_facts.vectorization_scalar_loop_dims == (0,)


def test_lowered_execution_facts_serialize_clause_vectorization_strategy():
    ir, tcx = _compile(
        """
        fn row(x, i) { x[i] }
        let X = [[1.0, 2.0], [3.0, 4.0]];
        let Y[i in 0..2, j in 0..2] = row(X, i)[j];
        let Z[t in 0..4, j in 0..2] = if t == 0 {
            X[0, j]
        } else {
            Z[t - 1, j]
        };
        """
    )

    y_expr = _binding_expr(ir, "Y")
    z_expr = _binding_expr(ir, "Z")
    assert isinstance(y_expr, LoweredEinsteinIR)
    y_clause = y_expr.items[0]
    z_clause = z_expr.body.items[0]

    facts_map = tcx.get_analysis(LoweredExecutionFactsPass)["clause_facts_by_id"]
    y_facts = facts_map[y_clause.execution_facts_id]
    z_facts = facts_map[z_clause.execution_facts_id]

    assert y_clause.vectorization_strategy == "call-scalar"
    assert tuple(y_clause.vectorization_scalar_loop_dims or ()) == (0,)
    assert y_facts.vectorization_strategy == "call-scalar"
    assert y_facts.vectorization_scalar_loop_dims == (0,)

    assert z_clause.vectorization_strategy == "recurrence-hybrid"
    assert tuple(z_clause.vectorization_scalar_loop_dims or ()) == (0,)
    assert z_facts.vectorization_strategy == "recurrence-hybrid"
    assert z_facts.vectorization_scalar_loop_dims == (0,)

    sexpr = serialize_ir(ir, include_location=False, include_type_info=False)
    assert ":vectorization_strategy" in sexpr
    assert "call-scalar" in sexpr
    assert "recurrence-hybrid" in sexpr


def test_lowered_execution_facts_serialize_reduction_execution_strategy():
    ir, tcx = _compile(
        """
        let x = [1.0, 2.0, 3.0];
        let y = sum[i in 0..3](if x[i] > 0.0 { x[i] } else { 0.0 });
        let A = [[1.0, 2.0], [3.0, 4.0]];
        let B = [[5.0, 6.0], [7.0, 8.0]];
        let z = sum[i in 0..2, j in 0..2](A[i, j] * B[i, j]);
        """
    )

    reductions = _reductions(ir)
    assert len(reductions) == 2
    y_reduction, z_reduction = reductions
    facts_map = tcx.get_analysis(LoweredExecutionFactsPass)["reduction_facts_by_id"]

    assert y_reduction.execution_strategy == "vectorized"
    assert facts_map[y_reduction.execution_facts_id].execution_strategy == "vectorized"
    assert z_reduction.execution_strategy == "matmul_sumprod"
    assert facts_map[z_reduction.execution_facts_id].execution_strategy == "matmul_sumprod"

    sexpr = serialize_ir(ir, include_location=False, include_type_info=False)
    assert ":execution_strategy" in sexpr
    assert "vectorized" in sexpr
    assert "matmul_sumprod" in sexpr

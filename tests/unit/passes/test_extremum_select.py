import numpy as np
import pytest

from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, BlockExpressionIR, EinsteinIR, IfExpressionIR, SelectAtArgmaxIR
from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute


def _compile(source: str, stop_after_pass: str):
    result = CompilerDriver().compile(
        source,
        source_file="<test>",
        stop_after_pass=stop_after_pass,
    )
    assert result.success, result.get_errors()
    assert result.ir is not None
    return result.ir


def _find_specialized_function(program, prefix: str):
    matches = [
        binding
        for binding in (program.bindings or [])
        if isinstance(binding, BindingIR) and isinstance(getattr(binding, "name", None), str) and binding.name.startswith(prefix)
    ]
    assert matches, f"expected a specialized function starting with {prefix!r}"
    return matches[0]


@pytest.mark.parametrize(
    ("source", "prefix", "use_argmin"),
    [
        (
            "use std::array::argmax; let x = [1.0, 3.0, 2.0]; let y = argmax(x);",
            "argmax_",
            False,
        ),
        (
            "use std::array::argmin; let x = [1.0, 3.0, 2.0]; let y = argmin(x);",
            "argmin_",
            True,
        ),
    ],
)
def test_array_extremum_specialization_rewrites_to_select_at_argmax(source, prefix, use_argmin):
    program = _compile(source, "ExtremumSelectionCanonicalizationPass")
    func = _find_specialized_function(program, prefix)

    assert isinstance(func.body, BlockExpressionIR)
    assert isinstance(func.body.final_expr, SelectAtArgmaxIR)
    assert func.body.final_expr.use_argmin is use_argmin

    local_binding_names = [stmt.name for stmt in (func.body.statements or []) if isinstance(stmt, BindingIR)]
    assert local_binding_names == []


@pytest.mark.parametrize(
    ("source", "prefix", "use_argmin"),
    [
        (
            "use std::ml::argmax; let x = [[1.0, 3.0, 2.0]]; let y = argmax(x);",
            "argmax_",
            False,
        ),
        (
            "use std::ml::argmin; let x = [[1.0, 3.0, 2.0]]; let y = argmin(x);",
            "argmin_",
            True,
        ),
    ],
)
def test_ml_extremum_weighted_pattern_rewrites_to_select_at_argmax(source, prefix, use_argmin):
    program = _compile(source, "ExtremumSelectionCanonicalizationPass")
    func = _find_specialized_function(program, prefix)

    assert isinstance(func.body, BlockExpressionIR)
    assert isinstance(func.body.final_expr, IfExpressionIR)
    else_block = func.body.final_expr.else_expr
    assert isinstance(else_block, BlockExpressionIR)

    binding_map = {
        stmt.name: stmt
        for stmt in (else_block.statements or [])
        if isinstance(stmt, BindingIR)
    }

    assert "result" in binding_map
    assert "max_val" not in binding_map
    assert "weighted" not in binding_map
    assert "sentinel" not in binding_map

    result_binding = binding_map["result"]
    assert isinstance(result_binding.expr, EinsteinIR)
    assert len(result_binding.expr.clauses or []) == 1
    assert isinstance(result_binding.expr.clauses[0].value, SelectAtArgmaxIR)
    assert result_binding.expr.clauses[0].value.use_argmin is use_argmin


@pytest.mark.parametrize(
    ("source", "output_name", "expected"),
    [
        (
            "use std::array::argmax; let x = [1.0, 3.0, 2.0]; let y = argmax(x);",
            "y",
            np.array(1, dtype=np.int64),
        ),
        (
            "use std::ml::argmax; let x = [[1.0, 3.0, 2.0], [9.0, 4.0, 9.0]]; let y = argmax(x);",
            "y",
            np.array([1, 0], dtype=np.int64),
        ),
        (
            "use std::ml::argmin; let x = [[1.0, 3.0, 2.0], [9.0, 4.0, 9.0]]; let y = argmin(x);",
            "y",
            np.array([0, 1], dtype=np.int64),
        ),
    ],
)
def test_runtime_extremum_canonicalization_preserves_behavior(source, output_name, expected):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compile_and_execute(source, compiler, runtime, source_file="<test>")

    assert result.success, result.errors if result.errors else result.error
    actual = np.asarray(result.outputs[output_name])
    np.testing.assert_array_equal(actual, expected)


def test_runtime_argmax_outputs_remain_valid_indices():
    source = """
use std::ml::argmax;
let logits = [[1.0, 3.0, 2.0], [0.0, 1.0, 5.0]];
let ids = argmax(logits);
let table = [10, 20, 30];
let picked[i in 0..2] = table[ids[i]];
"""
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compile_and_execute(source, compiler, runtime, source_file="<test>")

    assert result.success, result.errors if result.errors else result.error
    np.testing.assert_array_equal(np.asarray(result.outputs["ids"]), np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(np.asarray(result.outputs["picked"]), np.array([20, 30], dtype=np.int64))


def test_runtime_scalar_argmax_output_remains_valid_index():
    source = """
use std::array::argmax;
let logits = [1.0, 3.0, 2.0];
let idx = argmax(logits);
let labels = [10, 20, 30];
let picked = labels[idx];
"""
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    result = compile_and_execute(source, compiler, runtime, source_file="<test>")

    assert result.success, result.errors if result.errors else result.error
    assert int(result.outputs["idx"]) == 1
    assert int(result.outputs["picked"]) == 20

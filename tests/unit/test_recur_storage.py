from einlang.compiler.driver import CompilerDriver
from einlang.ir.nodes import BindingIR, LoweredRecurrenceIR


def _compile_recurrence_binding(source: str, name: str) -> LoweredRecurrenceIR:
    compiler = CompilerDriver()
    result = compiler.compile(
        source.strip(),
        source_file="<recurrence_storage>",
        stop_after_pass="RecurrenceOrderPass",
    )
    assert result.success, result.get_errors()
    ir = result.ir
    assert ir is not None
    for stmt in ir.statements or []:
        if isinstance(stmt, BindingIR) and stmt.name == name:
            assert isinstance(stmt.expr, LoweredRecurrenceIR), type(stmt.expr).__name__
            return stmt.expr
    raise AssertionError(f"binding {name!r} not found")


def test_recurrence_storage_uses_history_window_for_two_step_recurrence():
    rec = _compile_recurrence_binding(
        """
        let fib[0] = 0.0;
        let fib[1] = 1.0;
        let fib[t in 2..10] = fib[t - 1] + fib[t - 2];
        let last = fib[9];
        last;
        """,
        "fib",
    )
    assert rec.recurrence_output_dim == 0
    assert rec.history_lookback_steps == 2
    assert rec.downstream_tail_steps == 1
    assert rec.preserve_steps == 3
    assert rec.requires_full_output is False


def test_recurrence_storage_expands_for_last_few_downstream_steps():
    rec = _compile_recurrence_binding(
        """
        let x[0] = 1.0;
        let x[t in 1..10] = x[t - 1] + 1.0;
        let recent = [x[7], x[8], x[9]];
        recent;
        """,
        "x",
    )
    assert rec.history_lookback_steps == 1
    assert rec.downstream_tail_steps == 3
    assert rec.preserve_steps == 3
    assert rec.requires_full_output is False


def test_recurrence_storage_falls_back_to_full_output_for_whole_tensor_use():
    rec = _compile_recurrence_binding(
        """
        let x[0] = 1.0;
        let x[t in 1..10] = x[t - 1] + 1.0;
        x;
        """,
        "x",
    )
    assert rec.history_lookback_steps == 1
    assert rec.downstream_tail_steps is None
    assert rec.preserve_steps is None
    assert rec.requires_full_output is True


def test_recurrence_storage_recognizes_symbolic_extent_and_tail_access():
    rec = _compile_recurrence_binding(
        """
        let epochs = 9;
        let x[t in 0..epochs + 1] = if t == 0 {
            1.0
        } else {
            x[t - 1] + 1.0
        };
        let last = x[epochs];
        last;
        """,
        "x",
    )
    assert rec.history_lookback_steps == 1
    assert rec.downstream_tail_steps == 1
    assert rec.preserve_steps == 2
    assert rec.requires_full_output is False


def test_recurrence_storage_keeps_window_for_multidimensional_tensor():
    rec = _compile_recurrence_binding(
        """
        let epochs = 5;
        let width = 4;
        let x[0, i in 0..width] = i as f64;
        let x[t in 1..epochs + 1, i in 0..width] = x[t - 1, i] + 10.0;
        let last = x[epochs, 2];
        last;
        """,
        "x",
    )
    assert rec.recurrence_output_dim == 0
    assert rec.history_lookback_steps == 1
    assert rec.downstream_tail_steps == 1
    assert rec.preserve_steps == 2
    assert rec.requires_full_output is False


def test_recurrence_storage_falls_back_to_full_output_for_non_constant_tail_access():
    rec = _compile_recurrence_binding(
        """
        let x[0] = 1.0;
        let x[t in 1..10] = x[t - 1] + 1.0;
        let idx = x[0] as i32;
        let last = x[idx];
        last;
        """,
        "x",
    )
    assert rec.history_lookback_steps == 1
    assert rec.downstream_tail_steps is None
    assert rec.preserve_steps is None
    assert rec.requires_full_output is True

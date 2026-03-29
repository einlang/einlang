import einlang.backends.numpy_einstein_mixin_setup as recurrence_setup

from einlang.backends.numpy import NumPyBackend
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime

from tests.test_utils import compile_and_execute


def test_runtime_uses_circular_buffer_for_local_tail_only(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let y = {
        let x[0] = 1.0;
        let x[t in 1..10] = x[t - 1] + 1.0;
        x[7] + x[8] + x[9]
    };
    y;
    """

    calls = {"full": 0}
    ring_stats = {
        "created": 0,
        "materialized": 0,
        "full_shape": None,
        "recurrence_dim": None,
        "preserve_steps": None,
    }
    original = NumPyBackend._execute_lowered_recurrence_full
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__
    original_materialize = recurrence_setup._CircularRecurrenceBuffer.materialize

    def _tracking_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_stats["created"] += 1
        ring_stats["full_shape"] = tuple(full_shape)
        ring_stats["recurrence_dim"] = recurrence_dim
        ring_stats["preserve_steps"] = preserve_steps
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    def _tracking_materialize(self):
        ring_stats["materialized"] += 1
        return original_materialize(self)

    def _fail_if_full(self, node, variable_decl):
        calls["full"] += 1
        raise AssertionError("full recurrence path should not run for bounded local tail use")

    monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", _fail_if_full)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _tracking_init)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", _tracking_materialize)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_circular_runtime>")
    finally:
        monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", original)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", original_materialize)

    assert result.success, (result.errors if result.errors else result.error)
    assert result.outputs["y"] == 27.0
    assert calls["full"] == 0
    assert ring_stats["created"] == 1
    assert ring_stats["materialized"] == 0
    assert ring_stats["full_shape"] == (10,)
    assert ring_stats["recurrence_dim"] == 0
    assert ring_stats["preserve_steps"] == 3


def test_runtime_keeps_full_path_when_full_output_is_required(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let x[0] = 1.0;
    let x[t in 1..6] = x[t - 1] + 1.0;
    x;
    """
    ring_created = {"count": 0}
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__

    def _count_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_created["count"] += 1
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _count_init)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_full_runtime>")
    finally:
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
    assert result.success, (result.errors if result.errors else result.error)
    assert list(result.outputs["x"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert ring_created["count"] == 0


def test_runtime_uses_circular_buffer_for_rank2_multi_clause_tail_only(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let y = {
        let state[0, 0] = 1.0;
        let state[0, 1] = 2.0;
        let state[t in 1..6, 0] = state[t - 1, 0] + state[t - 1, 1];
        let state[t in 1..6, 1] = state[t - 1, 1] + 1.0;
        state[4, 0] + state[5, 1]
    };
    y;
    """

    calls = {"full": 0}
    ring_stats = {
        "created": 0,
        "materialized": 0,
        "full_shape": None,
        "recurrence_dim": None,
        "preserve_steps": None,
    }
    original = NumPyBackend._execute_lowered_recurrence_full
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__
    original_materialize = recurrence_setup._CircularRecurrenceBuffer.materialize

    def _tracking_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_stats["created"] += 1
        ring_stats["full_shape"] = tuple(full_shape)
        ring_stats["recurrence_dim"] = recurrence_dim
        ring_stats["preserve_steps"] = preserve_steps
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    def _tracking_materialize(self):
        ring_stats["materialized"] += 1
        return original_materialize(self)

    def _fail_if_full(self, node, variable_decl):
        calls["full"] += 1
        raise AssertionError("full recurrence path should not run for bounded rank-2 tail use")

    monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", _fail_if_full)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _tracking_init)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", _tracking_materialize)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_circular_runtime_rank2>")
    finally:
        monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", original)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", original_materialize)

    assert result.success, (result.errors if result.errors else result.error)
    assert result.outputs["y"] == 22.0
    assert calls["full"] == 0
    assert ring_stats["created"] == 1
    assert ring_stats["materialized"] == 0
    assert ring_stats["full_shape"] == (6, 2)
    assert ring_stats["recurrence_dim"] == 0
    assert ring_stats["preserve_steps"] == 2


def test_runtime_uses_circular_buffer_for_rank2_vectorized_tail_only(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let y = {
        let state[0, i in 0..4] = i;
        let state[t in 1..6, i in 0..4] = state[t - 1, i] + 1;
        state[4, 1] + state[5, 2]
    };
    y;
    """

    calls = {"full": 0}
    ring_stats = {
        "created": 0,
        "materialized": 0,
        "full_shape": None,
        "recurrence_dim": None,
        "preserve_steps": None,
    }
    original = NumPyBackend._execute_lowered_recurrence_full
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__
    original_materialize = recurrence_setup._CircularRecurrenceBuffer.materialize

    def _tracking_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_stats["created"] += 1
        ring_stats["full_shape"] = tuple(full_shape)
        ring_stats["recurrence_dim"] = recurrence_dim
        ring_stats["preserve_steps"] = preserve_steps
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    def _tracking_materialize(self):
        ring_stats["materialized"] += 1
        return original_materialize(self)

    def _fail_if_full(self, node, variable_decl):
        calls["full"] += 1
        raise AssertionError("full recurrence path should not run for bounded rank-2 vectorized tail use")

    monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", _fail_if_full)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _tracking_init)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", _tracking_materialize)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_circular_runtime_rank2_vectorized>")
    finally:
        monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", original)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", original_materialize)

    assert result.success, (result.errors if result.errors else result.error)
    assert result.outputs["y"] == 12.0
    assert calls["full"] == 0
    assert ring_stats["created"] == 1
    assert ring_stats["materialized"] == 0
    assert ring_stats["full_shape"] == (6, 4)
    assert ring_stats["recurrence_dim"] == 0
    assert ring_stats["preserve_steps"] == 2


def test_runtime_uses_circular_buffer_for_rank3_vectorized_tail_only(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let y = {
        let state[0, i in 0..3, j in 0..2] = i + j;
        let state[t in 1..6, i in 0..3, j in 0..2] = state[t - 1, i, j] + 1;
        state[4, 1, 1] + state[5, 2, 1]
    };
    y;
    """

    calls = {"full": 0}
    ring_stats = {
        "created": 0,
        "materialized": 0,
        "full_shape": None,
        "recurrence_dim": None,
        "preserve_steps": None,
    }
    original = NumPyBackend._execute_lowered_recurrence_full
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__
    original_materialize = recurrence_setup._CircularRecurrenceBuffer.materialize

    def _tracking_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_stats["created"] += 1
        ring_stats["full_shape"] = tuple(full_shape)
        ring_stats["recurrence_dim"] = recurrence_dim
        ring_stats["preserve_steps"] = preserve_steps
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    def _tracking_materialize(self):
        ring_stats["materialized"] += 1
        return original_materialize(self)

    def _fail_if_full(self, node, variable_decl):
        calls["full"] += 1
        raise AssertionError("full recurrence path should not run for bounded rank-3 vectorized tail use")

    monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", _fail_if_full)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _tracking_init)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", _tracking_materialize)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_circular_runtime_rank3_vectorized>")
    finally:
        monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", original)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", original_materialize)

    assert result.success, (result.errors if result.errors else result.error)
    assert result.outputs["y"] == 14.0
    assert calls["full"] == 0
    assert ring_stats["created"] == 1
    assert ring_stats["materialized"] == 0
    assert ring_stats["full_shape"] == (6, 3, 2)
    assert ring_stats["recurrence_dim"] == 0
    assert ring_stats["preserve_steps"] == 2


def test_runtime_uses_circular_buffer_for_rank4_vectorized_tail_only(monkeypatch):
    compiler = CompilerDriver()
    runtime = EinlangRuntime(backend="numpy")
    source = """
    let y = {
        let state[0, c in 0..2, i in 0..3, j in 0..2] = c + i + j;
        let state[t in 1..6, c in 0..2, i in 0..3, j in 0..2] = state[t - 1, c, i, j] + 1;
        state[4, 1, 1, 1] + state[5, 1, 2, 1]
    };
    y;
    """

    calls = {"full": 0}
    ring_stats = {
        "created": 0,
        "materialized": 0,
        "full_shape": None,
        "recurrence_dim": None,
        "preserve_steps": None,
    }
    original = NumPyBackend._execute_lowered_recurrence_full
    original_init = recurrence_setup._CircularRecurrenceBuffer.__init__
    original_materialize = recurrence_setup._CircularRecurrenceBuffer.materialize

    def _tracking_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer=None):
        ring_stats["created"] += 1
        ring_stats["full_shape"] = tuple(full_shape)
        ring_stats["recurrence_dim"] = recurrence_dim
        ring_stats["preserve_steps"] = preserve_steps
        return original_init(self, full_shape, dtype, recurrence_dim, preserve_steps, materializer)

    def _tracking_materialize(self):
        ring_stats["materialized"] += 1
        return original_materialize(self)

    def _fail_if_full(self, node, variable_decl):
        calls["full"] += 1
        raise AssertionError("full recurrence path should not run for bounded rank-4 vectorized tail use")

    monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", _fail_if_full)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", _tracking_init)
    monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", _tracking_materialize)
    try:
        result = compile_and_execute(source, compiler, runtime, source_file="<recurrence_circular_runtime_rank4_vectorized>")
    finally:
        monkeypatch.setattr(NumPyBackend, "_execute_lowered_recurrence_full", original)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "__init__", original_init)
        monkeypatch.setattr(recurrence_setup._CircularRecurrenceBuffer, "materialize", original_materialize)

    assert result.success, (result.errors if result.errors else result.error)
    assert result.outputs["y"] == 16.0
    assert calls["full"] == 0
    assert ring_stats["created"] == 1
    assert ring_stats["materialized"] == 0
    assert ring_stats["full_shape"] == (6, 2, 3, 2)
    assert ring_stats["recurrence_dim"] == 0
    assert ring_stats["preserve_steps"] == 2

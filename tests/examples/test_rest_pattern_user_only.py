"""
Test rest-pattern (..idx, ..batch, ..dims) expansion.
User relu (..idx) and stdlib (..dims in cast, ..batch) are expanded by RestPatternBodyTransformer
(including inside CastExpressionIR RHS). Demo should compile and run.
"""
import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


def test_rest_pattern_demo_compiles_and_runs(compiler, runtime):
    """Demo with rest patterns (user ..idx, stdlib ..dims/..batch) compiles and runs after full expansion."""
    project_root = Path(__file__).resolve().parent.parent.parent
    demo_path = project_root / "examples" / "demos" / "complete_neural_network_implementations.ein"
    if not demo_path.exists():
        pytest.skip(f"Demo file not found: {demo_path}")
    source = demo_path.read_text(encoding="utf-8")
    result = compile_and_execute(source, compiler, runtime, source_file=str(demo_path))
    assert result is not None
    assert getattr(result, "success", False), "Expected demo to compile and run; errors: {}".format(
        getattr(result, "errors", []) or []
    )

"""
Pytest configuration and shared fixtures for all Einlang tests.

This conftest.py provides session and module-scoped fixtures to speed up tests
by reusing expensive compiler and runtime instances.
Imports of CompilerDriver and EinlangRuntime are deferred to fixture use (faster collection).
"""

import os
# IR round-trip: off by default for fast local runs. GitHub workflow sets EINLANG_ROUND_TRIP=1 for coverage.
os.environ.setdefault("EINLANG_ROUND_TRIP", "0")

import sys
import pytest
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Session-scoped fixtures (shared across all tests)
# =============================================================================

@pytest.fixture(scope="session")
def session_compiler():
    """
    Session-scoped stateless compiler instance shared across ALL tests.

    Performance optimizations:
    - Compiler is stateless: creates fresh pass instances per compilation
    - Parser is created once with Lark native caching (2-3x faster)
    - Safe to share: no state pollution between tests
    """
    from einlang.compiler.driver import CompilerDriver
    return CompilerDriver()


@pytest.fixture(scope="session")
def session_runtime():
    """Session-scoped runtime instance shared across all tests."""
    from einlang.runtime.runtime import EinlangRuntime
    return EinlangRuntime(backend="numpy")


# =============================================================================
# Module-scoped fixtures (shared within a test module)
# =============================================================================

@pytest.fixture(scope="module")
def module_compiler(session_compiler):
    """Module-scoped compiler - returns session compiler (stateless, safe to share)."""
    return session_compiler


@pytest.fixture(scope="module")
def module_runtime():
    """Module-scoped runtime instance shared within a test module."""
    from einlang.runtime.runtime import EinlangRuntime
    return EinlangRuntime(backend="numpy")


# =============================================================================
# Class-scoped fixtures (shared within a test class)
# =============================================================================

@pytest.fixture(scope="class")
def class_compiler(session_compiler):
    """Class-scoped compiler - returns session compiler (stateless, safe to share)."""
    return session_compiler


@pytest.fixture(scope="class")
def class_runtime():
    """Class-scoped runtime instance shared within a test class."""
    from einlang.runtime.runtime import EinlangRuntime
    return EinlangRuntime(backend="numpy")


# =============================================================================
# Function-scoped fixtures (default - one per test)
# =============================================================================

@pytest.fixture(scope="class")
def compiler(session_compiler):
    """Class-scoped compiler - shared across all tests in a class for performance."""
    return session_compiler


@pytest.fixture
def runtime():
    """
    Function-scoped runtime - fresh backend per test.
    Ensures isolation under xdist (avoids cross-test state from reduction/comprehension).
    """
    from einlang.runtime.runtime import EinlangRuntime
    return EinlangRuntime(backend="numpy")


# =============================================================================
# Helper fixtures
# =============================================================================

@pytest.fixture(scope="session")
def compile_and_execute_factory(session_compiler, session_runtime):
    """
    Factory fixture that provides a compile_and_execute function.
    Uses session-scoped instances for maximum performance.
    """
    def _compile_and_execute(
        source_code: str,
        inputs: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        compiler: Optional[Any] = None,
        runtime: Optional[Any] = None
    ) -> Any:
        from tests.test_utils import ExecutionResult
        comp = compiler if compiler is not None else session_compiler
        rt = runtime if runtime is not None else session_runtime

        source_file_path = source_file if source_file is not None else "<test>"
        root_path = Path(source_file_path).parent if source_file_path != "<test>" else Path.cwd()
        result = comp.compile(source_code, source_file_path, root_path=root_path)

        if not result.success:
            return ExecutionResult(
                value=None,
                outputs={},
                success=False,
                error=None,
                errors=result.tcx.reporter.format_all_errors() if result.tcx and result.tcx.reporter.has_errors() else [],
            )

        exec_result = rt.execute(result, inputs=inputs or {})
        error_str = str(exec_result.error) if exec_result.error else None
        return ExecutionResult(
            value=exec_result.value,
            outputs=exec_result.outputs or {},
            success=exec_result.error is None,
            error=error_str,
            errors=[error_str] if error_str else [],
        )

    return _compile_and_execute


@pytest.fixture
def compile_and_execute(compile_and_execute_factory):
    """
    Convenience fixture that provides compile_and_execute function.
    Uses the factory with session-scoped instances.
    """
    return compile_and_execute_factory


# =============================================================================
# Performance optimization markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers for test organization."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests as fast"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "ir_only: marks tests to run only on IR path"
    )


# =============================================================================
# IR-only testing
# =============================================================================

@pytest.fixture
def execution_path():
    """Returns the execution path (IR-only)."""
    return "ir"


@pytest.fixture
def runtime_for_path():
    """Creates a runtime instance configured for IR execution."""
    from einlang.runtime.runtime import EinlangRuntime
    return EinlangRuntime(backend="numpy")


# =============================================================================
# Test execution hooks
# =============================================================================

@pytest.fixture(autouse=True)
def reset_compiler_session(request):
    """
    Auto-use fixture that resets compiler and runtime state after each test.
    This prevents state leakage while still allowing fixture reuse.
    """
    yield
    # Only touch runtime if this test actually requested a runtime fixture (cheap check first).
    if not hasattr(request, 'fixturenames'):
        return
    for fixture_name in request.fixturenames:
        if fixture_name != 'runtime':
            continue
        try:
            runtime_instance = request.getfixturevalue(fixture_name)
            executor = runtime_instance.executor if (runtime_instance is not None and hasattr(runtime_instance, 'executor')) else None
            if executor is not None:
                sm = executor.scope_manager if hasattr(executor, 'scope_manager') else None
                if sm is not None:
                    sm.reset()
        except Exception:
            pass
        return

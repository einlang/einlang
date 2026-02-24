"""
Pytest configuration and shared fixtures for all Einlang tests.

This conftest.py provides session and module-scoped fixtures to speed up tests
by reusing expensive compiler and runtime instances.
"""

import sys
import pytest
from typing import Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime, ExecutionResult


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
    return CompilerDriver()


@pytest.fixture(scope="session")
def session_runtime():
    """Session-scoped runtime instance shared across all tests."""
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
        compiler: Optional[CompilerDriver] = None,
        runtime: Optional[EinlangRuntime] = None
    ) -> "ExecutionResult":
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
    if hasattr(request, 'fixturenames'):
        for fixture_name in request.fixturenames:
            if 'runtime' in fixture_name:
                try:
                    runtime_instance = request.getfixturevalue(fixture_name)
                    if hasattr(runtime_instance, 'executor') and runtime_instance.executor:
                        if hasattr(runtime_instance.executor, 'scope_manager'):
                            runtime_instance.executor.scope_manager.reset()
                except Exception:
                    pass

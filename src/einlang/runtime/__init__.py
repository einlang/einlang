"""
Runtime

Rust Pattern: Minimal runtime, LLVM backend interface
Reference: RUNTIME_DESIGN.md
"""

from .runtime import (
    get_entry_file,
    get_script_dir,
    set_entry_file,
    EinlangRuntime,
    ExecutionResult,
    CompilationResult,
)

__all__ = [
    "get_entry_file",
    "get_script_dir",
    "set_entry_file",
    "EinlangRuntime",
    "ExecutionResult",
    "CompilationResult",
]

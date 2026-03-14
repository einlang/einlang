"""
Runtime

Rust Pattern: Minimal runtime, backend delegation
Reference: RUNTIME_DESIGN.md
"""

import os
from typing import Optional, Dict, Any, TYPE_CHECKING

from ..ir.nodes import ProgramIR, ExpressionIR
from ..shared.defid import DefId, assert_defid

if TYPE_CHECKING:
    from ..backends.base import Backend

# Python: interpreter sets __file__ in module globals (path to the script).
entry_file: Optional[str] = None


def set_entry_file(path: Optional[str]) -> None:
    """Set entry file path for this execution. Called by backend at execute() start."""
    global entry_file
    entry_file = path


def get_entry_file() -> Optional[str]:
    """Return entry file path (Python __file__), or None if stdin/inline."""
    return entry_file


def get_script_dir() -> str:
    """Script directory: Python style dirname(abspath(__file__)), or getcwd() if no file."""
    if entry_file:
        return os.path.dirname(os.path.abspath(entry_file))
    return os.getcwd()


def _resolve_input_defid(program: ProgramIR, name: str) -> Optional[DefId]:
    for f in program.functions:
        if f.name == name and f.defid:
            assert_defid(f.defid, allow_none=False)
            return f.defid
    for c in program.constants:
        if c.name == name and c.defid:
            assert_defid(c.defid, allow_none=False)
            return c.defid
    return None


def _get_name_from_defid(program: ProgramIR, defid: DefId) -> Optional[str]:
    assert_defid(defid, allow_none=False)
    for b in program.bindings:
        if b.defid == defid:
            return b.name
    return None


class ExecutionResult:
    """
    Execution result (Rust naming: similar to Rust's execution results).
    
    Rust Pattern: Execution results with optional errors
    """
    def __init__(
        self,
        value: Optional[Any] = None,  # Can be FunctionValue, data value, etc.
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None  # With source location from IR
    ):
        self.value = value
        self.outputs = outputs if outputs is not None else {}
        self.error = error
    
    @property
    def success(self) -> bool:
        """Whether execution succeeded (no error)"""
        return self.error is None
    
    @property
    def errors(self) -> list:
        """List of errors (legacy API)"""
        if self.error:
            return [str(self.error)]
        return []
    
    def get_errors(self) -> list:
        """Get errors (legacy API)"""
        return self.errors


class CompilationResult:
    """Compilation result"""
    def __init__(
        self,
        ir_program: Optional[ProgramIR] = None,
        success: bool = False
    ):
        self.ir_program = ir_program
        self.success = success


class EinlangRuntime:
    """
    Thin runtime layer (Rust naming: minimal runtime).
    
    Rust Pattern: Rust runtime is minimal (std library, not compiler runtime)
    LLVM Pattern: Runtime delegates to backends
    
    Implementation Alignment: Follows Rust's minimal runtime pattern:
    - < 200 lines of code
    - Only backend selection and delegation
    - No execution logic (all in backends)
    - Error propagation with source locations
    
    Reference: Rust runtime is minimal, LLVM runtime delegates to backends
    """
    
    def __init__(self, backend: str = "numpy"):
        """
        Initialize runtime with backend selection.
        
        Rust Pattern: Runtime selects execution strategy
        """
        from ..backends.numpy import NumPyBackend
        
        if backend == "numpy":
            self.backend: Backend = NumPyBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def execute(
        self, 
        compilation_result: CompilationResult, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute using selected backend.
        
        Rust Pattern: Runtime delegates to backend (LLVM pattern)
        Fresh backend per execute: ensures no state leaks between runs (critical under xdist).
        """
        if not compilation_result.success:
            return ExecutionResult(error=RuntimeError("Compilation failed"))
        
        ir = compilation_result.ir
        if ir is None:
            return ExecutionResult(error=RuntimeError("No IR in compilation result"))

        tcx = compilation_result.tcx
        resolver = tcx.resolver if tcx else None
        input_by_defid = {}
        if inputs:
            for name, value in inputs.items():
                defid = _resolve_input_defid(ir, name)
                if defid is not None:
                    input_by_defid[defid] = value
        main_defid = _resolve_input_defid(ir, "main")
        entry_source_file = compilation_result.entry_source_file
        backend = type(self.backend)()
        result = backend.execute(
            ir,
            inputs=None,
            resolver=resolver,
            tcx=tcx,
            input_by_defid=input_by_defid,
            main_defid=main_defid,
            entry_source_file=entry_source_file,
        )
        outputs_named = {}
        for defid, value in (result.outputs or {}).items():
            assert_defid(defid, allow_none=False)
            name = _get_name_from_defid(ir, defid)
            outputs_named[name if name is not None else str(defid)] = value
        return ExecutionResult(value=result.value, outputs=outputs_named, error=result.error)
    
    def execute_expression(
        self, 
        expr: ExpressionIR, 
        env: Dict[DefId, Any]
    ) -> Any:
        """
        Execute single expression using selected backend.
        
        Rust Pattern: Backend handles expression execution
        """
        return self.backend.execute_expression(expr, env)


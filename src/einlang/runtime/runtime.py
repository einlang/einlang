"""
Runtime

Rust Pattern: Minimal runtime, backend delegation
Reference: RUNTIME_DESIGN.md
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from ..ir.nodes import ProgramIR, ExpressionIR
from ..shared.defid import DefId, assert_defid
from .environment import ExecutionEnvironment, FunctionValue

if TYPE_CHECKING:
    from ..backends.base import Backend


def _resolve_input_defid(program: ProgramIR, name: str) -> Optional[DefId]:
    for f in getattr(program, "functions", []) or []:
        if getattr(f, "name", None) == name and getattr(f, "defid", None):
            out = f.defid
            assert_defid(out, allow_none=False)
            return out
    for c in getattr(program, "constants", []) or []:
        if getattr(c, "name", None) == name and getattr(c, "defid", None):
            out = c.defid
            assert_defid(out, allow_none=False)
            return out
    return None


def _get_name_from_defid(program: ProgramIR, defid: DefId) -> Optional[str]:
    assert_defid(defid, allow_none=False)
    for b in getattr(program, "bindings", []) or []:
        if getattr(b, "defid", None) == defid and getattr(b, "name", None):
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
        
        # Get IR from compilation result (may be 'ir' or 'ir_program')
        ir = getattr(compilation_result, 'ir', None) or getattr(compilation_result, 'ir_program', None)
        if not ir:
            return ExecutionResult(error=RuntimeError("No IR in compilation result"))
        
        # Get resolver from compilation result (for builtin DefId lookup)
        resolver = getattr(compilation_result, 'tcx', None)
        if resolver:
            resolver = resolver.resolver
        
        tcx = getattr(compilation_result, 'tcx', None)
        input_by_defid = {}
        if inputs:
            for name, value in inputs.items():
                defid = _resolve_input_defid(ir, name)
                if defid is not None:
                    input_by_defid[defid] = value
        main_defid = _resolve_input_defid(ir, "main")
        backend = type(self.backend)()
        result = backend.execute(
            ir,
            inputs=None,
            resolver=resolver,
            tcx=tcx,
            input_by_defid=input_by_defid,
            main_defid=main_defid,
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


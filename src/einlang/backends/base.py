"""
Backend Interface

Rust Pattern: LLVM TargetMachine
Reference: BACKEND_ARCHITECTURE_DESIGN.md
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..ir.nodes import ProgramIR, ExpressionIR
from ..shared.defid import DefId
from ..runtime.runtime import ExecutionResult


class Backend(ABC):
    """
    Backend interface (Rust naming: rustc_codegen_llvm::Backend).
    
    Rust Pattern: rustc_codegen_llvm::Backend trait
    LLVM Pattern: TargetMachine interface
    
    Implementation Alignment: Follows LLVM's backend interface:
    - All backends implement same interface
    - Backend owns all execution responsibility
    - Backend trusts compiler IR (no re-analysis)
    - Backend handles target-specific codegen
    
    Reference: LLVM TargetMachine, Rust `rustc_codegen_llvm::Backend`
    """
    
    @abstractmethod
    def execute(
        self, 
        program: ProgramIR, 
        inputs: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute IR program.
        
        Rust Pattern: Backend executes compiled code
        
        Implementation Alignment: Follows Rust's backend execution:
        - Trusts compiler IR (uses metadata from IR)
        - No re-analysis (uses type_info, shape_info from IR)
        - Handles all execution logic
        - Returns execution result with source locations in errors
        
        Reference: Rust backend execution pattern
        """
        raise NotImplementedError
    
    @abstractmethod
    def execute_expression(
        self, 
        expr: ExpressionIR, 
        env: Dict[DefId, Any]
    ) -> Any:
        """
        Execute single expression.
        
        Rust Pattern: Backend handles expression evaluation
        
        Implementation Alignment: Follows Rust's expression execution:
        - Uses DefIds for value lookup only (no name-based lookup after name resolution)
        - Trusts IR metadata (type_info, shape_info)
        - Preserves source locations in errors
        
        Reference: Rust expression evaluation pattern
        """
        raise NotImplementedError
    
    @abstractmethod
    def codegen(self, program: ProgramIR) -> Any:
        """
        Generate target code from IR.
        
        Rust Pattern: Backend generates target code
        
        Implementation Alignment: Follows Rust's codegen:
        - Converts IR to target format
        - Uses IR metadata for optimization
        - Returns target code (bytecode, native code, etc.)
        
        Reference: Rust codegen pattern
        """
        raise NotImplementedError


"""
Base Classes and Utilities for Einlang
Core classes, dataclasses, result types, and utility functions
"""

# Python 3.7 compatibility
try:
    pass
except ImportError:
    pass

from typing import Dict, List, Optional, Union, Callable, Any, Generic, TypeVar
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import types from shared module
from ..shared.types import T
from ..shared import ExpressionValue

# Import from shared to avoid circular imports

# ==================== EXECUTION FLOW CONTROL ====================

class ExecutionFlowTag(Enum):
    """Execution flow control tags"""
    CONTINUE = "continue"
    RETURN = "return"
    DIRECT_RESULT = "direct_result"

@dataclass
class ExecutionFlow(Generic[T]):
    """Represents execution flow control without exceptions"""
    tag: ExecutionFlowTag
    value: Optional[T] = None
    
    @classmethod
    def continue_execution(cls) -> 'ExecutionFlow[T]':
        """Continue normal execution"""
        return cls(ExecutionFlowTag.CONTINUE)
    
    @classmethod
    def return_value(cls, value: T) -> 'ExecutionFlow[T]':
        """Return from function with value"""
        return cls(ExecutionFlowTag.RETURN, value)
    
    @classmethod
    def direct_result(cls, value: T) -> 'ExecutionFlow[T]':
        """Direct result from function call"""
        return cls(ExecutionFlowTag.DIRECT_RESULT, value)
    
    def is_continue(self) -> bool:
        return self.tag == ExecutionFlowTag.CONTINUE
    
    def is_return(self) -> bool:
        return self.tag == ExecutionFlowTag.RETURN
    
    def is_direct_result(self) -> bool:
        return self.tag == ExecutionFlowTag.DIRECT_RESULT
    
    def get_value(self) -> Optional[T]:
        return self.value

# Type alias for function execution results
FunctionExecutionResult = ExecutionFlow[ExpressionValue]

# Function execution context types
FunctionExecutionContext = Dict[str, Any]  # Simplified type for function context

# ==================== COMPONENT BASE CLASS ====================

class ComponentBase:
    """Base class for executor components.
    
    Conforms to ComponentBase protocol defined in shared.interfaces.
    All executor components (evaluators, handlers, analyzers) should inherit from this.
    """
    def __init__(self, executor):
        self.executor = executor

# ==================== RESULT TYPES ====================

# Type variables for Result
U = TypeVar('U')
E = TypeVar('E')

class ResultTag(Enum):
    """Result discriminant"""
    OK = "ok"
    ERR = "err"

@dataclass  
class Result(Generic[T, E]):
    """Result type: Ok(T) | Err(E)"""
    tag: ResultTag
    value: Union[T, E]
    
    @classmethod
    def ok(cls, value: T) -> 'Result[T, E]':
        """Create successful result"""
        return cls(ResultTag.OK, value)
    
    @classmethod
    def err(cls, error: E) -> 'Result[T, E]':
        """Create error result"""
        return cls(ResultTag.ERR, error)
    
    def is_ok(self) -> bool:
        """Check if result is Ok"""
        return self.tag == ResultTag.OK
    
    def is_err(self) -> bool:
        """Check if result is Err"""
        return self.tag == ResultTag.ERR
    
    def unwrap(self) -> T:
        """Extract Ok value (throws if Err)"""
        if self.is_err():
            raise ValueError(f"Called unwrap() on Err: {self.value}")
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Extract Ok value or return default"""
        return self.value if self.is_ok() else default
    
    def map(self, func: Callable[[T], U]) -> 'Result[U, E]':
        """Transform Ok value, leave Err unchanged"""
        if self.is_ok():
            return Result.ok(func(self.value))
        return Result.err(self.value)
    
    def map_err(self, func: Callable[[E], U]) -> 'Result[T, U]':
        """Transform Err value, leave Ok unchanged"""
        if self.is_err():
            return Result.err(func(self.value))
        return Result.ok(self.value)
    
    def and_then(self, func: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Monadic bind - chain operations"""
        if self.is_ok():
            return func(self.value)
        return Result.err(self.value)
    
    def __str__(self) -> str:
        """String representation"""
        if self.is_ok():
            return f"Ok({self.value})"
        return f"Err({self.value})"
    
    def __repr__(self) -> str:
        return self.__str__()

# ==================== ERROR TYPES ====================

class IoError(Exception):
    """I/O operation error"""
    def __init__(self, message: str):
        super().__init__(message)

class ValidationError(Exception):
    """Validation error"""  
    def __init__(self, message: str):
        super().__init__(message)

# Type aliases for common patterns
# Note: TryResult, IoResult, ParseResult reference error types that would create circular imports
# These are defined as Any for now and can be properly typed when imported
TryResult = Result[Any, Any]  # Result[Any, EinlangError]
IoResult = Result[Any, IoError]  
ParseResult = Result[Any, Any]  # Result[Any, EinlangSourceError]

# ==================== UTILITY FUNCTIONS ====================

def handle_token(token: Any) -> Dict[str, Any]:
    """Simple token handler - no complex dispatch needed"""
    # Handle Lark Token objects
    try:
        if hasattr(token, 'type') and hasattr(token, 'value'):
            return {
                'type': str(token.type),
                'value': str(token.value),
                'is_terminal': True
            }
    except AttributeError:
        pass
    
    # Handle basic types
    if isinstance(token, str):
        return {'type': 'string', 'value': token, 'is_terminal': True}
    elif isinstance(token, (int, float, np.integer, np.floating)):
        return {'type': 'numeric', 'value': token, 'is_terminal': True}
    else:
        return {'type': 'unknown', 'value': str(token), 'is_terminal': True}

def extract_location_info(meta: Any) -> Dict[str, Any]:
    """Simple location extraction - no complex handler needed"""
    result = {
        'has_location': False,
        'line': 0,
        'column': 0,
        'start_pos': 0,
        'end_pos': 0
    }
    
    if meta is None:
        return result
    
    try:
        if hasattr(meta, 'line') and hasattr(meta, 'column'):
            result.update({
                'has_location': True,
                'line': meta.line or 0,
                'column': meta.column or 0,
                'start_pos': getattr(meta, 'start_pos', 0),
                'end_pos': getattr(meta, 'end_pos', 0)
            })
    except AttributeError:
        pass
    
    return result

def is_iterable_not_string(obj: Any) -> bool:
    """Check if object is iterable but not a string"""
    return (hasattr(obj, '__iter__') and 
            not isinstance(obj, str) and 
            not isinstance(obj, bytes))

def is_callable_method(obj: Any, method_name: str) -> bool:
    """Check if object has a callable method"""
    try:
        method = getattr(obj, method_name, None)
        return callable(method)
    except (AttributeError, TypeError):
        return False

def safe_getattr(obj: Any, attr: str) -> Any:
    """Get attribute - fail fast if it doesn't exist"""
    return getattr(obj, attr)

# ==================== DECLARATION GROUP ====================

@dataclass
class DeclarationGroup:
    """Groups related Einstein tensor declarations that operate on the same array/tensor"""
    declarations: List
    array_name: str
    max_dimensions: List[int] = None
    has_simple_assignments: bool = False
    has_einstein_assignments: bool = False
    is_complete: bool = False  # Set by coverage analysis when group has complete coverage


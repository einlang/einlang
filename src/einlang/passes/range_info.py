"""
Range Information for Einstein Declaration Analysis 



Represents different kinds of ranges for loop variables:
- StaticRange: Compile-time constant (e.g., 0..10)
- DynamicRange: Runtime expression (e.g., 0..array.shape[0])
- DependentRange: Depends on outer loop vars (e.g., j in 0..i)
"""

from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..ir.nodes import ExpressionIR, RangeIR
    
from ..shared.types import PrimitiveType


@dataclass(frozen=True)
class StaticRange:
    """
    Compile-time constant range.
    
    Example: i in 0..10
    
    Compatible with Python range interface (has .start, .stop, .step)
    """
    start: int
    end: int  # Exclusive
    
    @property
    def stop(self) -> int:
        """Alias for .end to match Python range interface"""
        return self.end
    
    @property
    def step(self) -> int:
        """Step size (always 1 for StaticRange)"""
        return 1
    
    def __iter__(self):
        """Make StaticRange iterable like Python range"""
        return iter(range(self.start, self.end))
    
    def __len__(self) -> int:
        """Length of the range"""
        return max(0, self.end - self.start)
    
    def __contains__(self, value: int) -> bool:
        """Check if value is in range"""
        return self.start <= value < self.end
    
    def to_range_ir(self, location) -> 'RangeIR':
        """Convert to IR range node"""
        from ..ir.nodes import LiteralIR, RangeIR
        from ..shared.types import UNKNOWN
        start_lit = LiteralIR(value=self.start, location=location, type_info=PrimitiveType(name='i32'))
        end_lit = LiteralIR(value=self.end, location=location, type_info=PrimitiveType(name='i32'))
        return RangeIR(start=start_lit, end=end_lit, location=location, type_info=UNKNOWN)
    
    def to_python_range(self) -> range:
        """Convert to Python range for static execution"""
        return range(self.start, self.end)
    
    def __str__(self) -> str:
        return f"{self.start}..{self.end}"
    
    def __repr__(self) -> str:
        return f"StaticRange({self.start}, {self.end})"


@dataclass(frozen=True)
class DynamicRange:
    """
    Runtime-evaluated range with IR expression bounds.
    
    Example: i in 0..(shape[0] - 1) / 2
    
    The start/end are IR expression nodes that get evaluated at runtime.
    """
    start: Any  # ExpressionIR
    end: Any  # ExpressionIR
    
    def to_range_ir(self, location) -> 'RangeIR':
        """Convert to IR range node (expressions are already IR)"""
        from ..ir.nodes import RangeIR
        from ..shared.types import UNKNOWN
        return RangeIR(start=self.start, end=self.end, location=location, type_info=UNKNOWN)
    
    def __str__(self) -> str:
        return f"{self.start}..{self.end}"


@dataclass(frozen=True)
class DependentRange:
    """
    Range that depends on outer loop variables.
    
    Example: j in 0..i (triangular iteration space)
    
    Used for non-rectangular iteration spaces common in polyhedral compilation.
    """
    start: Any  # ExpressionIR
    end: Any  # ExpressionIR
    depends_on: set  # Set of variable names this range depends on
    
    def to_range_ir(self, location) -> 'RangeIR':
        """Convert to IR range node"""
        from ..ir.nodes import RangeIR
        from ..shared.types import UNKNOWN
        return RangeIR(start=self.start, end=self.end, location=location, type_info=UNKNOWN)
    
    def __str__(self) -> str:
        deps = ", ".join(sorted(self.depends_on))
        return f"{self.start}..{self.end} (depends on: {deps})"


# Type alias for any range type
RangeInfo = Union[StaticRange, DynamicRange, DependentRange]

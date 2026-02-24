"""
Source Location (Span)

Rust Pattern: rustc_span::Span
Reference: ERROR_REPORTING_DESIGN.md
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    """
    Source location (Rust Span pattern).
    
    Rust Pattern: rustc_span::Span
    
    Implementation Alignment: Follows Rust's `rustc_span::Span` structure:
    - File, line, column (+ optional start/end byte positions for compatibility)
    - Code snippets extracted from source files when needed (not stored here)
    - Immutable (frozen) for hashability
    
    Reference: `rustc_span::Span` contains byte positions, we use line/column for simplicity
    """
    file: str
    line: int
    column: int
    start: int = 0  # compatibility
    end: int = 0     # compatibility
    end_line: int = 0  # extension
    end_column: int = 0  # extension
    
    def __str__(self) -> str:
        """Format as file:line:column (Rust pattern)"""
        return f"{self.file}:{self.line}:{self.column}"


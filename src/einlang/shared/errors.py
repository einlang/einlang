"""
Error Reporting

Rust Pattern: rustc_errors::Diagnostic
Reference: ERROR_REPORTING_DESIGN.md
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from .source_location import SourceLocation


@dataclass
class Error:
    """
    Compiler or runtime error.
    
    Rust Pattern: rustc_errors::Diagnostic
    """
    message: str
    location: SourceLocation
    code: Optional[str] = None  # Error code (e.g., "E0308")
    help: Optional[str] = None  # Helpful suggestion
    note: Optional[str] = None  # Additional note


class ErrorReporter:
    """
    Error reporter with Rust-style formatting.
    
    Rust Pattern: rustc_errors::Emitter
    
    Implementation Alignment: Follows Rust's `rustc_errors::Emitter` implementation:
    - Collect errors with source locations
    - Format errors with code snippets and highlighting
    - Support error codes, help messages, and notes
    - Rust-style formatting (arrow, code snippet, suggestions)
    
    Reference: `rustc_errors::Emitter::emit_diagnostic()` for formatting
    """
    
    def __init__(self, source_files: Dict[str, str]):
        self.source_files = source_files
        self.errors: List[Error] = []
    
    def report_error(
        self,
        message: str,
        location: SourceLocation,
        code: Optional[str] = None,
        help: Optional[str] = None,
        note: Optional[str] = None
    ) -> None:
        """Report an error"""
        error = Error(
            message=message,
            location=location,
            code=code,
            help=help,
            note=note
        )
        self.errors.append(error)
    
    def format_error(self, error: Error) -> str:
        """
        Format error in Rust style.
        
        Rust Pattern: rustc_errors::Emitter::emit_diagnostic()
        """
        lines = []
        
        # Error header
        code_str = f"[{error.code}]" if error.code else ""
        lines.append(f"error{code_str}: {error.message}")
        lines.append("")
        
        # Source location (handle case where location is None)
        if error.location is None:
            lines.append("  --> <unknown location>")
            return "\n".join(lines)
        
        lines.append(f"  --> {error.location.file}:{error.location.line}:{error.location.column}")
        lines.append("   |")
        
        # Code snippet with highlighting (extract from source file)
        code_lines = []
        start_line = 0
        if error.location.file in self.source_files:
            source = self.source_files[error.location.file]
            source_lines = source.split('\n')
            # Extract relevant lines around the error
            start_line = max(0, error.location.line - 1)
            end_line = min(len(source_lines), error.location.end_line)
            code_lines = source_lines[start_line:end_line]
        
        if code_lines:
            for i, code_line in enumerate(code_lines):
                line_num = start_line + i + 1
                lines.append(f"{line_num:3} | {code_line}")
                
                # Highlight error location
                if line_num == error.location.line:
                    highlight = " " * (error.location.column - 1) + "^" * max(1, error.location.end_column - error.location.column)
                lines.append(f"   | {highlight}")
        
        # Help message
        if error.help:
            lines.append("")
            lines.append(f"   = help: {error.help}")
        
        # Note
        if error.note:
            lines.append("")
            lines.append(f"   = note: {error.note}")
        
        return "\n".join(lines)
    
    def format_all_errors(self) -> str:
        """Format all errors"""
        return "\n\n".join(self.format_error(e) for e in self.errors)
    
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return len(self.errors) > 0
    
    def print_errors(self) -> None:
        """Print all errors to stdout"""
        for error in self.errors:
            print(self.format_error(error))


# ============================================================================
# Exception Classes
# ============================================================================

class EinlangError(Exception):
    """Base exception for all Einlang errors"""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        super().__init__(message)
        self.message = message
        self.location = location
    
    def __str__(self):
        if self.location:
            return f"error[E0001]: {self.message}\n  --> {self.location.file}:{self.location.line}:{self.location.column}"
        return self.message


class EinlangSourceError(EinlangError):
    """
    Error in Einlang source code (.ein files) with rich Rust-style formatting.
    
    Use this for ANY error that occurs when processing Einlang source code:
    - Parsing errors in .ein files
    - Runtime errors during .ein execution  
    - Type checking errors in Einlang code
    - Mathematical/tensor operation errors in user code
    
    Always provides rich formatting with source highlighting and helpful diagnostics.
    """
    def __init__(self, 
                 message: str, 
                 location: Optional[SourceLocation] = None,
                 error_code: str = "E0001",
                 category: str = "runtime",
                 source_code: Optional[str] = None):
        super().__init__(message, location)
        self.error_code = error_code
        self.category = category
        self.source_code = source_code
    
    def __str__(self):
        """Simple formatting for source errors"""
        lines = []
        lines.append(f"error[{self.error_code}]: {self.message}")
        
        if self.location:
            lines.append(f"  --> {self.location.file}:{self.location.line}:{self.location.column}")
        
        return "\n".join(lines)


class EinlangImplementationError(Exception):
    """
    Error in Python implementation code (not user's Einlang code).
    
    Use this for internal Python errors:
    - Missing implementations
    - Invalid internal state
    - Python-level bugs
    
    Never use this for errors in user's Einlang code - use EinlangSourceError instead.
    """
    def __init__(self, message: str, error_code: str = "E9999"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


# Make EinlangSourceError globally available to fix import issues during dynamic execution
import builtins
builtins.EinlangSourceError = EinlangSourceError


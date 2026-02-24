"""
Error Reporting

Rust Pattern: rustc_errors::Diagnostic
Reference: ERROR_REPORTING_DESIGN.md
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .source_location import SourceLocation


# ---------------------------------------------------------------------------
# ANSI color helpers (disabled when NO_COLOR is set or not a TTY)
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    explicit = os.environ.get("EINLANG_COLOR", "").lower()
    if explicit in ("0", "false", "no", "never"):
        return False
    return True

_BOLD   = "\033[1m"
_RED    = "\033[31m"
_BLUE   = "\033[34m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"

def _style(text: str, *codes: str, color: bool = True) -> str:
    if not color:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{_RESET}" if prefix else text


# ---------------------------------------------------------------------------
# Secondary label (for multi-span diagnostics)
# ---------------------------------------------------------------------------

@dataclass
class Label:
    location: SourceLocation
    message: str = ""
    primary: bool = False


# ---------------------------------------------------------------------------
# Error dataclass
# ---------------------------------------------------------------------------

@dataclass
class Error:
    """
    Compiler or runtime error.

    Rust Pattern: rustc_errors::Diagnostic
    """
    message: str
    location: Optional[SourceLocation]
    code: Optional[str] = None
    help: Optional[str] = None
    note: Optional[str] = None
    label: Optional[str] = None
    labels: List[Label] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Formatting engine
# ---------------------------------------------------------------------------

def _format_diagnostic(
    error: Error,
    source_files: Dict[str, str],
    color: bool = False,
) -> str:
    """
    Render a single diagnostic in rustc style.

    Example output (plain, no color)::

        error[E0308]: mismatched types
         --> main.ein:5:17
          |
        5 |     let x: i32 = "hello";
          |                  ^^^^^^^ expected `i32`, found `str`
          |
          = help: try an explicit cast with `as`
    """
    out: List[str] = []

    # ---- header -----------------------------------------------------------
    code_str = f"[{error.code}]" if error.code else ""
    out.append(
        _style(f"error{code_str}", _BOLD, _RED, color=color)
        + _style(f": {error.message}", _BOLD, color=color)
    )

    # ---- location arrow ---------------------------------------------------
    if error.location is None:
        out.append(_style(" --> ", _BOLD, _BLUE, color=color) + "<unknown location>")
        _append_annotations(out, error, 1, color)
        return "\n".join(out)

    loc = error.location

    # ---- source snippet ---------------------------------------------------
    source = source_files.get(loc.file)
    if source is None:
        out.append(
            _style(" --> ", _BOLD, _BLUE, color=color)
            + f"{loc.file}:{loc.line}:{loc.column}"
        )
        _append_annotations(out, error, 1, color)
        return "\n".join(out)

    src_lines = source.split("\n")

    err_line = loc.line
    err_end_line = loc.end_line if loc.end_line and loc.end_line >= err_line else err_line
    err_col = max(loc.column, 1)
    err_end_col = loc.end_column if loc.end_column else 0
    multiline = err_end_line > err_line

    display_start = err_line
    display_end = min(len(src_lines), err_end_line)

    gw = max(len(str(display_end)), 1)

    def arrow_line() -> str:
        prefix = " " * gw + "--> "
        return _style(prefix, _BOLD, _BLUE, color=color) + f"{loc.file}:{loc.line}:{loc.column}"

    def empty_gutter() -> str:
        return _style(" " * (gw + 1) + "|", _BOLD, _BLUE, color=color)

    def code_gutter(num: int) -> str:
        return _style(str(num).rjust(gw) + " | ", _BOLD, _BLUE, color=color)

    def underline_gutter() -> str:
        return _style(" " * (gw + 1) + "| ", _BOLD, _BLUE, color=color)

    out.append(arrow_line())
    out.append(empty_gutter())

    inline_label = error.label or ""

    if not multiline:
        for line_num in range(display_start, display_end + 1):
            idx = line_num - 1
            code_line = src_lines[idx] if 0 <= idx < len(src_lines) else ""
            out.append(f"{code_gutter(line_num)}{code_line}")

            if line_num == err_line:
                col_start = err_col - 1
                if err_end_col > err_col:
                    span_len = err_end_col - err_col
                else:
                    span_len = _guess_span(code_line, col_start)
                span_len = max(1, span_len)
                carets = " " * col_start + "^" * span_len
                label_suffix = f" {inline_label}" if inline_label else ""
                out.append(f"{underline_gutter()}{_style(carets + label_suffix, _BOLD, _RED, color=color)}")
    else:
        for line_num in range(display_start, display_end + 1):
            idx = line_num - 1
            code_line = src_lines[idx] if 0 <= idx < len(src_lines) else ""

            if line_num == err_line:
                out.append(f"{code_gutter(line_num)}{code_line}")
                col_0 = err_col - 1
                if col_0 > 0:
                    opening = " " + "_" * (col_0 - 1) + "^"
                else:
                    opening = "^"
                out.append(f"{underline_gutter()}{_style(opening, _BOLD, _RED, color=color)}")
            elif err_line < line_num < err_end_line:
                out.append(f"{code_gutter(line_num)}{_style('| ', _BOLD, _RED, color=color)}{code_line}")
            elif line_num == err_end_line:
                out.append(f"{code_gutter(line_num)}{_style('| ', _BOLD, _RED, color=color)}{code_line}")
                end_col_0 = (err_end_col - 1) if err_end_col > 0 else len(code_line.rstrip())
                label_suffix = f" {inline_label}" if inline_label else ""
                closing = "|" + "_" * max(1, end_col_0) + "^" + label_suffix
                out.append(f"{underline_gutter()}{_style(closing, _BOLD, _RED, color=color)}")
            else:
                out.append(f"{code_gutter(line_num)}{code_line}")

    _append_annotations(out, error, gw, color)

    return "\n".join(out)


def _guess_span(code_line: str, col_start: int) -> int:
    """Guess token length when end_column is unavailable."""
    if col_start >= len(code_line):
        return 1
    rest = code_line[col_start:]
    length = 0
    for ch in rest:
        if ch in (" ", "\t", ";", ",", ")", "]", "}"):
            break
        length += 1
    return max(1, length)


def _append_annotations(
    out: List[str],
    error: Error,
    gw: int,
    color: bool,
) -> None:
    has_ann = error.help or error.note
    if not has_ann:
        return
    out.append(_style(" " * (gw + 1) + "|", _BOLD, _BLUE, color=color))
    pad = " " * (gw + 1)
    if error.help:
        out.append(
            _style(f"{pad}= ", _BOLD, _CYAN, color=color)
            + _style("help: ", _BOLD, color=color)
            + error.help
        )
    if error.note:
        out.append(
            _style(f"{pad}= ", _BOLD, _CYAN, color=color)
            + _style("note: ", _BOLD, color=color)
            + error.note
        )


# ---------------------------------------------------------------------------
# ErrorReporter
# ---------------------------------------------------------------------------

class ErrorReporter:
    """
    Error reporter with Rust-style formatting.

    Rust Pattern: rustc_errors::Emitter
    """

    def __init__(self, source_files: Dict[str, str]):
        self.source_files = source_files
        self.errors: List[Error] = []

    def report_error(
        self,
        message: str,
        location: Optional[SourceLocation],
        code: Optional[str] = None,
        help: Optional[str] = None,
        note: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        error = Error(
            message=message,
            location=location,
            code=code,
            help=help,
            note=note,
            label=label,
        )
        self.errors.append(error)

    def format_error(self, error: Error, color: Optional[bool] = None) -> str:
        use_color = color if color is not None else _use_color()
        return _format_diagnostic(error, self.source_files, color=use_color)

    def format_all_errors(self, color: Optional[bool] = None) -> str:
        parts = [self.format_error(e, color=color) for e in self.errors]
        use_color = color if color is not None else _use_color()
        count = len(self.errors)
        summary = f"aborting due to {count} previous error{'s' if count != 1 else ''}"
        parts.append(
            _style("error", _BOLD, _RED, color=use_color)
            + _style(f": {summary}", _BOLD, color=use_color)
        )
        return "\n\n".join(parts)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def print_errors(self) -> None:
        color = _use_color()
        for error in self.errors:
            print(self.format_error(error, color=color), file=sys.stderr)
        count = len(self.errors)
        if count:
            summary = f"aborting due to {count} previous error{'s' if count != 1 else ''}"
            msg = (
                _style("error", _BOLD, _RED, color=color)
                + _style(f": {summary}", _BOLD, color=color)
            )
            print(f"\n{msg}", file=sys.stderr)


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
            return f"error[E0001]: {self.message}\n --> {self.location.file}:{self.location.line}:{self.location.column}"
        return self.message


class EinlangSourceError(EinlangError):
    """
    Error in Einlang source code (.ein files) with rich Rust-style formatting.

    Use this for ANY error that occurs when processing Einlang source code:
    - Parsing errors in .ein files
    - Runtime errors during .ein execution
    - Type checking errors in Einlang code
    - Mathematical/tensor operation errors in user code
    """
    def __init__(self,
                 message: str,
                 location: Optional[SourceLocation] = None,
                 error_code: str = "E0001",
                 category: str = "runtime",
                 source_code: Optional[str] = None,
                 help: Optional[str] = None,
                 note: Optional[str] = None,
                 label: Optional[str] = None):
        super().__init__(message, location)
        self.error_code = error_code
        self.category = category
        self.source_code = source_code
        self.help_text = help
        self.note_text = note
        self.label_text = label

    def __str__(self):
        source_files: Dict[str, str] = {}
        if self.source_code and self.location:
            source_files[self.location.file] = self.source_code
        err = Error(
            message=self.message,
            location=self.location,
            code=self.error_code,
            help=self.help_text,
            note=self.note_text,
            label=self.label_text,
        )
        return _format_diagnostic(err, source_files, color=_use_color())


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

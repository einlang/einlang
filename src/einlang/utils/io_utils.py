"""
Centralized file I/O utilities.

- Single place for encoding and temp-path handling
- Use Path.read_text() consistently (no raw open/read)
"""

from pathlib import Path
from typing import Union

from .config import DEFAULT_FILE_ENCODING


def read_source_file(path: Union[Path, str]) -> str:
    """Read source file with standard encoding."""
    p = Path(path) if not isinstance(path, Path) else path
    return p.read_text(encoding=DEFAULT_FILE_ENCODING)


def read_source_lines(path: Union[Path, str]) -> list:
    """Read source file as lines (for error context)."""
    return read_source_file(path).splitlines()


def is_temp_path(path: Union[Path, str]) -> bool:
    """True if path is under temp dir (e.g. /tmp, /var/folders)."""
    s = str(path).lower()
    return "/tmp" in s or "/var/folders" in s

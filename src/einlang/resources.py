"""Helpers for locating runtime resources in source and installed layouts."""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


def _search_up_for_stdlib(start_path: Path, max_levels: int = 5) -> Optional[Path]:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent
    for _ in range(max_levels):
        candidate = current / "stdlib"
        if candidate.exists() and candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


@lru_cache(maxsize=1)
def bundled_stdlib_root() -> Optional[Path]:
    candidate = Path(__file__).resolve().parent / "_bundled" / "stdlib"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def default_stdlib_root(start_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Prefer a project-local stdlib and fall back to bundled package assets."""
    if start_path is not None:
        found = _search_up_for_stdlib(Path(start_path))
        if found is not None:
            return found
    return bundled_stdlib_root()

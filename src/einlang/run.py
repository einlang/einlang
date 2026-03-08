"""
Single-call run API: compile and execute in one step.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union

from .compiler.driver import CompilerDriver
from .runtime.runtime import EinlangRuntime, ExecutionResult


def run(
    source: Optional[str] = None,
    file: Optional[Union[str, Path]] = None,
    backend: str = "numpy",
    inputs: Optional[Dict[str, Any]] = None,
    source_file: str = "<inline>",
    root_path: Optional[Union[str, Path]] = None,
) -> ExecutionResult:
    """
    Compile and execute Einlang source in one call.

    Provide either `source` (code string) or `file` (path to .ein file).
    If `file` is given, it is read and used for `source`; `source_file` and
    `root_path` are derived from the path.

    Args:
        source: Einlang source code. Ignored if `file` is set.
        file: Path to a .ein file. If set, the file is read and used as source.
        backend: Execution backend (default "numpy").
        inputs: Optional dict of input name -> value for the program.
        source_file: Name used in error messages (default "<inline>").
        root_path: Root for module resolution (default cwd when using source).

    Returns:
        ExecutionResult with .outputs, .error, .success.

    Example:
        >>> from einlang import run
        >>> out = run(source="let x = 1+1; print(x);")
        >>> out.success
        True
        >>> out = run(file="examples/hello.ein")
    """
    if file is not None:
        path = Path(file).resolve()
        if not path.exists() or not path.is_file():
            return ExecutionResult(error=FileNotFoundError(f"file not found: {path}"))
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            return ExecutionResult(error=e)
        source_file = str(path)
        root_path = path.parent
    else:
        if source is None:
            source = ""
        if root_path is None:
            root_path = Path.cwd()
        else:
            root_path = Path(root_path)

    compiler = CompilerDriver()
    result = compiler.compile(source, source_file, root_path=root_path)

    if not result.success:
        msg = "compilation failed"
        if result.tcx and result.tcx.reporter and result.tcx.reporter.has_errors():
            msg = result.tcx.reporter.format_all_errors()
        return ExecutionResult(error=RuntimeError(msg))

    return EinlangRuntime(backend=backend).execute(result, inputs=inputs)

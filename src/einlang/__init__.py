"""Einlang - a programming language for tensor computations."""

__version__ = "0.1.0"

from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime
from einlang.run import run

__all__ = ["__version__", "CompilerDriver", "EinlangRuntime", "run"]

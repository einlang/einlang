"""
Parser

Rust Pattern: rustc_parse
Reference: COMPILER_FLOW_DESIGN.md
"""

from typing import Any, Dict, Optional
from pathlib import Path
from lark import Lark
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, ParseError as LarkParseError
import logging

# Import AST nodes and transformer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from einlang.shared.nodes import Program as ASTProgram
from einlang.utils.config import DEFAULT_PARSER_CACHE_FILE
from .transformers.base import EinlangTransformer

logger = logging.getLogger("einlang.frontend.parser")

# Process-level Lark singletons keyed by cache path. Disk cache is disabled by
# default to avoid persistent parser files; callers can opt in with an explicit
# cache file path.
_shared_lark: Dict[Optional[str], Lark] = {}


def _get_shared_lark(cache_file: Optional[str]) -> Lark:
    cache_key = str(cache_file) if cache_file else None
    shared = _shared_lark.get(cache_key)
    if shared is None:
        grammar_path = Path(__file__).parent / "grammar.lark"
        shared = Lark.open(
            grammar_path,
            start='program',
            parser='lalr',
            cache=cache_key if cache_key else False,
            propagate_positions=True,
            maybe_placeholders=False,
        )
        _shared_lark[cache_key] = shared
    return shared


class Parser:
    """
    Parser (Rust naming: rustc_parse).
    
    Rust Pattern: rustc_parse::parse()
    
    Implementation Alignment: Follows Rust's parser interface:
    - Takes source code, returns AST
    - Preserves source locations
    - Handles parse errors
    - Uses Lark parser with caching
    
    Reference: `rustc_parse::parse()` for parser interface
    """
    
    def __init__(self, cache_file: Optional[str] = DEFAULT_PARSER_CACHE_FILE):
        """
        Initialize parser with Lark best practices.
        
        Rust Pattern: rustc_parse initialization
        """
        self.parser = _get_shared_lark(cache_file)
    
    def parse(self, source: str, source_file: str = "main.ein") -> ASTProgram:
        """
        Parse source code to AST.
        
        Rust Pattern: rustc_parse::parse()
        
        Returns: AST (Program node)
        """
        try:
            transformer = EinlangTransformer(source_file)
            
            # Parse and transform
            tree = self.parser.parse(source)
            ast = transformer.transform(tree)
            
            return ast
        
        except (UnexpectedToken, UnexpectedCharacters, LarkParseError) as e:
            # Convert Lark parse errors to EinlangSourceError
            error_msg = f"Parse error: {str(e)}"
            location = None
            
            # Try to extract location from Lark error
            if hasattr(e, 'line') and hasattr(e, 'column'):
                # SourceLocation is in einlang.shared.nodes, not einlang.shared.source_location
                from einlang.shared.nodes import SourceLocation
                location = SourceLocation(
                    file=source_file,
                    line=e.line,
                    column=e.column,
                    start=0,
                    end=0
                )
            
            raise ParseError(error_msg, source_file, location) from e
        
        except Exception as e:
            # Wrap other errors
            raise ParseError(f"Parse error: {str(e)}", source_file) from e


class ParseError(Exception):
    """Parse error with source location"""
    def __init__(self, message: str, source_file: str, location: Optional[Any] = None):
        self.message = message
        self.source_file = source_file
        self.location = location
        super().__init__(f"{message} in {source_file}")

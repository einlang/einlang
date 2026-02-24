"""
Configuration constants to replace magic numbers throughout Einlang
"""

# Tensor configuration constants
DEFAULT_TENSOR_CHANNEL_SIZE = 3  # Default tensor size (common for RGB channels)

# Display and formatting constants  
MAX_ERROR_LINE_LENGTH = 80  # Maximum length for error line display
ERROR_CONTEXT_CHARS = 40   # Characters of context around error position
ERROR_DISPLAY_PADDING = 30  # Padding for error display truncation

# Parser configuration constants (cache under temp dir to avoid cluttering project root)
import os
import tempfile
DEFAULT_PARSER_CACHE_FILE = os.path.join(tempfile.gettempdir(), "einlang_parser_improved.cache")

# Module resolution constants
STD_MODULE_PREFIX = "std"
PYTHON_MODULE_PREFIX = "python"
MODULE_SEPARATOR = "::"
MODULE_FILE_EXTENSION = ".ein"

# String literal constants
STRING_QUOTE_CHAR = '"'
BOOLEAN_TRUE_LITERAL = "true"
BOOLEAN_FALSE_LITERAL = "false"

# Numeric parsing constants
DECIMAL_SEPARATOR = "."
SCIENTIFIC_NOTATION_INDICATOR = "e"

# Default numeric types (Rust-like: integer default i32, float default f32)
DEFAULT_INT_TYPE = "i32"
DEFAULT_FLOAT_TYPE = "f32"

# Float formatting constants
DEFAULT_FLOAT_PRECISION = 4  # Default precision for float formatting in g format

# Array indexing constants
ARRAY_INDEX_START = 0  # Arrays are 0-indexed
FIRST_ELEMENT_INDEX = 0
ARRAY_INDEX_OFFSET = 1  # Offset for converting 1-based to 0-based indexing

# File encoding constants
DEFAULT_FILE_ENCODING = "utf-8"

# Error reporting constants
ERROR_CONTEXT_PREFIX = "..."
ERROR_POINTER_CHAR = "^"
ERROR_LINE_PREFIX_LENGTH = 4
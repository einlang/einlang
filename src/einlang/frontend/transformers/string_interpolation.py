"""
String Interpolation Parser - Extracted from EinlangTransformer
Handles parsing of string interpolation with double brace escaping
"""

from typing import List, Union, Optional
from ...shared import Literal, InterpolatedString, InterpolationPart, Identifier, SourceLocation

class StringInterpolationParser:
    """Dedicated parser for string interpolation patterns"""
    
    # Unique placeholders to avoid conflicts with user content
    OPEN_PLACEHOLDER = "\x00OPEN_BRACE\x00"
    CLOSE_PLACEHOLDER = "\x00CLOSE_BRACE\x00"
    
    @staticmethod
    def parse(string_content: str, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse string with interpolation support using double braces for escaping"""
        if '{' not in string_content:
            return Literal(value=string_content, location=location)
        
        parser = StringInterpolationParser()
        return parser._parse_with_interpolation(string_content, location)
    
    def _parse_with_interpolation(self, string_content: str, location: SourceLocation) -> Union[Literal, InterpolatedString]:
        """Parse string content with interpolation patterns"""
        # Replace escape sequences with placeholders
        temp_content = self._replace_escape_sequences(string_content)
        
        # Parse interpolation parts
        parts = self._extract_interpolation_parts(temp_content, location)
        
        # Check if any actual interpolation was found
        # Parts are now all AST nodes (Literal or InterpolationPart)
        if not any(isinstance(part, InterpolationPart) for part in parts):
            # All parts are Literal - flatten into single Literal
            combined_value = ''.join(lit.value for lit in parts)
            return Literal(value=combined_value, location=location)
        
        return InterpolatedString(parts=parts, location=location)
    
    def _replace_escape_sequences(self, content: str) -> str:
        """Replace double braces with unique placeholders"""
        content = content.replace('{{', self.OPEN_PLACEHOLDER)
        content = content.replace('}}', self.CLOSE_PLACEHOLDER)
        return content
    
    def _restore_escape_sequences(self, content: str) -> str:
        """Restore placeholders back to single braces"""
        content = content.replace(self.OPEN_PLACEHOLDER, '{')
        content = content.replace(self.CLOSE_PLACEHOLDER, '}')
        return content
    
    def _extract_interpolation_parts(self, temp_content: str, location: SourceLocation) -> List[Union[Literal, InterpolationPart]]:
        """Extract text and interpolation parts from content"""
        parts = []
        current_pos = 0
        
        while current_pos < len(temp_content):
            brace_pos = temp_content.find('{', current_pos)
            
            if brace_pos == -1:
                # No more braces, add remaining text
                self._add_remaining_text(parts, temp_content, current_pos, location)
                break
            
            # Add text before brace
            self._add_text_before_brace(parts, temp_content, current_pos, brace_pos, location)
            
            # Process interpolation expression
            current_pos = self._process_interpolation(parts, temp_content, brace_pos, location)
            
            if current_pos == -1:  # Unclosed brace
                self._add_remaining_text(parts, temp_content, brace_pos, location)
                break
        
        return parts
    
    def _add_remaining_text(self, parts: List, temp_content: str, start_pos: int, location: SourceLocation = None) -> None:
        """Add remaining text to parts list"""
        if start_pos < len(temp_content):
            remaining = self._restore_escape_sequences(temp_content[start_pos:])
            # Wrap in Literal so all parts have accept() method
            parts.append(Literal(value=remaining, location=location))
    
    def _add_text_before_brace(self, parts: List, temp_content: str, current_pos: int, brace_pos: int, location: SourceLocation = None) -> None:
        """Add text before interpolation brace"""
        if brace_pos > current_pos:
            before_text = self._restore_escape_sequences(temp_content[current_pos:brace_pos])
            # Wrap in Literal so all parts have accept() method
            parts.append(Literal(value=before_text, location=location))
    
    def _process_interpolation(self, parts: List, temp_content: str, brace_pos: int, location: SourceLocation) -> int:
        """Process a single interpolation expression, returns next position or -1 if unclosed"""
        closing_brace = temp_content.find('}', brace_pos + 1)
        
        if closing_brace == -1:
            # Unclosed brace - treat as literal
            remaining = self._restore_escape_sequences(temp_content[brace_pos:])
            # Wrap in Literal so all parts have accept() method
            parts.append(Literal(value=remaining, location=location))
            return -1
        
        # Extract and parse interpolation expression
        expr_content = temp_content[brace_pos + 1:closing_brace]
        interpolation_part = self._create_interpolation_part(expr_content, location)
        
        if interpolation_part:
            parts.append(interpolation_part)
        else:
            # Failed to parse - treat as literal
            literal_part = self._restore_escape_sequences(temp_content[brace_pos:closing_brace + 1])
            # Wrap in Literal so all parts have accept() method
            parts.append(Literal(value=literal_part, location=location))
        
        return closing_brace + 1
    
    def _create_interpolation_part(self, expr_content: str, location: SourceLocation) -> Optional[InterpolationPart]:
        """Create an InterpolationPart from expression content"""
        if not expr_content.strip():
            return None
        
        # Parse format specification if present
        format_spec = None
        if ':' in expr_content:
            expr_part, format_spec = expr_content.split(':', 1)
            expr_content = expr_part.strip()
            format_spec = format_spec.strip()
        
        try:
            # Create expression AST node (simplified for now)
            expr_ast = self._parse_interpolation_expression(expr_content.strip(), location)
            return InterpolationPart(expr=expr_ast, format_spec=format_spec)
        except Exception:
            return None  # Let caller handle as literal
    
    def _parse_interpolation_expression(self, expr_content: str, location: SourceLocation) -> Identifier:
        """Parse interpolation expression - simplified implementation"""
        # Simplified implementation: handle as identifier (covers most common cases)
        # Future: Implement full expression parsing for complex cases
        return Identifier(name=expr_content, location=location)

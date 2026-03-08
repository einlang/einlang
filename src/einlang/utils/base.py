"""
Base utilities for Einlang (token/location helpers used by frontend).
"""

from typing import Dict, Any
import numpy as np

# ==================== UTILITY FUNCTIONS ====================

def handle_token(token: Any) -> Dict[str, Any]:
    """Simple token handler - no complex dispatch needed"""
    # Handle Lark Token objects
    try:
        if hasattr(token, 'type') and hasattr(token, 'value'):
            return {
                'type': str(token.type),
                'value': str(token.value),
                'is_terminal': True
            }
    except AttributeError:
        pass
    
    # Handle basic types
    if isinstance(token, str):
        return {'type': 'string', 'value': token, 'is_terminal': True}
    elif isinstance(token, (int, float, np.integer, np.floating)):
        return {'type': 'numeric', 'value': token, 'is_terminal': True}
    else:
        return {'type': 'unknown', 'value': str(token), 'is_terminal': True}

def extract_location_info(meta: Any) -> Dict[str, Any]:
    """Simple location extraction - no complex handler needed"""
    result = {
        'has_location': False,
        'line': 0,
        'column': 0,
        'start_pos': 0,
        'end_pos': 0,
        'end_line': 0,
        'end_column': 0,
    }
    
    if meta is None:
        return result
    
    try:
        if hasattr(meta, 'line') and hasattr(meta, 'column'):
            result.update({
                'has_location': True,
                'line': meta.line or 0,
                'column': meta.column or 0,
                'start_pos': getattr(meta, 'start_pos', 0),
                'end_pos': getattr(meta, 'end_pos', 0),
                'end_line': getattr(meta, 'end_line', 0) or 0,
                'end_column': getattr(meta, 'end_column', 0) or 0,
            })
    except AttributeError:
        pass
    
    return result


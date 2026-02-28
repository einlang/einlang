"""
Analysis Guard - Determines which functions should be analyzed.

Rule: Only analyze SPECIALIZED functions (all parameters have concrete types).
      Skip GENERIC functions (any parameter without type annotation or with dynamic rank).

GENERIC = ANY parameter has no type annotation OR has dynamic rank [T; *]
SPECIALIZED = ALL parameters have concrete types with known ranks


"""

import logging
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from ..ir.nodes import BindingIR

logger = logging.getLogger(__name__)


def is_generic_function(func_def: 'BindingIR') -> bool:
    """
    Check if function has generic parameters.
    
    A function is generic if ANY parameter:
    - Has no type annotation (param_type is None)
    - Has dynamic rank type ([T; *])
    
    Args:
        func_def: Function definition IR node
        
    Returns:
        True if function is generic (needs monomorphization)
        False if function has all concrete types
    """
    from ..shared.types import RectangularType
    
    for param in func_def.parameters:
        # No type annotation → generic (needs monomorphization)
        if not param.param_type:
            return True
        
        # Dynamic rank [T; *] → generic (needs monomorphization)
        if isinstance(param.param_type, RectangularType):
            if getattr(param.param_type, 'is_dynamic_rank', False):
                return True
    
    # All parameters have concrete types → specialized
    return False


def should_analyze_function(func_def: 'BindingIR', tcx: Optional[Any] = None) -> bool:
    """
    Determine if a function should be analyzed by analysis passes.
    Never run stdlib-specific passes: same rule for all code (stdlib or user).
    
    Args:
        func_def: Function definition IR node
        tcx: Optional TyCtxt (unused; kept for API compatibility)
        
    Returns:
        True: Analyze this function now (has concrete types)
        False: Skip for now (generic function, will be monomorphized later)
    """
    if not is_generic_function(func_def):
        logger.debug(f"SPECIALIZED: '{func_def.name}' → Analyze now")
        return True
    logger.debug(f"GENERIC: '{func_def.name}' → Skip analysis (will be monomorphized)")
    return False

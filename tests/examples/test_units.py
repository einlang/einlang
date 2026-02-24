#!/usr/bin/env python3
"""
Parametrized units tests - loads all file contents together upfront for speed.
"""

import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


# Load all file contents once at module import time
_UNITS_CACHE = {}

def _load_all_units():
    """Load all units file contents into cache once"""
    if _UNITS_CACHE:
        return _UNITS_CACHE
    
    project_root = Path(__file__).parent.parent.parent
    units_dir = project_root / "examples" / "units"
    if units_dir.exists():
        for f in sorted(units_dir.glob("*.ein")):
            with open(f, 'r', encoding='utf-8') as fp:
                _UNITS_CACHE[f.stem] = fp.read()
    return _UNITS_CACHE

# Trigger load at import
_load_all_units()


# Units that require stdlib (e.g. std::math) - skip when not available to avoid spurious failures
# Stdlib units: unskipped when std:: load/cache works (source_file set so root_path finds stdlib)
STDLIB_DEPENDENT_UNITS = frozenset()

def get_units_params():
    """Get parametrized test cases with content already loaded"""
    params = []
    for name in _UNITS_CACHE.keys():
        if name in STDLIB_DEPENDENT_UNITS:
            params.append(pytest.param(name, id=name, marks=pytest.mark.skip(reason="Requires stdlib math module")))
        else:
            params.append(pytest.param(name, id=name))
    return params


def _unit_source_file(unit_name: str) -> str:
    """Path to unit file so compiler can resolve stdlib (root_path = parent of this file)."""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "examples" / "units" / f"{unit_name}.ein")


class TestUnits:
    """Tests for units tutorial files - content pre-loaded for speed"""
    
    @pytest.mark.parametrize("unit_name", get_units_params())
    def test_execution(self, compiler, runtime, unit_name):
        """Test unit tutorial execution"""
        content = _UNITS_CACHE[unit_name]
        source_file = _unit_source_file(unit_name)

        result = compile_and_execute(content, compiler, runtime, source_file=source_file)
        
        assert result is not None, f"No result for {unit_name}"
        assert getattr(result, 'success', False), \
            f"{unit_name} failed: {getattr(result, 'errors', ['Unknown'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

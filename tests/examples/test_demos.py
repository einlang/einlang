#!/usr/bin/env python3
"""
Parametrized demos tests - loads all file contents together upfront for speed.
"""

import pytest
from pathlib import Path
from tests.test_utils import compile_and_execute


# Load all file contents once at module import time
_DEMOS_CACHE = {}
_DEMOS_PATHS = {}

def _load_all_demos():
    """Load all demos file contents into cache once"""
    if _DEMOS_CACHE:
        return
    
    project_root = Path(__file__).parent.parent.parent
    demos_dir = project_root / "examples" / "demos"
    unsupported = ['enum ', 'type ', 'while ', 'tensor[', '-> tensor', 'scan[+](', 'data = [']
    
    if demos_dir.exists():
        for f in sorted(demos_dir.glob("*.ein")):
            with open(f, 'r', encoding='utf-8') as fp:
                content = fp.read()
            # Skip unsupported syntax
            if not any(kw in content for kw in unsupported):
                _DEMOS_CACHE[f.stem] = content
                _DEMOS_PATHS[f.stem] = str(f)

# Trigger load at import
_load_all_demos()


def get_demos_params():
    """Get parametrized test cases with content already loaded"""
    return [pytest.param(name, id=name) for name in _DEMOS_CACHE.keys()]


class TestDemos:
    """Tests for demos tutorial files - content pre-loaded for speed"""
    
    @pytest.mark.parametrize("demo_name", get_demos_params())
    def test_execution(self, compiler, runtime, demo_name):
        """Test demo execution"""
        content = _DEMOS_CACHE[demo_name]
        source_file = _DEMOS_PATHS[demo_name]
        
        # Check for expected failure marker
        expected_fail = "EXPECTED TO FAIL" in content
        
        try:
            result = compile_and_execute(content, compiler, runtime, source_file=source_file)
            
            if result is None or not getattr(result, 'success', False):
                if expected_fail:
                    return  # Expected to fail
                errors = getattr(result, 'errors', ['Unknown']) if result else ['No result']
                pytest.fail(f"{demo_name} failed: {errors}")
        except Exception as e:
            if expected_fail:
                return  # Expected to fail
            pytest.fail(f"{demo_name} exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Parametrized units tests - loads all file contents together upfront for speed.
"""

import pytest
from tests.test_utils import compile_and_execute, load_example_sources


_UNIT_SOURCES = load_example_sources("examples/units")


class TestUnits:
    """Tests for units tutorial files - content pre-loaded for speed"""
    
    @pytest.mark.parametrize("unit_source", _UNIT_SOURCES, ids=lambda source: source.name)
    def test_execution(self, compiler, runtime, unit_source):
        """Test unit tutorial execution"""
        result = compile_and_execute(
            unit_source.content,
            compiler,
            runtime,
            source_file=str(unit_source.path),
        )
        
        assert result is not None, f"No result for {unit_source.name}"
        assert result.success, f"{unit_source.name} failed: {result.errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

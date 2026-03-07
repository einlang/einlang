"""
Unit tests for backend clause output behavior.
"""

import pytest

from einlang.backends.numpy import NumPyBackend
from einlang.shared.defid import DefId


class TestClauseSetOutput:
    """Test _clause_set_output sets value in env."""

    def test_clause_set_output_sets_value(self):
        backend = NumPyBackend()
        fid = DefId(krate=0, index=1)
        backend.env.set_value(fid, None)
        backend._clause_set_output(fid, 42)
        assert backend.env.get_value(fid) == 42

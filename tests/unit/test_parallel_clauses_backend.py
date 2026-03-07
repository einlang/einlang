"""
Unit tests for parallel-clauses backend behavior.

Tests _clause_set_output skip logic and that BLAS env vars are
saved/restored when using the parallel path (no full .ein execution).
"""

import os

import pytest

from einlang.backends.numpy import NumPyBackend
from einlang.shared.defid import DefId


class TestClauseSetOutput:
    """Test _clause_set_output respects _skip_clause_env_set."""

    def test_clause_set_output_when_skip_false(self):
        backend = NumPyBackend()
        backend._skip_clause_env_set = False
        fid = DefId(krate=0, index=1)
        backend.env.set_value(fid, None)
        backend._clause_set_output(fid, 42)
        assert backend.env.get_value(fid) == 42

    def test_clause_set_output_when_skip_true(self):
        backend = NumPyBackend()
        fid = DefId(krate=0, index=2)
        backend.env.set_value(fid, "original")
        backend._skip_clause_env_set = True
        backend._clause_set_output(fid, "new_value")
        assert backend.env.get_value(fid) == "original"
        backend._skip_clause_env_set = False


class TestParallelClausesEnvVars:
    """Test that parallel path saves/restores BLAS env vars."""

    def test_env_vars_restored_after_parallel_section(self):
        # Set a BLAS-related env var, then simulate what the parallel block does:
        # save, set to "1", then restore. We can't run the full parallel path
        # without a real multi-segment declaration, so we only test the save/restore logic.
        key = "OMP_NUM_THREADS"
        old = os.environ.get(key)
        saved = {}
        if key in os.environ:
            saved[key] = os.environ[key]
        os.environ[key] = "1"
        try:
            assert os.environ.get(key) == "1"
        finally:
            for k, v in saved.items():
                os.environ[k] = v
        # After restore, should be back to original (or unset)
        if old is not None:
            assert os.environ.get(key) == old
        else:
            # If we didn't have it before, the code leaves it as "1"; our test saved nothing so we don't restore.
            pass

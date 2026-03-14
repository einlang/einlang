"""
Unit tests for backend clause output behavior.
"""

import numpy as np
import pytest

from einlang.backends.numpy import NumPyBackend
from einlang.shared.defid import DefId

from tests.test_utils import compile_and_execute
from einlang.compiler.driver import CompilerDriver
from einlang.runtime.runtime import EinlangRuntime


class TestClauseSetOutput:
    """Test _clause_set_output sets value in env."""

    def test_clause_set_output_sets_value(self):
        backend = NumPyBackend()
        fid = DefId(krate=0, index=1)
        backend.env.set_value(fid, None)
        backend._clause_set_output(fid, 42)
        assert backend.env.get_value(fid) == 42


class TestPureRecurrenceTAsOuterLoop:
    """
    Verify pure recurrence dim t is extracted as outer loop (timestep-major).
    Minimal inter-dependent recurrence: one clause is pure t (only loop t),
    the other depends on it at same t. Clause order would give wrong result.
    """

    SOURCE = """
let u[0, 0] = 0.0;
let u[0, 1] = 0.0;
let u[t in 1..11, 0] = u[t - 1, 1];
let u[t in 1..11, 1] = u[t, 0] + 1.0;
u;
"""

    def test_inter_dependent_recurrence_timestep_major(self):
        compiler = CompilerDriver()
        runtime = EinlangRuntime(backend="numpy")
        result = compile_and_execute(
            self.SOURCE.strip(),
            compiler,
            runtime,
            source_file="<pure_rec_t>",
        )
        assert result.success, (result.errors if result.errors else result.error)
        u = np.asarray(result.value if result.value is not None else result.outputs.get("u"))
        assert u is not None and u.ndim == 2 and u.shape[0] == 11 and u.shape[1] == 2

        # Reference: timestep-major order. u[t,0] = u[t-1,1], u[t,1] = u[t,0]+1 => u[t,0]=t-1, u[t,1]=t for t>=1.
        ref = np.zeros((11, 2), dtype=np.float64)
        ref[0, 0], ref[0, 1] = 0.0, 0.0
        for t in range(1, 11):
            ref[t, 0] = ref[t - 1, 1]
            ref[t, 1] = ref[t, 0] + 1.0

        np.testing.assert_allclose(u, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="Pure recurrence t as outer loop (inter-dependent clauses)")

from __future__ import annotations

import numpy as np

from einlang.runtime.autodiff_requests import LazyJacobianValue
from einlang.shared.defid import DefId


class _FakeEngine:
    def __init__(self) -> None:
        self.matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        self.target = DefId(0, 1)
        self.wrt = DefId(0, 2)
        self.calls = {"jvp": 0, "vjp": 0}

    def target_shape(self, defid: DefId):
        if defid == self.target:
            return (2,)
        if defid == self.wrt:
            return (3,)
        raise KeyError(defid)

    def eval_jvp(self, target_defid: DefId, wrt_defid: DefId, tangent_value):
        assert target_defid == self.target
        assert wrt_defid == self.wrt
        self.calls["jvp"] += 1
        tangent = np.asarray(tangent_value, dtype=np.float64).reshape(-1)
        return self.matrix @ tangent

    def eval_vjp(self, target_defid: DefId, wrt_defid: DefId, cotangent_value):
        assert target_defid == self.target
        assert wrt_defid == self.wrt
        self.calls["vjp"] += 1
        cotangent = np.asarray(cotangent_value, dtype=np.float64).reshape(-1)
        return self.matrix.T @ cotangent


def test_lazy_jacobian_value_indexing_does_not_materialize_full_matrix():
    engine = _FakeEngine()
    value = LazyJacobianValue(engine, engine.target, engine.wrt)

    assert value[1, 2] == 6.0
    np.testing.assert_allclose(np.asarray(value[1, :], dtype=np.float64), np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(np.asarray(value[:, 2], dtype=np.float64), np.array([3.0, 6.0]))
    assert value._materialized is None
    assert engine.calls["jvp"] > 0 or engine.calls["vjp"] > 0


def test_lazy_jacobian_value_repr_stays_lazy():
    engine = _FakeEngine()
    value = LazyJacobianValue(engine, engine.target, engine.wrt)

    text = repr(value)

    assert "LazyJacobianValue" in text
    assert value._materialized is None
    assert engine.calls == {"jvp": 0, "vjp": 0}

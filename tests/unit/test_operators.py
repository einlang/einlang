#!/usr/bin/env python3
"""
Tests for operators - concatenated for speed.
"""

import pytest
from tests.test_utils import compile_and_execute


class TestOperators:
    """Complete operator coverage - small cases concatenated before compile/execute"""

    def _run(self, source: str, compiler, runtime):
        result = compile_and_execute(source, compiler, runtime)
        assert result.success, f"Execution failed: {result.errors}"

    def test_addition(self, compiler, runtime):
        source = """
        let o1 = 5 + 3; assert(o1 == 8);
        let o2 = 1 + 2 + 3; assert(o2 == 6);
        let o3a = 10; let o3b = 5; let o3 = o3a + o3b; assert(o3 == 15);
        """
        self._run(source, compiler, runtime)

    def test_subtraction(self, compiler, runtime):
        source = """
        let o4 = 10 - 4; assert(o4 == 6);
        let o5 = 20 - 5 - 2; assert(o5 == 13);
        let o6a = 15; let o6b = 7; let o6 = o6a - o6b; assert(o6 == 8);
        """
        self._run(source, compiler, runtime)

    def test_multiplication(self, compiler, runtime):
        source = """
        let o7 = 3 * 4; assert(o7 == 12);
        let o8 = 2 * 3 * 4; assert(o8 == 24);
        let o9a = 5; let o9b = 6; let o9 = o9a * o9b; assert(o9 == 30);
        """
        self._run(source, compiler, runtime)

    def test_division(self, compiler, runtime):
        source = """
        let o10 = 12 / 3; assert(o10 == 4);
        let o11 = 24 / 2 / 3; assert(o11 == 4);
        let o12a = 20; let o12b = 4; let o12 = o12a / o12b; assert(o12 == 5);
        let o13 = 100 / 10; assert(o13 == 10);
        let o14 = 1 / 1; assert(o14 == 1);
        let o15 = 1000 / 100; assert(o15 == 10);
        """
        self._run(source, compiler, runtime)

    def test_division_float_precision(self, compiler, runtime):
        source = """
        let o16 = 1.0 / 3.0; let o16y = o16 * 3.0; assert(o16y > 0.99 && o16y < 1.01);
        let o17 = 10.0 / 3.0; assert(o17 > 3.33 && o17 < 3.34);
        let o18 = 22.0 / 7.0; assert(o18 > 3.14 && o18 < 3.15);
        """
        self._run(source, compiler, runtime)

    def test_modulo(self, compiler, runtime):
        source = """
        let o19 = 10 % 3; assert(o19 == 1);
        let o20 = 15 % 4; assert(o20 == 3);
        let o21a = 17; let o21b = 5; let o21 = o21a % o21b; assert(o21 == 2);
        let o22 = 100 % 7; assert(o22 == 2);
        let o23 = 8 % 2; assert(o23 == 0);
        let o24 = 7 % 7; assert(o24 == 0);
        let o25 = 5 % 10; assert(o25 == 5);
        """
        self._run(source, compiler, runtime)

    def test_exponentiation(self, compiler, runtime):
        source = """
        let o26 = 2 ** 3; assert(o26 == 8);
        let o27 = 3 ** 2; assert(o27 == 9);
        let o28a = 2; let o28b = 4; let o28 = o28a ** o28b; assert(o28 == 16);
        let o29 = 10 ** 2; assert(o29 == 100);
        let o30 = 5 ** 3; assert(o30 == 125);
        let o31 = 2 ** 10; assert(o31 == 1024);
        let o32 = 1 ** 100; assert(o32 == 1);
        let o33 = 0 ** 5; assert(o33 == 0);
        """
        self._run(source, compiler, runtime)

    def test_pow_mixed_types(self, compiler, runtime):
        source = """
        let a = 2.0 ** 3;
        let b: f32 = 2.0; let n: i32 = 3; let c = b ** n;
        let d = 2.0 ** 0.5;
        """
        self._run(source, compiler, runtime)

    def test_equality(self, compiler, runtime):
        source = """
        let o34 = 5 == 5; assert(o34 == true);
        let o35 = 5 == 3; assert(o35 == false);
        let o36a = 10; let o36b = 10; let o36 = o36a == o36b; assert(o36 == true);
        """
        self._run(source, compiler, runtime)

    def test_inequality(self, compiler, runtime):
        source = """
        let o37 = 5 != 3; assert(o37 == true);
        let o38 = 5 != 5; assert(o38 == false);
        let o39a = 10; let o39b = 5; let o39 = o39a != o39b; assert(o39 == true);
        """
        self._run(source, compiler, runtime)

    def test_less_than(self, compiler, runtime):
        source = """
        let o40 = 3 < 5; assert(o40 == true);
        let o41 = 5 < 3; assert(o41 == false);
        let o42a = 2; let o42b = 7; let o42 = o42a < o42b; assert(o42 == true);
        """
        self._run(source, compiler, runtime)

    def test_greater_than(self, compiler, runtime):
        source = """
        let o43 = 5 > 3; assert(o43 == true);
        let o44 = 3 > 5; assert(o44 == false);
        let o45a = 8; let o45b = 3; let o45 = o45a > o45b; assert(o45 == true);
        """
        self._run(source, compiler, runtime)

    def test_less_equal(self, compiler, runtime):
        source = """
        let o46 = 3 <= 5; assert(o46 == true);
        let o47 = 5 <= 5; assert(o47 == true);
        let o48 = 5 <= 3; assert(o48 == false);
        """
        self._run(source, compiler, runtime)

    def test_greater_equal(self, compiler, runtime):
        source = """
        let o49 = 5 >= 3; assert(o49 == true);
        let o50 = 5 >= 5; assert(o50 == true);
        let o51 = 3 >= 5; assert(o51 == false);
        """
        self._run(source, compiler, runtime)

    def test_logical_and(self, compiler, runtime):
        source = """
        let o52 = true && true; assert(o52 == true);
        let o53 = true && false; assert(o53 == false);
        let o54 = false && true; assert(o54 == false);
        let o55 = false && false; assert(o55 == false);
        """
        self._run(source, compiler, runtime)

    def test_logical_or(self, compiler, runtime):
        source = """
        let o56 = true || true; assert(o56 == true);
        let o57 = true || false; assert(o57 == true);
        let o58 = false || true; assert(o58 == true);
        let o59 = false || false; assert(o59 == false);
        """
        self._run(source, compiler, runtime)

    def test_logical_not(self, compiler, runtime):
        source = """
        let o60 = !true; assert(o60 == false);
        let o61 = !false; assert(o61 == true);
        let o62a = true; let o62 = !o62a; assert(o62 == false);
        """
        self._run(source, compiler, runtime)

    def test_unary_minus(self, compiler, runtime):
        source = """
        let o63 = -5; assert(o63 == -5);
        let o64 = -(-3); assert(o64 == 3);
        let o65a = 10; let o65 = -o65a; assert(o65 == -10);
        """
        self._run(source, compiler, runtime)

    def test_unary_plus(self, compiler, runtime):
        source = """
        let o66 = +5; assert(o66 == 5);
        let o67 = +(-3); assert(o67 == -3);
        let o68a = -7; let o68 = +o68a; assert(o68 == -7);
        """
        self._run(source, compiler, runtime)

    def test_operator_precedence(self, compiler, runtime):
        source = """
        let o69 = 2 + 3 * 4; assert(o69 == 14);
        let o70 = 2 * 3 + 4; assert(o70 == 10);
        let o71 = 2 + 3 * 4 - 1; assert(o71 == 13);
        let o72 = (2 + 3) * 4; assert(o72 == 20);
        let o73 = 2 ** 3 ** 2; assert(o73 == 512);
        let o74 = 10 - 3 - 2; assert(o74 == 5);
        let o75 = 100 / 10 / 2; assert(o75 == 5);
        let o76 = 2 * 3 * 4; assert(o76 == 24);
        let o77 = (10 + 5) * (3 - 1); assert(o77 == 30);
        let o78 = 2 + 3 * 4 ** 2; assert(o78 == 50);
        let o79 = (2 + 3) * 4 ** 2; assert(o79 == 80);
        """
        self._run(source, compiler, runtime)

    def test_operator_associativity(self, compiler, runtime):
        source = """
        let o80 = 100 - 50 - 10; assert(o80 == 40);
        let o81 = 100 / 10 / 2; assert(o81 == 5);
        let o82 = 2 ** 3 ** 2; assert(o82 == 512);
        let o83 = 8 / 4 / 2; assert(o83 == 1);
        """
        self._run(source, compiler, runtime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

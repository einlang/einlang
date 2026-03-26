from __future__ import annotations

import json
import sys

from einlang.compiler.driver import CompilerDriver
from tests.print_at_fixtures import compile_capture_rewritten_print_at
from tests.unit.test_print_at_golden import GOLDEN_PRINT_CASES
from tests.unit.test_print_at_ml_smoke import ML_ACTIVATION_PRINT_AT_GOLDEN_CASES


def main() -> int:
    which = sys.argv[1]
    compiler = CompilerDriver()
    cases = {
        "golden": GOLDEN_PRINT_CASES,
        "ml_smoke": ML_ACTIVATION_PRINT_AT_GOLDEN_CASES,
    }[which]
    mismatches: dict[str, dict[str, str]] = {}
    for label, source, expected in cases:
        ok, actual, err = compile_capture_rewritten_print_at(source, compiler=compiler)
        if not ok:
            mismatches[label] = {"error": err}
        elif actual != expected:
            mismatches[label] = {"actual": actual, "expected": expected}
    print(json.dumps(mismatches, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

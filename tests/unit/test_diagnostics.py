"""Uncaught pass exceptions get IR-based spans instead of always 1:1."""

from pathlib import Path

from einlang.compiler.driver import CompilerDriver, span_for_uncaught_compile_exception


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_span_for_uncaught_prefers_entry_file_binding() -> None:
    from einlang.passes.ast_to_ir import ASTToIRLoweringPass
    from einlang.frontend.parser import Parser
    from einlang.passes.name_resolution import NameResolutionPass
    from einlang.passes.base import TyCtxt
    from einlang.passes.merge_diff_rules import merge_diff_rules_into_functions

    source = "let first = 1.0;\n\nlet second = 2.0;\n"
    path = "user_entry.ein"
    tcx = TyCtxt()
    tcx.source_files[path] = source
    p = Parser()
    ast = p.parse(source, path)
    merge_diff_rules_into_functions(ast)
    NameResolutionPass().run(ast, tcx)
    ir = ASTToIRLoweringPass().run(ast, tcx)

    loc = span_for_uncaught_compile_exception(RuntimeError("x"), ir, path)
    assert loc.line == 1
    assert "first" in source.split("\n")[loc.line - 1]


def test_compile_injected_pass_failure_points_past_line_one() -> None:
    from einlang.passes.type_inference import TypeInferencePass

    compiler = CompilerDriver()
    source = "let pad = 0.0;\n\nlet x = 1.0;\n"
    path = "span_test.ein"

    real_run = TypeInferencePass.run

    def boom(self, ir, tcx):
        raise RuntimeError("injected pass failure")

    try:
        TypeInferencePass.run = boom  # type: ignore[method-assign]
        result = compiler.compile(
            source, source_file=path, root_path=_REPO_ROOT
        )
    finally:
        TypeInferencePass.run = real_run  # type: ignore[method-assign]

    assert result.success is False
    assert result.tcx is not None
    assert result.tcx.reporter.has_errors()
    err = result.tcx.reporter.errors[0]
    assert err.location is not None
    assert err.location.line >= 1
    assert err.note is not None
    assert "TypeInferencePass" in err.note
    first_non_empty = next(i for i, ln in enumerate(source.splitlines(), start=1) if ln.strip())
    assert err.location.line == first_non_empty

from einlang.analysis.module_system.module_loader import _get_cache_dir
from einlang.compiler.driver import CompilerDriver
from einlang.frontend.parser import Parser
from einlang.shared.debug_trace import emit_debug_log
from einlang.utils.config import DEFAULT_PARSER_CACHE_FILE


def test_structured_debug_logging_does_not_create_files(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "einlang-debug.ndjson"

    monkeypatch.setenv("EINLANG_DEBUG_MODE", "1")
    monkeypatch.setenv("EINLANG_DEBUG_LOG_FILE", str(log_path))

    emit_debug_log("autodiff.runtime", "unit-test", "hello", {"x": 1})

    captured = capsys.readouterr()
    assert '"topic": "autodiff.runtime"' in captured.err
    assert not log_path.exists()


def test_parser_disk_cache_is_disabled_by_default():
    parser = Parser()
    parser.parse("let x = 1;", "<test>")

    assert DEFAULT_PARSER_CACHE_FILE is None


def test_module_parse_cache_requires_explicit_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("EINLANG_CACHE_DIR", raising=False)
    assert _get_cache_dir() is None

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("EINLANG_CACHE_DIR", str(cache_dir))

    resolved = _get_cache_dir()

    assert resolved == cache_dir
    assert cache_dir.exists()


def test_env_driven_ir_dump_does_not_write_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EINLANG_DUMP_FINAL_IR", "1")
    monkeypatch.setenv("EINLANG_DUMP_IR_PER_PASS", "1")

    result = CompilerDriver().compile("let x = 42;", str(tmp_path / "main.ein"), root_path=tmp_path)

    assert result.success
    assert not (tmp_path / "ir_dump").exists()
    assert not (tmp_path / "ir_dumps").exists()

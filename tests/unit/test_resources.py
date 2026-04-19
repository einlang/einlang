from pathlib import Path

import einlang.resources as resources


def test_default_stdlib_root_prefers_repo_layout():
    stdlib_root = resources.default_stdlib_root(Path(__file__))
    assert stdlib_root is not None
    assert stdlib_root.exists()
    assert stdlib_root.name == "stdlib"


def test_default_stdlib_root_falls_back_to_bundled_assets(monkeypatch, tmp_path):
    fake_package_root = tmp_path / "einlang"
    bundled_stdlib = fake_package_root / "_bundled" / "stdlib"
    bundled_stdlib.mkdir(parents=True)
    (bundled_stdlib / "mod.ein").write_text("// bundled stdlib marker\n", encoding="utf-8")

    fake_resources_file = fake_package_root / "resources.py"
    fake_resources_file.write_text("# synthetic package location\n", encoding="utf-8")

    monkeypatch.setattr(resources, "__file__", str(fake_resources_file))
    resources.bundled_stdlib_root.cache_clear()
    try:
        stdlib_root = resources.default_stdlib_root(tmp_path / "no-project-stdlib")
        assert stdlib_root == bundled_stdlib
    finally:
        resources.bundled_stdlib_root.cache_clear()

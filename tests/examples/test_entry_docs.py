#!/usr/bin/env python3
"""
Contract tests for the visitor-facing docs entry path.
"""

from pathlib import Path
import re
from typing import Set

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CURATED_FIRST_VISIT_PATHS = {
    "examples/hello.ein",
    "examples/autodiff_small.ein",
    "examples/demos/matrix_operations.ein",
    "examples/applications/linear_regression_autodiff.ein",
    "examples/recurrence/recurrence_suite.ein",
    "examples/ode/ode_suite.ein",
    "examples/mnist/main.ein",
}

ENTRY_DOCS_WITH_CURATED_PATH = (
    "README.md",
    "docs/GETTING_STARTED.md",
    "docs/README.md",
    "examples/README.md",
)

ARCHITECTURE_ENTRY_DOCS = (
    "README.md",
    "docs/README.md",
    "docs/index.md",
    "CONTRIBUTING.md",
)


def _read_text(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _extract_example_paths(text: str) -> Set[str]:
    return set(re.findall(r"examples/[A-Za-z0-9_./-]+\.ein", text))


def _extract_markdown_image_refs(text: str) -> Set[str]:
    refs = set()
    for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if ref.startswith(("http://", "https://")):
            continue
        refs.add(ref)
    return refs


def test_curated_first_visit_paths_exist():
    missing = sorted(path for path in CURATED_FIRST_VISIT_PATHS if not (PROJECT_ROOT / path).is_file())
    assert not missing, f"missing curated first-visit examples: {missing}"


@pytest.mark.parametrize("relative_path", ENTRY_DOCS_WITH_CURATED_PATH, ids=lambda path: path)
def test_entry_docs_reference_curated_first_visit_paths(relative_path):
    paths = _extract_example_paths(_read_text(relative_path))
    missing = sorted(CURATED_FIRST_VISIT_PATHS - paths)
    assert not missing, f"{relative_path} is missing curated first-visit examples: {missing}"

def test_architecture_guide_exists():
    assert (PROJECT_ROOT / "docs" / "ARCHITECTURE.md").is_file()


@pytest.mark.parametrize("relative_path", ARCHITECTURE_ENTRY_DOCS, ids=lambda path: path)
def test_architecture_guide_is_linked_from_entry_docs(relative_path):
    text = _read_text(relative_path)
    assert "ARCHITECTURE.md" in text or "docs/ARCHITECTURE" in text, (
        f"{relative_path} should link to the architecture guide"
    )
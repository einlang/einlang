from pathlib import Path
import shutil

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py


PROJECT_ROOT = Path(__file__).parent.resolve()
STDLIB_SOURCE = PROJECT_ROOT / "stdlib"
GRAMMAR_SOURCE = PROJECT_ROOT / "src" / "einlang" / "frontend" / "grammar.lark"
VERSION_NS = {}
exec((PROJECT_ROOT / "src" / "einlang" / "_version.py").read_text(encoding="utf-8"), VERSION_NS)
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

INSTALL_REQUIRES = [
    "numpy>=1.24",
    "lark>=1.1",
    "sexpdata>=1.0",
    "typing_extensions>=3.7",
]


class build_py(_build_py):
    """Copy runtime assets into the built package for wheel installs."""

    def run(self):
        super().run()

        build_root = Path(self.build_lib) / "einlang"
        bundled_stdlib = build_root / "_bundled" / "stdlib"
        bundled_stdlib.parent.mkdir(parents=True, exist_ok=True)
        if bundled_stdlib.exists():
            shutil.rmtree(bundled_stdlib)
        shutil.copytree(STDLIB_SOURCE, bundled_stdlib)

        grammar_target = build_root / "frontend" / "grammar.lark"
        grammar_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GRAMMAR_SOURCE, grammar_target)

setup(
    name="einlang",
    version=VERSION_NS["__version__"],
    description="A programming language for tensor computations with Einstein notation",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    license="Apache-2.0",
    url="https://github.com/einlang/einlang",
    author="Einlang contributors",
    author_email="opensource@einlang.dev",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={"console_scripts": ["einlang=einlang.__main__:main"]},
    cmdclass={"build_py": build_py},
)

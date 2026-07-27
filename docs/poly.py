"""Legacy development-only sphinx-polyversion preview.

Released documentation is never rebuilt by this file. Publication archives the exact candidate's
strict Sphinx output, and the Pages workflow authenticates and assembles those immutable archives.
This compatibility helper builds only ``main``/the local working tree for development preview.
"""

from datetime import UTC, datetime
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.pyvenv import Pip, VenvWrapper
from sphinx_polyversion.sphinx import SphinxBuilder

#: Branches to build docs for: the moving development line only.
BRANCH_REGEX = r"^main$"

#: Released tags are immutable archives and must never be rebuilt here.
TAG_REGEX = r"(?!)"

#: Output dir relative to the repo root.
OUTPUT_DIR = "docs/_build/html"

#: Source directory (relative to each checkout's root).
SOURCE_DIR = "docs"

#: Extra `pip install` args for mixle itself. CPU torch is installed separately (see the builder's
#: pre_cmd below) because its package index would otherwise shadow PyPI for every other dependency
#: in the same `pip install` invocation.
PIP_ARGS = ["-e", ".[docs]"]

#: Mock data used for `-l`/`--local` fast-iteration builds (working tree only, no other refs checked
#: out). The version-switcher partial just sees a single "local" entry in that case.
MOCK_DATA = {
    "revisions": [GitRef("local", "", "", GitRefType.BRANCH, datetime.now(UTC))],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.now(UTC)),
}

#: Whether to build using only local files and mock data (set via `-l`/`--local`, for fast iteration).
MOCK = False

#: Whether to run the builds in sequence instead of in parallel (set via `--sequential`).
SEQUENTIAL = False

# Load overrides read from the command line (e.g. `-o OUTPUT_DIR=...`).
apply_overrides(globals())

root = Git.root(Path(__file__).parent)
src = Path(SOURCE_DIR)

DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        predicate=file_predicate([src / "conf.py"]),  # skip refs that predate the docs/ tree
    ),
    builder=SphinxBuilder(
        src,
        pre_cmd=[
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
    ),
    env=Pip.factory(
        venv=Path(".venv"),
        args=PIP_ARGS,
        creator=VenvWrapper(),
        temporary=True,
    ),
    template_dir=root / src / "_root_templates",
    mock=MOCK_DATA,
).run(MOCK, SEQUENTIAL)

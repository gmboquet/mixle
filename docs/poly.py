"""sphinx-polyversion configuration: builds versioned docs (tags + main) into one site.

Run with ``sphinx-polyversion docs/poly.py``. Each matched git ref is checked out into its own
temporary venv, ``mixle[docs]`` is installed there, and its ``docs/`` tree is built with
``sphinx-build`` into ``OUTPUT_DIR/<ref-name>/``. ``versions.json`` (the list of built refs) lands
at ``OUTPUT_DIR/versions.json``; ``docs/_templates/sidebar/version-switcher.html`` reads it
client-side to render the version dropdown (Furo has no host-agnostic switcher of its own -- its
built-in one is Read the Docs-specific), so it works uniformly across every built version, including
tags whose own ``conf.py`` predates this setup. ``docs/_root_templates/index.html`` is rendered once
at the site root and redirects ``/`` to ``/main/`` (the latest merged, released state).

Iterate locally with ``sphinx-polyversion -l docs/poly.py`` (builds only the working tree, no other
refs) before running the real multi-version build, which is slower (one venv + full doc build per ref).
"""

from datetime import datetime, timezone
from pathlib import Path

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.pyvenv import Pip, VenvWrapper
from sphinx_polyversion.sphinx import SphinxBuilder

#: Branches to build docs for: the mainline (always the latest merged/released state).
BRANCH_REGEX = r"^main$"

#: Tags to build docs for: released versions, back to the first one with a docs/ tree.
TAG_REGEX = r"^v\d+\.\d+\.\d+$"

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
    "revisions": [GitRef("local", "", "", GitRefType.BRANCH, datetime.now(timezone.utc))],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.now(timezone.utc)),
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

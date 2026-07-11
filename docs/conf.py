"""Sphinx configuration for the mixle documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mixle.stats  # noqa: F401,E402

project = "mixle"
author = "Grant Boquet"
copyright = "2014-2026, Grant Boquet and contributors"

pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
release = pyproject["project"]["version"]
version = ".".join(release.split(".")[:2])


def _git_commit() -> str:
    """Short commit the docs were built from, so a page can name its exact source.

    Prefers a CI-provided SHA (GitHub Actions / Read the Docs) so the value is correct
    even when the build runs from an exported tree with no ``.git``; falls back to a
    local ``git`` call, then to ``"unknown"``. Never raises -- a docs build must not fail
    because provenance is unavailable.
    """
    import os
    import subprocess

    for env_var in ("MIXLE_DOCS_COMMIT", "GITHUB_SHA", "READTHEDOCS_GIT_COMMIT_HASH"):
        sha = os.environ.get(env_var)
        if sha:
            return sha[:12]
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


commit = _git_commit()

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ABSTRACTIONS.md",
    "ARCHITECTURE.md",
    "CAPABILITIES.md",
    "WORKPLAN.md",
    "interfaces/*.md",
    "interfaces/sections/*.md",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 0
myst_all_links_external = False
suppress_warnings = [
    "docutils",
    "myst.header",
    "myst.xref_missing",
]

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "no-index": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Several legacy library docstrings contain plain-text math and argument
# sketches that are not valid reStructuredText. Keep the Sphinx source pages
# strict while preventing inherited autodoc formatting debt from blocking
# public manual builds.
suppress_warnings = ["docutils"]

# Doctest is not enabled as a documentation release gate. The generated API
# reference includes wrapped NumPy/SciPy callables whose upstream examples are
# version-repr sensitive; executable examples are tracked through the example
# execution manifest instead.

# Optional runtime backends should not be required to build the API reference.
autodoc_mock_imports = [
    "dask",
    "distributed",
    "lightning",
    "mpi4py",
    "networkx",
    "numpyro",
    "pandas",
    "py4j",
    "pyarrow",
    "pymongo",
    "pyspark",
    "ray",
    "sage",
    "sqlalchemy",
    "umap",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme = "furo"
html_title = "mixle"
html_logo = None
html_favicon = str(Path(__file__).parent / "_static" / "mixle_icon.png")
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "mixle_logo_transparent_sidebar.png",
    "dark_logo": "mixle_logo_transparent_sidebar_dark.png",
    "light_css_variables": {
        "color-brand-primary": "#17615f",
        "color-brand-content": "#114e4c",
        "color-api-name": "#155b58",
        "color-api-pre-name": "#52616b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6bc6bf",
        "color-brand-content": "#8adbd4",
        "color-api-name": "#8adbd4",
        "color-api-pre-name": "#b7c6cc",
    },
}
html_css_files = ["mixle-docs.css"]

# Make the exact version and build commit available to every template so a reader can
# tell which release (and which source revision) they are looking at -- X12.5.
html_context = {
    "mixle_release": release,
    "mixle_version": version,
    "mixle_commit": commit,
}

# Furo ships no host-agnostic version switcher (its built-in one only activates under Read the Docs
# hosting) -- this repo's own _templates/sidebar/version-switcher.html reads the version list from
# switcher.json, rendered once at the site root by sphinx-polyversion (see docs/poly.py). Only takes
# effect on builds run through `sphinx-polyversion`; a plain `sphinx-build` (single-version, e.g. local
# `make html`) still renders the partial, but its fetch of `../switcher.json` 404s harmlessly -- the
# button just shows an empty menu.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/version-switcher.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

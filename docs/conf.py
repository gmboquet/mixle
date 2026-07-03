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

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "ABSTRACTIONS.md",
    "ARCHITECTURE.md",
    "CAPABILITIES.md",
    "LEDGER.md",
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
html_title = f"mixle {release}"
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

todo_include_todos = False

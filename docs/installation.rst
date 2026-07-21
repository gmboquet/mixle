Installation
============

``mixle`` supports Python 3.11 and newer. The PyPI package and import package
are both named ``mixle``.

Base Install
------------

.. code-block:: sh

   pip install mixle

The base install includes the local NumPy/SciPy path and core distribution
families. It is enough to score, sample, and fit ordinary distribution,
combinator, mixture, and HMM models locally.

Base install should not import heavyweight optional stacks at module import
time. If ``import mixle`` requires Torch, Spark, MPI, JAX, pandas, Arrow, or a
symbolic package, treat that as an installation bug unless the dependency is
guarded behind an explicitly requested extra.

Use the base install when validating ordinary docs examples. That catches
accidental imports from optional surfaces before they become package-index
failures for users who only need the core modeling library.

Extras
------

Install only the optional integrations you need:

.. list-table::
   :header-rows: 1

   * - Extra
     - Adds
     - Use when
   * - ``torch``
     - Torch engine, GPU/autograd, neural and Transformer leaves
     - using :doc:`neural-llm` or task distillation
   * - ``scientist``
     - Torch, Transformers, sentence-transformers, and datasets
     - running :mod:`mixle.scientist`, ``laptop_scientist.py``, or foundation
       capability distillation workflows
   * - ``numba``
     - JIT hot paths and TBB support
     - large local fits need faster kernels
   * - ``spark`` / ``dask`` / ``mpi``
     - distributed encoded-data backends
     - fitting on clusters or multi-process data
   * - ``jax``
     - JAX and NumPyro-backed routes
     - differentiable or probabilistic-programming experiments
   * - ``highprec``
     - mpmath arbitrary-precision fallback
     - high-precision engine operations without ``gmpy2``
   * - ``data``
     - pandas, Arrow, SQL, Mongo, fsspec connectors
     - loading from structured external data sources
   * - ``umap``
     - model-based UMAP helpers
     - embedding records or posterior features
   * - ``sympy`` / ``sage``
     - symbolic export
     - inspecting closed-form density expressions
   * - ``grammar``
     - NetworkX-backed grammar models
     - graph grammar workflows

Common installs:

.. code-block:: sh

   pip install "mixle[torch]"
   pip install "mixle[highprec]"
   pip install "mixle[scientist]"
   pip install "mixle[spark]"
   pip install "mixle[all]"

The ``all`` extra is a broad convenience bundle for common local acceleration,
distributed, data, Torch, and graph helpers. It is not every declared extra;
install narrower extras such as ``scientist``, ``jax``, ``sympy``, ``sage``,
``gmpy2``, or ``docs`` explicitly when those surfaces are needed.

The ``scientist`` extra installs Python packages only. The assembled
``mixle.scientist`` workflow loads open-weight models from the local Hugging
Face cache and sets offline defaults at import time; prepare those weights
explicitly before depending on that workflow.

For release validation, test the base install and the extras you document. A
passing ``mixle[all]`` environment does not prove the base package is clean, and
a passing base install does not prove optional integration imports are guarded
correctly.

Record extras separately in release evidence. A Torch failure, Spark failure,
or symbolic-export failure should identify the extra that was installed and
should not be reported as a base-install failure unless ``import mixle`` or a
core model import fails.

Development Install
-------------------

From a repository checkout:

.. code-block:: sh

   python -m venv .venv
   . .venv/bin/activate
   pip install -e ".[test,lint]"

For the broad convenience bundle plus test and lint tooling:

.. code-block:: sh

   pip install -e ".[all,test,lint]"

Use editable installs for development only. Release checks should install from
the built wheel in a fresh environment so missing package data, undeclared
dependencies, and accidental source-tree imports are caught.

Smoke Test
----------

.. code-block:: sh

   python - <<'PY'
   from mixle.inference import optimize
   from mixle.stats import GaussianEstimator

   model = optimize([1.0, 1.2, 0.9, 1.1], GaussianEstimator(), out=None)
   print(round(model.mu, 3))
   PY

For the neural quickstart:

.. code-block:: sh

   python examples/shared_embedding_example.py

Run example commands from a clean checkout or installed wheel when using them
as release evidence. Editable installs are useful while writing, but they can
hide missing package data or undeclared dependencies.

Documentation Build
-------------------

.. code-block:: sh

   . .venv/bin/activate
   pip install -e ".[docs]"
   make -C docs apidoc
   make -C docs html SPHINXOPTS="-W --keep-going"

The generated HTML lands in ``docs/_build/html``. That directory is local build
output. The generated ``docs/api/*.rst`` sources are part of the documentation
source tree: review and include changed API pages in the documentation PR when
modules are added, removed, or renamed.

The ``apidoc`` target includes selected private and implementation modules so
reviewers can audit the shipped package surface. For user-facing imports, start
with :doc:`api-overview` and the narrative guides rather than private autodoc
pages.

Fresh-Wheel Check
-----------------

Before treating an install as release evidence, build and install the artifact:

.. code-block:: sh

   python -m build
   python -m venv /tmp/mixle-wheel-check
   . /tmp/mixle-wheel-check/bin/activate
   pip install dist/mixle-*.whl
   python - <<'PY'
   import mixle
   from mixle.stats import GaussianDistribution

   print(mixle.__name__)
   print(GaussianDistribution(0.0, 1.0).log_density(0.0))
   PY

Run this from outside the repository checkout. Otherwise Python can accidentally
import local source files and hide packaging defects.

After the smoke import, run at least one fitted-model command from the wheel
environment and verify that generated artifacts do not depend on files inside
the source checkout.

Installation
============

``mixle`` supports Python 3.10 and newer. The PyPI package and import package
are both named ``mixle``.

Base Install
------------

.. code-block:: sh

   pip install mixle

The base install includes the local NumPy/SciPy path and core distribution
families. It is enough to score, sample, and fit ordinary distribution,
combinator, mixture, and HMM models locally.

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
   * - ``numba``
     - JIT hot paths and TBB support
     - large local fits need faster kernels
   * - ``spark`` / ``dask`` / ``mpi``
     - distributed encoded-data backends
     - fitting on clusters or multi-process data
   * - ``jax``
     - JAX and NumPyro-backed routes
     - differentiable or probabilistic-programming experiments
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
   pip install "mixle[spark]"
   pip install "mixle[all]"

Development Install
-------------------

From a repository checkout:

.. code-block:: sh

   python -m venv .venv
   . .venv/bin/activate
   pip install -e ".[test,lint]"

For all optional integrations:

.. code-block:: sh

   pip install -e ".[all,test,lint]"

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

   python examples/hybrid_llm_example.py

Documentation Build
-------------------

.. code-block:: sh

   . .venv/bin/activate
   pip install -r docs/requirements.txt
   make -C docs apidoc
   .venv/bin/sphinx-build -W -b html docs docs/_build/html

The generated HTML lands in ``docs/_build/html``.

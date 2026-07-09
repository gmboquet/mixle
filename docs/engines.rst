Compute Engines
===============

Compute engines separate model semantics from array execution. A distribution
owns the likelihood and sufficient-statistic math; an engine owns the array
representation, precision, symbolic payload, or device boundary used to execute
that math.

Use engines when you need GPU execution, JAX arrays, symbolic export, generated
kernels, explicit precision control, or safe conversion between array backends.

The model semantics should not change when the engine changes. Treat a new
engine route as an execution change that needs parity evidence against the
NumPy baseline before it is used in release notes, benchmarks, or production
artifacts.

Built-in Engines
----------------

``NumpyEngine``
    Default host engine and the baseline for local kernels.

``TorchEngine``
    Torch tensor engine for GPU/autograd-capable workflows and neural leaves.

``JaxEngine``
    JAX array engine and bridge to JAX/NumPyro-oriented workflows.

``SymbolicEngine``
    Symbolic expression engine for exporting log densities to SymPy, Sage, or
    LaTeX.

Basic Usage
-----------

Pass an engine to ``optimize`` without changing the model:

.. code-block:: python

   from mixle.engines import TorchEngine
   from mixle.inference import optimize

   model = optimize(
       data,
       estimator,
       engine=TorchEngine(device="cuda", dtype="float32"),
       out=None,
   )

Move data folding to a backend separately:

.. code-block:: python

   model = optimize(data, estimator, backend="mp", num_workers=4, out=None)

``engine=`` controls array math. ``backend=`` controls where encoded data are
processed.

Validate those concerns separately. If a distributed backend fails, reproduce
the same estimator locally first. If an engine route changes scores, compare
encoded payloads, dtype choices, and precision settings before changing the
model.

Torch DTensor Sharding
----------------------

``TorchEngine`` can represent component-sharded work through Torch DTensor when
the installed Torch version supports the operations Mixle needs. The fully
sharded component path is explicitly gated to Torch 2.5 or newer. Older
Torch versions expose partial DTensor APIs but lack sharding strategies for
operations used by mixture E-steps, which can otherwise fail deep inside Torch.

When the gate rejects a DTensor component-sharding request, use the
engine-agnostic route instead:

.. code-block:: python

   model = optimize(data, estimator, backend="model_parallel", out=None)

The native model-parallel backend is the portable choice across Torch versions
and devices. Use DTensor sharding only when the Torch runtime is new enough and
you have a specific reason to keep component tensors resident in a distributed
Torch mesh.

Record the Torch version, device type, mesh shape, and fallback route when
DTensor behavior is part of release evidence. A CPU-only smoke check does not
prove the distributed mesh path.

Engine Detection
----------------

``engine_of`` detects the owning engine of an encoded payload. ``to_numpy`` is
the explicit boundary for returning to NumPy.

.. code-block:: python

   from mixle.engines import engine_of, to_numpy

   engine = engine_of(encoded_payload)
   host_payload = to_numpy(encoded_payload)

Mixing incompatible array engines inside one payload raises an error instead of
silently moving data across devices.

Use this error as a boundary check. Hidden device transfers can make timing,
memory, and reproducibility evidence misleading, so conversions should be
explicit and visible in the workflow.

Precision
---------

Precision helpers route computations explicitly:

.. code-block:: python

   from mixle.engines import auto_precision, engine_with_precision

   precision = auto_precision(data, engine=engine)
   engine = engine_with_precision(engine, precision)

``optimize`` also accepts ``precision="auto"`` and ``precision="minimal"``.
Use ``auto`` for device-aware defaults and ``minimal`` for data-aware reduced
precision when verified safe.

Reduced precision is a release claim only after score parity, convergence, and
non-finite behavior have been checked on representative data. Keep the chosen
precision policy with the fitted artifact when it differs from the default.

The precision spectrum includes:

* ``DoubleDouble``, ``dd_sum``, and ``dd_dot`` for extended precision;
* ``Interval`` and ``sum_error_bound`` for error tracing;
* ``AffineForm`` and ``allocate_precision`` for uncertainty-aware allocation;
* ``FloatFormat``, ``FixedPointFormat``, and ``CodebookFormat`` for format
  experiments;
* ``accurate_sum`` and ``sum_certificate`` for stable reductions.

Symbolic Export
---------------

Symbolic engines make density expressions inspectable:

.. code-block:: python

   from mixle.engines import SYMBOLIC_ENGINE, to_latex, to_sympy

   symbolic = model.seq_log_density(encoded, engine=SYMBOLIC_ENGINE)
   expr = to_sympy(symbolic)
   latex = to_latex(symbolic)

Use this for reports, audits, or checking closed-form expressions. It is a
symbolic inspection tool, not a replacement for numeric fitting.

Symbolic export should be compared with a numeric evaluation on small inputs
when it is used as evidence. The exported expression explains the form; it does
not prove the numerical route is stable.

Registering Array Types
-----------------------

External array types can be associated with a compute engine:

.. code-block:: python

   from mixle.engines import register_array_type

   register_array_type(MyArray, my_engine)

Registering an array type makes ``engine_of`` and recursive payload inspection
route that type correctly.

Practical Guidance
------------------

* Start with the default NumPy path until the model shape is correct.
* Use ``TorchEngine`` for neural leaves and GPU-backed numeric work.
* Use ``backend=`` for parallel or distributed data folding.
* Use ``backend="model_parallel"`` for portable component parallelism across
  Torch versions.
* Use symbolic export for inspection, not for production scoring.
* Keep host/device boundaries explicit with ``to_numpy``.
* Use ``mixle.describe(model)`` to check whether a model supports backend
  scoring before assuming an engine will accelerate it.

Release Evidence
----------------

For engine-backed workflows, keep:

* the baseline NumPy score or fit result used for comparison;
* engine name, dtype, device, precision policy, and backend settings;
* optional dependency versions such as Torch, JAX, Spark, Dask, MPI, or SymPy;
* score-parity or convergence evidence on representative data;
* explicit host/device conversion points; and
* fallback behavior when an optional engine is unavailable.

This evidence prevents acceleration work from being mistaken for a change in
the statistical model.

API Map
-------

.. list-table::
   :header-rows: 1

   * - Import
     - Purpose
   * - ``NumpyEngine``, ``NUMPY_ENGINE``, ``FUSED_NUMPY_ENGINE``
     - local host execution
   * - ``TorchEngine``
     - Torch tensors, GPU, autograd-aware workflows
   * - ``JaxEngine``
     - JAX arrays and JAX-oriented routes
   * - ``SymbolicEngine``, ``SYMBOLIC_ENGINE``
     - symbolic payloads and expression export
   * - ``SymbolicExpression``
     - symbolic payload node used by symbolic export
   * - ``engine_of``, ``to_numpy``, ``register_array_type``
     - engine detection and explicit conversion
   * - ``normalize_numpy_dtype``, ``normalize_torch_dtype``
     - dtype normalization for engine setup
   * - ``auto_precision``, ``engine_with_precision``, ``precision_name``
     - precision routing
   * - ``to_sympy``, ``to_sage``, ``to_latex``
     - symbolic export formats
   * - ``DoubleDouble``, ``Interval``, ``AffineForm``
     - precision and error-analysis tools
   * - ``float64_sum_is_accurate``
     - quick check for whether ordinary float64 summation is sufficient

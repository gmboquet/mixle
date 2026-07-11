Scale-Out Economics: When a Backend Helps
=========================================

Switching ``optimize(..., backend=...)`` from local to a distributed backend
(Spark, Dask, Ray, MPI) is not automatically faster. This page states, from
measurement, when it is expected to help -- and when it is not.

How a distributed fit spends its time
-------------------------------------

Each EM iteration on a distributed backend does three things:

1. **Compute** -- every worker folds its data shard into sufficient statistics.
   This is O(N / workers) per iteration.
2. **Reduce** -- each worker's sufficient-statistic *payload*, exactly
   ``pickle.dumps((count, accumulator.value()))``, is gathered to the root and
   combined. See ``mixle/utils/parallel/mpi.py``.
3. **Broadcast** -- the root re-estimates the model and sends it back to every
   worker.

The reduce and broadcast are fixed overhead per iteration; only the compute
shrinks as workers are added.

The payload is O(model), not O(data)
------------------------------------

The key measured fact is that the sufficient-statistic payload does **not** grow
with the dataset size. It is a function of the *model*, not N. Measured directly
(``mixle/tests/reduction_payload_telemetry_test.py``):

============================  ===========  ================  ==================
Model                         payload      grows with N?     serialize
============================  ===========  ================  ==================
single Gaussian               ~188 bytes   no (flat 1e3-1e6)  ~5 microseconds
5-component Gaussian mixture   ~612 bytes   no                ~17 microseconds
============================  ===========  ================  ==================

A single Gaussian's payload is ~188 bytes whether it folds one thousand or one
million rows; a five-component mixture is ~612 bytes, likewise flat in N. The
payload scales with the number of model parameters, not with the data.

When distribution is expected to help
-------------------------------------

Because the reduce payload is tiny and fixed while compute scales with N,
distribution pays off when **per-shard compute time dominates the fixed
gather + fold + broadcast overhead**. Concretely:

* **Helps:** large N *and* non-trivial per-row work -- a rich model (large
  mixtures, HMMs, neural leaves), an expensive encoder, or per-row scoring that
  is not memory-bandwidth-bound. Here more workers cut the dominant compute term
  while the byte-sized payload is negligible.
* **Does not help:** small-to-moderate N with a cheap model (for example a plain
  Gaussian or a small mixture). The per-row work is so cheap that the local fit
  finishes before a distributed backend has paid its serialization and
  scheduling overhead. A ~188-byte reduce buys nothing if the whole local fit is
  milliseconds.
* **Backs up the wrong way:** a network with high per-message latency penalizes
  the per-iteration reduce/broadcast on every EM step; iteration count matters,
  not just data size.

Rule of thumb: distribute when the *single-worker* fit is compute-bound and takes
long enough that cutting compute by the worker count outweighs a fixed
per-iteration overhead measured in microseconds of serialization plus the
backend's own scheduling latency. If a local fit already finishes quickly, prefer
local execution.

Measuring for your own model
----------------------------

The payload and its serialization cost for any estimator/model pair can be
measured the same way the telemetry test does: build the accumulator, fold a
shard, and size ``pickle.dumps((count, accumulator.value()))``. If that payload
is small and flat in N (it will be for the classical families), the reduce cost
is not your bottleneck, and the decision comes down to per-shard compute versus
backend scheduling latency.

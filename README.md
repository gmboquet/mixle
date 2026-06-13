<p align="left">
  <img src="pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

**pysparkplug** is a Python package for distributed heterogeneous density estimation. With a few lines of code you can specify a complex probabilistic model over messy, variable-length, mixed-type records — tuples of categories, counts, reals, sets, sequences, trees — and fit it with EM, locally (vectorized NumPy + Numba) or at scale on Apache Spark.

The core idea: every distribution is **composable**. A Gaussian and a categorical compose into a tuple model; tuple models become mixture components; mixtures become HMM emissions; everything keeps the same five-part interface, so estimation, sampling, and vectorized evaluation work uniformly at any level of nesting.

## Installation

Python 3.10+ (developed against 3.12). NumPy 2.x, SciPy, Numba, and PySpark 4.x are supported.

The base install is lightweight (numpy, scipy, pandas, mpmath) and covers every distribution and local estimation:

```sh
pip install git+https://github.com/gmboquet/pysparkplug.git
```

Acceleration and integration back-ends are opt-in extras:

```sh
pip install "pysparkplug[numba] @ git+https://github.com/gmboquet/pysparkplug.git"   # JIT-compiled estimation
pip install "pysparkplug[spark] @ git+https://github.com/gmboquet/pysparkplug.git"   # distributed estimation on RDDs
pip install "pysparkplug[dask]  @ git+https://github.com/gmboquet/pysparkplug.git"   # distributed estimation on dask workers
pip install "pysparkplug[torch] @ git+https://github.com/gmboquet/pysparkplug.git"   # GPU/autograd engine
pip install "pysparkplug[umap]  @ git+https://github.com/gmboquet/pysparkplug.git"   # UMAP embeddings
pip install "pysparkplug[all]   @ git+https://github.com/gmboquet/pysparkplug.git"   # everything
```

Without the extras, numba-flagged code paths run as pure Python (correct, just slower), Spark inputs are unavailable, and `humap` is unavailable. `htsne` includes exact and internal Barnes-Hut engines in the base install. For development: `git clone` and `pip install -e ".[all]"`.

## Quickstart

Fit a two-component mixture over heterogeneous records of the form `(category, real value, variable-length count sequence)`:

```python
import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize

# Ground truth: a mixture of composite (tuple) distributions
component = lambda mu, p: CompositeDistribution((
    CategoricalDistribution({'a': p, 'b': 1.0 - p}),
    GaussianDistribution(mu, 1.0),
    SequenceDistribution(PoissonDistribution(mu + 5.0),
                         len_dist=CategoricalDistribution({2: 0.5, 3: 0.5})),
))
truth = MixtureDistribution([component(0.0, 0.8), component(5.0, 0.2)], [0.6, 0.4])

data = truth.sampler(seed=1).sample(2000)
# data[0] -> ('a', -0.3, [5, 4, 6])  — mixed types, variable length

# Estimators mirror the distribution structure
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100, rng=np.random.RandomState(1))
print(model.w)   # ≈ [0.6, 0.4]
```

## Core concepts

Each model family implements five cooperating pieces:

| Piece             | Role                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------- |
| `...Distribution` | Parameters + `log_density(x)` / vectorized `seq_log_density(enc)`                     |
| `...Sampler`      | Draw samples (`dist.sampler(seed).sample(size)`)                                      |
| `...Estimator`    | Specifies the model to fit; builds accumulators; closed-form M-step via `estimate()`  |
| `...Accumulator`  | Collects sufficient statistics (E-step), mergeable across data partitions             |
| `...DataEncoder`  | `seq_encode(data)` flattens raw Python data into NumPy arrays for the vectorized path |

Driver functions in `pysp.stats` and `pysp.utils.estimation` tie these together:

- `seq_encode(data, model=...)` — encode once, reuse across EM iterations
- `seq_initialize` / `seq_estimate` / `seq_log_density_sum` — one vectorized EM step at a time
- `optimize(data, est, max_its, ...)` — EM to convergence
- `optimize(..., engine=TorchEngine(...), precision='float32')` — the same EM contract through a compute engine
- `optimize(df, est, fields=[...], resources=...)` — DataFrame ingestion and advisory local placement through the same EM API
- `optimize(data, est, backend='mp')` — persistent multiprocessing encoding through the shared encoded-data protocol
- `optimize(data, est, backend='dask', client=client)` — dask.distributed worker-resident encoding through the same protocol
- `optimize(rdd, est, backend='spark')` — Spark RDD encoding through the same handle protocol
- `optimize(data, est, backend='torchrun')` — SPMD torch.distributed ranks through the same handle protocol
- `StreamingEstimator(est).update(...)` — decay-mode online EM over raw batches or encoded-data handles
- `IncrementalEstimator(est).update(chunk_id, ...)` — Neal-Hinton chunk-replacement EM for revisited data
- `fit_mle(enc, model, ...)` / `fit_map(enc, model, priors=...)` — autograd fitting over declaration-backed models
- `best_of(...)` — EM from multiple random restarts, keeping the best validation log-likelihood
- `pysp.utils.automatic.get_estimator(data)` — infer a reasonable estimator from raw data automatically
- `RecordDistribution` / `field(name, source=...)` — named dict/DataFrame observations, including repeated views of the same input variable

All of these accept either an in-memory list **or a Spark RDD** — the same code scales out by swapping the data argument.
Typed MAP priors include `ConditionalPrior` and `RecordPrior`, which route child priors by condition key or model field name.
For local multi-rank `torchrun` smoke jobs, prefer a loopback launch such as
`python -m torch.distributed.run --standalone --rdzv_endpoint=127.0.0.1:0 --local-addr=127.0.0.1 --nproc_per_node=2 ...`;
on macOS, leave `GLOO_SOCKET_IFNAME` unset if a forced loopback interface hangs,
while Linux loopback launches can set `GLOO_SOCKET_IFNAME=lo`.

## Compute engines and objectives

The compute-engine layer keeps backend mechanics out of distribution files:
distributions own likelihood/statistic math, while engines provide array
operations and device/precision policy.
Planning metadata is public through `capabilities_for(...)`,
`declared_distribution_types()`, `numpy_only_distribution_types()`, and
`generated_log_density(...)`, `generated_sufficient_statistics(...)`, and
`generated_numba_log_density(...)`, plus
`generated_log_density_diagnostics(...)`, so engine selection can distinguish
declaration-backed families, transitional legacy NumPy paths, generated
declaration paths, long-tail families intentionally kept NumPy-only, and the
symbolic formula behind a generated scorer. Generated numba scoring covers
declaration-backed scalar/vector/matrix leaves and composes through homogeneous composite,
optional-wrapper, and sequence mixtures whose children are generated-capable.
`NumbaKernelFactory` prefers those generated declaration kernels when the actual
stacked parameter bundle is valid, while retaining the legacy fused kernels for
support/table layouts that still need them. When the fused builder declines a
homogeneous wrapper mixture whose family-owned stacked hooks can compose the
children, the factory uses the generic stacked mixture route instead of
erroring.
`declaration_issues(...)` /
`validate_declaration(...)` provide schema
checks for generated-kernel metadata, while
`statistic_layout_issues(...)` / `validate_statistic_layout(...)` check real
accumulator payloads against declared statistic names and child roles.
`SymbolicEngine` is
available for lightweight expression tracing over scalar and object-array
formulas used in generated-kernel diagnostics.
Wrapper mixtures over composites and optional values use the same generic
stacked-component route, so generated child scorers compose without
distribution-specific code in the kernel layer.
When a `TorchEngine` is constructed with a torch `DeviceMesh` and
`shard="components"`, generated stacked-mixture component parameters and
mixture weights are placed on DTensor component shards while encoded arrays are
replicated on the mesh; explicit stacked routes can declare their own component
axis metadata for table layouts such as categorical support-by-component
matrices, while vector directional/ranking families such as
`VonMisesFisherDistribution` and `SpearmanRankingDistribution` can score with
precomputed normalizers on the active engine.

```python
from pysp.engines import TorchEngine
from pysp.utils.estimation import fit_map
from pysp.utils.priors import DirichletPrior, MixturePrior, NormalGammaPrior

enc = model.dist_to_encoder().seq_encode(data)
engine = TorchEngine(device="cpu", dtype="float64")

fitted, objective = fit_map(
    enc,
    model,
    engine=engine,
    priors=MixturePrior(
        components=[NormalGammaPrior(mu0=-2.0), NormalGammaPrior(mu0=2.0)],
        weights=DirichletPrior([2.0, 2.0]),
    ),
)
```

Set `return_result=True` on `fit_mle` / `fit_map` to inspect objective
history, convergence, best iteration/value, final raw-gradient norm, and final
likelihood/prior contributions, including `prior_sensitivity` for how much of
the final objective magnitude came from the prior term. Typed MAP priors attach
to constrained parameters, including Gamma priors on positive ordered-bound
deltas such as `high_minus_low`.
`calibrate_resources(..., workload="score"|"estep"|"em")` can time scoring or
small E-step/EM workloads, and calibrated `Resources` can be saved with
`resources.save(path)` and reused with `Resources.load(path)` when planning
later runs. Pass `catalog_path=...` to append model/workload calibration
records that can be reloaded with `CalibrationCatalog.load(...)`.
`LocalEncodedData` now keeps each shard's historical host encoding alongside a
numeric engine-resident encoding, so local scoring kernels do not repeatedly
recreate Torch/mesh tensors while legacy accumulators can still consume their
unchanged NumPy payloads.
For homogeneous Gaussian, LogGaussian, Gamma, Exponential, Poisson, and
Bernoulli mixtures, `StackedMixtureKernel.resident_accumulate` also computes
posterior-weighted sufficient statistics on the active engine and converts
the small result back to the ordinary `MixtureEstimator` M-step contract.
`StackedMixtureResidentStats.local_value()` and
`estimate_component_shard(...)` expose the component-local M-step needed by
component-sharded model-parallel orchestrators without gathering every
component statistic back to the driver. `estimate_component_shard_value(...)`
does the same M-step from an explicit shard payload after cross-rank reduction,
and `tie_component_shard_values(...)` applies the existing key-tying protocol
to component-sharded statistic ranges before those local M-steps.
Beta, Rayleigh, Geometric, NegativeBinomial, StudentT, Logistic, Weibull, and
DiagonalGaussian vector-moment mixtures demonstrate the same resident reduction
generated from local declarations, without family-specific stacked-stat
methods.
Full-covariance MultivariateGaussian mixtures use declaration-generated
matrix-valued exp-family scoring over cached inverse-covariance parameters,
while keeping matrix-moment resident statistics family-owned for the existing
M-step.
Dirichlet mixtures use a distribution-owned explicit vector-stat route for
engine-backed scoring and resident sufficient-statistic reduction while keeping
the iterative M-step in the existing estimator.
Object/integer categorical, Bernoulli-set, integer-uniform-spike, integer
Bernoulli-set, and integer-multinomial mixtures keep their table/support layouts
family-owned while reducing count-map/count-vector statistics resident on the
active engine.
Null mixtures use an explicit zero-score/empty-stat route, keeping placeholder
fields resident without any family-specific high-level special case.
PointMass mixtures with identical fixed atoms use an explicit zero-or-impossible
score route with empty resident statistics, including legacy-compatible
posterior weights for observations outside all component supports.
Fixed-support MarkovChain mixtures keep initial-state, transition, and length
statistics on the stacked route while returning the legacy sparse count-map
payloads for the existing estimator.
IntegerMarkovChain mixtures use the same family-owned route for grouped
transition-table counts plus composed initial/length child statistics.
Conditional, Sequence, Multinomial, IntegerMultinomial, Optional, Composite,
DiracLengthMixture, Record, Select, Ignored, Weighted, and Transform mixtures
compose those same child resident-stat routes, so condition-key/given models,
variable-length element/length models, count-vector value/trial-count and
integer-count/trial-count models, length-or-dirac mixtures, missing-value
wrappers, tuple/named-record products, fixed choice-routed partitions, fixed
empty-stat fields, per-observation weighted data, and fixed inverse-transform
models do not need to fall back to host accumulators when their children are
resident-capable.
Uniform and Pareto keep their support-bound updates family-owned while still
running the reductions resident on the active engine.
Binomial uses the same resident path with an estimator-aware family hook so
the support bounds seeded by its accumulator factory remain identical to the
legacy M-step contract.
Laplace keeps the raw `(values, weights)` payload required by its exact
weighted-median M-step in a family-owned resident hook.

For custom differentiable problems, `pysp.utils.objectives` exposes
`fit_objective`, `ExpectedLogDensity`, `variational_projection`, and
`UnnormalizedLogLikelihood`. For likelihoods that are not distributions at
all, use `ObjectiveParameter` and `fit_parameter_objective` to optimize named
real, positive, unit-interval, simplex, matrix-simplex, or coupled bound
parameters such as `greater_than:low` while the objective owns the
model-specific math. Pass `return_result=True` to receive
`ObjectiveFitResult` diagnostics, including objective history, convergence,
best iteration/value, final raw-gradient norm, and improvement. The lower-level
`optimize_torch_objective(..., return_result=True)` reports the same diagnostics
for arbitrary trainable Torch tensors, and the higher-level
`GaussianProcessRegressor.fit(..., return_result=True)` /
`GaussianRegressionNN.fit(..., return_result=True)` /
`CategoricalClassificationNN.fit(..., return_result=True)` /
`PoissonRegressionNN.fit(..., return_result=True)` helpers forward those
diagnostics for non-iid modeling experiments. Objective helpers restore the
best observed model/parameter state by default, so an oversized optimizer step
does not replace the best state already found:

```python
from pysp.utils.objectives import ObjectiveParameter, fit_parameter_objective

params, value = fit_parameter_objective(
    [
        ObjectiveParameter("mu", 0.0),
        ObjectiveParameter("sigma2", 1.0, constraint="positive"),
    ],
    lambda p, enc, engine: engine.sum(
        -0.5 * ((engine.asarray(enc) - p["mu"]) ** 2 / p["sigma2"]
                + engine.log(engine.asarray(6.283185307179586) * p["sigma2"]))
    ),
    enc=data,
    engine=engine,
)
```

Model helpers in `pysp.models` keep non-iid
objective math modular too, including Gaussian-process regression, neural
Gaussian regression, categorical classification, count regression, random graph
models, POMDPs, knowledge graphs, grammar learning, and
conditional-dependence / causal-skeleton utilities.
EM strategy objects in `pysp.utils.em` include standard, hard,
deterministic-annealing, generalized, objective-gated acceleration, ECM,
Monte Carlo, variational, online/stochastic, incremental chunk-replacement,
and restart EM over the same estimator/kernel contracts.
For posterior simulation or unnormalized targets, `pysp.utils.mcmc` provides
generic Metropolis-Hastings utilities over ordinary `dist.log_density(x)`
objects or user-supplied log-target callables, including adaptive random-walk
and adaptive-covariance proposals, mixture, block, Langevin, and Hamiltonian
Monte Carlo transitions, posterior predictive sampling helpers, plus
acceptance/ESS/MCSE diagnostics.
On the Bayesian side, `pysp.bstats.BayesianStreamingEstimator` supports
posterior-carry recursive Bayes and forgetting/power-prior updates over the
existing conjugate estimator protocol.

The old `pysp.stats.torch_engine.TorchMixture` import is retained only as a
small compatibility shim for existing code. New code should use `TorchEngine`,
`dist.kernel(engine=...)`, `optimize(..., engine=...)`, `fit_mle` / `fit_map`,
and `pysp.utils.objectives`.

## Distribution catalog

Around 60 composable families in `pysp.stats`, including:

- **Scalar/basic:** Gaussian, StudentT/Cauchy, Logistic, LogGaussian/log-normal, Laplace, Uniform, Exponential, Gamma/chi-square, Beta, Weibull, Rayleigh, Pareto, Poisson, Bernoulli, Geometric, Binomial, Negative Binomial, von Mises-Fisher, multivariate/diagonal Gaussian, Dirichlet, categorical & integer-categorical
- **Combinators:** `CompositeDistribution` (tuples), `SequenceDistribution` (variable-length i.i.d. with length model), `OptionalDistribution` (missing data), `TransformDistribution` (fixed invertible transforms), `IgnoredDistribution`, `ConditionalDistribution`, `WeightedDistribution`, fixed `PointMassDistribution`
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA, PLSI, hidden Markov models (standard, segmental/variable-emission, lookback, tree-structured), heterogeneous/induced PCFGs, Markov chains, hidden associations, Spearman ranking, Bernoulli set/edit-set models
- **Bayesian (`pysp.bstats`):** conjugate-prior/variational counterparts, finite mixtures with joint weight/component priors via `mixture_prior(...)`, posterior-carry/forgetting streaming, and Dirichlet-process mixtures (`bexamples/` shows DPM auto-modeling)

Estimators accept `pseudo_count` for regularization and `keys` for tying sufficient statistics across model parts. Several models (HMMs, tree HMM, PLSI) take `use_numba=True` to switch to parallel Numba kernels; the first call pays a JIT compile that is cached afterwards.

Hidden-association models are the closest probabilistic analogue to attention in this library: they infer soft latent alignments between observed items and hidden explanatory slots, but the mechanism is a generative EM model rather than a transformer-style learned query/key/value layer.

### API naming conventions

The public API is converging on a single stem per family (`<Stem>Distribution`, `<Stem>Estimator`,
`<Stem>Sampler`, `<Stem>Accumulator`, `<Stem>AccumulatorFactory`, `<Stem>DataEncoder`,
`<Stem>Enumerator`) and descriptive constructor argument names. Both the legacy and preferred
spellings work — the old names are kept as aliases — but new code should prefer:

- Constructor arguments: `weights` (over `w`), `prob_map` (over `pmap`), `prob_vec` (over `p_vec`),
  `covariance` (over `covar`), `num_values` (over `num_vals`), `max_iter` (over `max_its`).
- Class names: e.g. `HiddenMarkovModelEstimator` (alias of `HiddenMarkovEstimator`),
  `ConditionalEstimator`/`ConditionalAccumulator` (aliases of the `ConditionalDistribution*`
  classes), `GrammarAccumulator` (alias of `GrammarEstimatorAccumulator`), and the
  `*Accumulator`/`*AccumulatorFactory` aliases for families that historically used
  `*EstimatorAccumulator`.

Passing both the legacy and preferred argument spelling raises `TypeError`. See
`notes/distribution_api_naming_accounting.md` for the full target convention and migration plan.

## Examples

Local examples live in [examples/examples_pysp/](examples/examples_pysp/) (Bayesian counterparts in [examples/examples_bayes/](examples/examples_bayes/)) and run from that directory:

```sh
cd examples/examples_pysp
python mixture_example.py
python hidden_markov_example.py
python accelerated_engines_example.py
```

Multiprocessing and MPI twins of the mixture example live in [examples/examples_mp/](examples/examples_mp/) and [examples/examples_mpi/](examples/examples_mpi/) (`pip install pysparkplug[mpi]`, run with `mpiexec -n 4 python examples/examples_mpi/mixture_example.py`).

A few examples need datasets that are not shipped: `set_example.py` (NIPS submissions) and the two `wikipedia_*` examples (a Wikipedia corpus + stop-word list). Everything else generates its own data — e.g. `hmm_numba_example.py` fits an HMM to generated text-like sequences and compares Numba vs. pure-NumPy fitting, while `accelerated_engines_example.py` compares the ordinary sequence path with `NumbaKernelFactory`, `TorchEngine`, `fit_mle`, and `fit_map`. `lda_example.py` runs to a tight tolerance and takes a while.

## Running on Spark

Spark examples are in [examples/examples_spark/](examples/examples_spark/). PySpark 4.x needs a JVM (Java 17 or 21) and the workers must use the same Python as the driver:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # or your JDK 17/21 path
export PYSPARK_PYTHON=/path/to/your/venv/bin/python
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON

python examples/examples_spark/mixture_example.py
```

The estimation helpers detect RDD inputs automatically: sampling (`pysp.stats.rdd_sampler`), initialization, encoding, and each EM step run per-partition, and sufficient statistics are merged on the driver — so a model fit locally and one fit on a cluster go through identical math.

## Tests

```sh
python -m pytest -m fast
python -m pytest -m "not optional and not benchmark"
```

The test suite still uses `unittest.TestCase` internally, but pytest provides
collection, markers, and CI tiers.  Marker conventions live in
[`pysp/tests/README.md`](pysp/tests/README.md).  `base_dist_test.py` checks each
enabled distribution end-to-end: sampler repeatability, `str`/`eval`
round-trips, vectorized-vs-scalar log densities, and that EM-to-convergence
improves (in KL) with more data.

## License

pysparkplug is distributed under the terms of the MIT license. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Originally developed at Lawrence Livermore National Laboratory (LLNL-CODE-844837).

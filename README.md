<p align="left">
  <img src="pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-1600%2B-brightgreen)

**Composable, distributed density estimation for messy, mixed-type records** — tuples of
categories, counts, reals, vectors, sets, sequences, and trees. Specify a probabilistic model in a
few lines and fit it with EM — locally (vectorized NumPy + Numba) or at scale on Spark, Dask, or
Torch (GPU).

- **Mixed-type & composable** — a Gaussian and a categorical compose into a tuple model, tuple
  models become mixture components, mixtures become HMM emissions. Nest to any depth.
- **One interface, everywhere** — every family shares the same five parts (distribution · sampler ·
  estimator · accumulator · encoder), so sampling, scoring, and estimation work uniformly.
- **Fit anywhere** — the same `optimize(...)` call runs local NumPy/Numba or distributed Spark /
  Dask / Torch by swapping one argument.
- **Frequentist *or* Bayesian** — MLE, MAP, conjugate posteriors, and variational mixtures
  (Dirichlet processes) selected by a single `prior=` switch.
- **A PPL surface** — [`pysp.ppl`](#probabilistic-programming-pyspppl): put `free` or another
  distribution in any parameter slot, then `.fit().sample().posterior()`.

## Installation

Python 3.10+ (developed on 3.12). The base install (numpy, scipy, pandas, mpmath) covers every
distribution and local estimation:

```sh
pip install git+https://github.com/gmboquet/pysparkplug.git
```

Back-ends are opt-in extras — `numba` (JIT estimation), `spark` / `dask` (distributed),
`torch` (GPU/autograd), `umap`, or `all`:

```sh
pip install "pysparkplug[all] @ git+https://github.com/gmboquet/pysparkplug.git"
```

Without extras, numba-flagged paths run as pure Python (correct, slower) and Spark/Dask inputs are
unavailable. For development: `git clone` then `pip install -e ".[all]"`.

## Quickstart

Fit a two-component mixture over heterogeneous `(category, real, variable-length count sequence)`
records:

```python
import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize

component = lambda mu, p: CompositeDistribution((
    CategoricalDistribution({'a': p, 'b': 1.0 - p}),
    GaussianDistribution(mu, 1.0),
    SequenceDistribution(PoissonDistribution(mu + 5.0),
                         len_dist=CategoricalDistribution({2: 0.5, 3: 0.5})),
))
truth = MixtureDistribution([component(0.0, 0.8), component(5.0, 0.2)], [0.6, 0.4])
data = truth.sampler(seed=1).sample(2000)   # data[0] -> ('a', -0.3, [5, 4, 6])

# Estimators mirror the distribution structure
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100, rng=np.random.RandomState(1))
print(model.w)   # ≈ [0.6, 0.4]
```

## Probabilistic programming (`pysp.ppl`)

`pysp.ppl` is a concise, optional dialect over the same distributions. **One rule:** any parameter
slot is a value, the token `free` (estimate it), or another distribution (a prior — random /
hierarchical):

```python
from pysp.ppl import Normal, Mix, Markov, free

Normal(0.0, 1.0)             # value        — fixed parameter
Normal(free, free)           # free         — estimate mean & sd
Normal(Normal(0, 10), 1.0)   # distribution — a prior on the mean (hierarchical)
```

Build a model, `.fit(data)`, then query with `.sample` / `.log_prob` / `.posterior` / `.params`:

```python
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)   # 2-component Gaussian mixture
m.posterior(data)                                             # responsibilities
Markov(Normal(free, free), states=2).fit(sequences)          # 2-state Gaussian HMM (k-means++ seeded)
```

A `free` coefficient times a `Field` turns the same surface into generalized linear models:

```python
from pysp.ppl import Normal, Bernoulli, Poisson, Field, free

Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})  # linear
Bernoulli(free * Field("x") + free).fit(y, given={"x": x})                                 # logistic
Poisson(free * Field("x") + free).fit(y, given={"x": x})                                   # Poisson
```

`how=` selects the engine — `auto` (default) takes an exact route when one exists, else falls back to
EM / gradient / sampling:

```python
Mix([Normal(free, free), Normal(free, free)]).fit(data, how="nuts")
Markov(Normal(free, free), states=2).fit(seqs, how="ensemble", chains=4, parallel=True)  # R̂ + pooled ESS
# how = auto | conjugate | conjugate_mixture | em | map | vi | vmp | mcmc | hmc | nuts | ensemble
```

**Constraints** among random variables are plain comparisons (combine with `& | ~`); they shape
both inference and sampling:

```python
a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Mix([Normal(a, 1), Normal(b, 1)]).fit(data, constraints=a < b)   # ordered means break label-switching
constrain(2*a - b >= 1).sample(100)                              # draw from the truncated joint
```

```python
# model constructors: Mix · Seq · Markov · LDA · MVN · DiagGaussian · LocalLevel · AR1 · Graph
compare([model_a, model_b], data)   # rank fitted models, best first
```

It's a thin surface — the `pysp.stats` classes underneath are untouched.

**Head-to-head speed** (`python -m pysp.ppl.benchmark_vs`, same machine/data/model vs the actual
competing PPLs):

| task | pysp.ppl | competitor | result |
| ---- | -------- | ---------- | ------ |
| Poisson-Gamma posterior, N=200k | **5 ms** (exact, 1 pass) | NumPyro NUTS 5690 ms | **~1000× faster**, identical |
| Beta-Bernoulli posterior, N=100k | **3 ms** (exact) | NumPyro NUTS 3619 ms | **~1400× faster**, identical |
| Gaussian MLE, N=500k | **45 ms** (EM) | Pyro SVI 11778 ms | **~260× faster**, same answer |
| Gaussian posterior (ESS/sec) | **8945** (`how='ensemble'`) | emcee 7883 · NumPyro 624 | **highest mixing throughput** |

For conjugate / exponential-family / mixture models pysp returns the *exact* posterior with no
sampling; for general posteriors the ensemble sampler leads on ESS/sec. See
[`pysp/ppl/BENCHMARKS.md`](pysp/ppl/BENCHMARKS.md).

## Core concepts

Each model family implements five cooperating pieces:

| Piece             | Role                                                                        |
| ----------------- | -------------------------------------------------------------------------- |
| `...Distribution` | Parameters + `log_density(x)` / vectorized `seq_log_density(enc)`          |
| `...Sampler`      | Draw samples (`dist.sampler(seed).sample(size)`)                           |
| `...Estimator`    | Specifies the model to fit; closed-form M-step via `estimate()`           |
| `...Accumulator`  | Collects sufficient statistics (E-step), mergeable across partitions       |
| `...DataEncoder`  | `seq_encode(data)` flattens raw Python data into NumPy for the fast path   |

`optimize(data, est)` (in `pysp.utils.estimation`) ties these together — EM to convergence locally
(vectorized NumPy/Numba), scaling out to Spark/Dask/Torch/MPI by swapping one argument (see
[Engines & orchestration](#engines--orchestration)).

Also available: `best_of` (random restarts), `StreamingEstimator` / `IncrementalEstimator`
(online EM), `fit_mle` / `fit_map` (autograd fitting with typed priors), `RecordDistribution` /
`field(...)` (named dict/DataFrame observations), and `pysp.utils.automatic.get_estimator(data)`
(infer an estimator straight from raw data).

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the
array ops, device, and precision — so the same EM contract runs unchanged on NumPy, Numba, Torch
(CPU / GPU / multi-device), or a symbolic backend.

```python
from pysp.engines import TorchEngine

optimize(data, est, engine=TorchEngine(device="cuda", dtype="float32"))   # GPU
optimize(data, est, engine=TorchEngine(mesh=mesh, shard="components"))    # multi-GPU (DTensor)
```

**Precision is data-aware** — `precision='auto'` picks float32/float64 from the data and engine, and
sufficient statistics always accumulate in float64, so reduced precision stays safe:

```python
optimize(data, est, precision="auto")
```

**Scale by swapping the back-end, not the model** — local and distributed go through identical math:

```python
optimize(data, est)                                   # local NumPy / Numba
optimize(data, est, backend="mp", num_workers=8)      # multiprocessing
optimize(rdd,  est, backend="spark")                  # Spark
optimize(data, est, backend="dask", client=client)    # Dask
optimize(data, est, backend="mpi", comm=comm)         # MPI / torchrun
```

**The planner** (`pysp.planner`) turns a hardware budget into a memory-aware *placement* — chunking,
device assignment, and (on Torch) model sharding — that you compute once and reuse:

```python
from pysp.planner import plan, Resources

placement = plan(data, model=model, estimator=est, resources=Resources.local(num_cpus=8))
optimize(data, est, placement=placement)
optimize(data, est, resources=Resources.local(num_cpus=8))   # or let optimize plan it for you
```

`Resources.{single_cpu, local, from_spark, from_dask, from_mpi, from_specs}` describe the hardware.

**Symbolic export.** The `SymbolicEngine` runs a distribution's density through SymPy, so a model can
emit its closed-form log-density as LaTeX / SymPy / Sage:

```python
from pysp.engines import SYMBOLIC_ENGINE, to_latex

x = SYMBOLIC_ENGINE.symbol("x")
to_latex(GaussianDistribution(0.0, 1.0).backend_seq_log_density(x, SYMBOLIC_ENGINE))
# '- 0.5 x^{2} - 0.918938533204673'
```

## Frequentist & Bayesian — one switch

The prior is the single switch — no prior is maximum likelihood, a conjugate `prior=` makes the
same machinery Bayesian:

```python
from pysp.utils.priors import NormalGammaPrior

GaussianEstimator()                          # MLE
GaussianEstimator(prior=NormalGammaPrior())  # closed-form conjugate posterior — same EM call
```

`optimize` / `fit` auto-select the objective from the model (MLE, MAP, or variational ELBO; force it
with `objective=`); `fit(...)` returns the posterior, `BayesianStreamingEstimator` carries it across
batches, and `pysp.stats.dpm` / `hdpm` add (hierarchical) Dirichlet-process mixtures. Gradient MAP
with typed priors is first-class too:

```python
from pysp.engines import TorchEngine
from pysp.utils.fit import fit_map
from pysp.utils.priors import DirichletPrior, MixturePrior, NormalGammaPrior

enc = model.dist_to_encoder().seq_encode(data)
fitted, objective = fit_map(enc, model, engine=TorchEngine(device="cpu", dtype="float64"),
                            priors=MixturePrior(
                                components=[NormalGammaPrior(mu0=-2.0), NormalGammaPrior(mu0=2.0)],
                                weights=DirichletPrior([2.0, 2.0])))
```

## Enumeration & ranking

Discrete and structured models can **enumerate their support in descending-probability order** and
answer exact **rank / cumulative-probability** queries — even when the support is enormous or
unbounded.

```python
from pysp.utils.density_rank import density_rank, count_dp_seek

dist.enumerator().top_k(5)          # the 5 most probable (value, log_prob), in order
density_rank(dist, value)           # exact-head + sampling rank & CDF of an observation
count_dp_seek(dist, index=10_000)   # the ~10,000th most probable value, by structural count-DP
```

For decomposable families (`Composite` / `Sequence` / `MarkovChain`), rank↔value is an exact count
dynamic program at any depth (`count_dp_rank`, `count_dp_seek`, `cumulative_probability`,
`mixture_cross_rank`). For very large or infinite supports, **budget-bounded quantized indexes** seek
and unrank over just the most-probable region without enumerating everything:

```python
index = dist.count_budget_index(budget_bits=20)               # index the top ~2**20 values
for value, log_prob in dist.count_budget_distinct(budget_bits=20):
    ...
```

`pysp.utils.enumeration` provides the shared machinery (bounded best-first union, quantization,
Kronecker-substitution count convolution).

## Distribution catalog

~90 composable families live in `pysp.stats`, grouped into subpackages (`leaf`, `multivariate`,
`combinator`, `sets`, `latent`, `graph`, `bayes`, `compute`) but all re-exported at the top level —
`from pysp.stats import GaussianDistribution` works regardless of where the file lives.

- **Scalar / basic:** Gaussian, Student-t / Cauchy, Logistic, LogGaussian, Laplace, Uniform,
  Exponential, Gamma, Inverse Gamma, Inverse Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh,
  Pareto, Poisson, Bernoulli, Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet,
  categorical, plus multivariate / diagonal Gaussian, von Mises–Fisher, and multivariate Student-t.
- **Combinators:** `CompositeDistribution` (tuples), `SequenceDistribution`, `OptionalDistribution`
  (missing data), `TransformDistribution`, `ConditionalDistribution`, `WeightedDistribution`.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains,
  hidden associations, IBP, Pitman-Yor processes, Bernoulli sets.
- **Permutations & graphs:** Mallows and Plackett-Luce rankings, matchings, spanning trees, random
  graphs (Erdős–Rényi, stochastic block, random dot-product), Spearman ranking.
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet, SymmetricDirichlet)
  and variational Dirichlet-process / hierarchical-DP mixtures.

Estimators accept `pseudo_count` (regularization), `prior` (a conjugate prior — `None` gives MLE),
and `keys` (tying statistics across model parts). HMM-family models take `use_numba=True` for
parallel Numba kernels (the first call pays a cached JIT cost).

### API naming

One stem per family (`<Stem>Distribution` / `Estimator` / `Sampler` / `Accumulator`, …) and
descriptive argument names. Legacy spellings stay as aliases (prefer `weights` over `w`,
`covariance` over `covar`, `max_iter` over `max_its`, …); passing both raises `TypeError`.

## Beyond fitting

- **Inference & analysis** — `pysp.utils.mcmc` (Metropolis–Hastings / HMC / VMP), `pysp.utils.em`
  (hard, annealed, ECM, Monte-Carlo, variational, online, restart EM), `pysp.utils.fisher`
  (Fisher-geometry views), and `pysp.utils.hvis` (model-based embeddings — t-SNE / UMAP).
- **Non-iid models** — `pysp.models` holds GP regression, neural regressors, random graphs,
  grammars, and knowledge graphs.

## Examples & notebooks

Worked tutorials live in the companion
[**pysparkplug-notebooks**](https://github.com/gmboquet/pysparkplug-notebooks) repo.

Runnable scripts ship in [examples/](examples/) — `examples_pysp/` (core), `examples_bayes/`
(Bayesian), `examples_spark/`, `examples_mp/`, and `examples_mpi/`:

```sh
cd examples/examples_pysp
python mixture_example.py
python hidden_markov_example.py
```

Every script is self-contained — it samples from a known model, then refits and recovers it (no
downloads). The `gallery_*_example.py` scripts tour the families in bulk; the rest focus on
individual models end to end.

### Running on Spark

PySpark 4.x needs a JVM (Java 17/21), and workers must use the driver's Python:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PYSPARK_PYTHON=/path/to/venv/bin/python
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
python examples/examples_spark/mixture_example.py
```

Estimation helpers detect RDD inputs automatically, so a model fit locally and one fit on a cluster
go through identical math.

## Tests

```sh
python -m pytest -m fast                          # quick correctness gate
python -m pytest -m "not optional and not benchmark"   # full local suite
```

Tests use `unittest.TestCase` internally with pytest markers / CI tiers (see
[`pysp/tests/README.md`](pysp/tests/README.md)). `base_dist_test.py` checks each distribution
end-to-end: sampler repeatability, `str`/`eval` round-trips, vectorized-vs-scalar densities, and
EM convergence.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Originally developed at Lawrence Livermore
National Laboratory (LLNL-CODE-844837).

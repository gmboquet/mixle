<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/pysparkplug/main/pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2500%2B-brightgreen)

**Composable density estimation for heterogeneous records.** A single observation can be a tuple of a
category, a real number, a count sequence, a vector, a set, or a tree — and pysparkplug fits a
probabilistic model to a dataset of them by expectation–maximization, locally on vectorized
NumPy/Numba or distributed across Spark, Dask, Ray, MPI, or Torch (GPU).

The unit of composition is the distribution. Leaves (Gaussian, categorical, Poisson, …) combine into
tuples; tuples become mixture components; mixtures become HMM emissions; and so on, to any depth. A
model and the estimator that fits it have the same shape, so what you can express you can fit.

- **Heterogeneous & composable** — model a record like `('a', -0.31, [5, 4, 6])` directly, no flattening.
- **Uniform contract** — every family exposes the same five pieces (distribution · sampler · estimator ·
  accumulator · encoder), so sampling, scoring, and estimation behave identically across all of them.
- **Local or distributed** — the same `optimize(...)` runs on one machine or a cluster; the backend is
  an argument, not a rewrite.
- **Frequentist or Bayesian** — MLE, MAP, closed-form conjugate posteriors, and variational
  (Dirichlet-process) mixtures, selected by a single `prior=`.
- **A probabilistic-programming surface** — [`pysp.ppl`](#probabilistic-programming-pyspppl): write the
  generative model in a few lines, then `.fit().sample().posterior()`.

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Core concepts](#core-concepts) ·
[Distribution catalog](#distribution-catalog) · [Probabilistic programming](#probabilistic-programming-pyspppl) ·
[Frequentist & Bayesian](#frequentist--bayesian) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Beyond fitting](#beyond-fitting) ·
[Companion packages](#companion-packages) · [Examples](#examples) · [Tests](#tests) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). The base install (numpy, scipy, mpmath, networkx, tqdm) covers
every distribution and local estimation. The package is published on PyPI as `pysp-learn`; the import
name is `pysp`:

```sh
pip install pysp-learn
```

Acceleration and scale-out are opt-in extras — `numba` (JIT estimation), `gmpy2` (fast big-integer
enumeration), `spark` / `dask` (distributed), `torch` (GPU/autograd), `pandas`, `umap`, or `all`:

```sh
pip install "pysp-learn[all]"
```

Without an extra, that path still works — numba-flagged kernels fall back to pure Python (correct,
slower) and the distributed backends are simply unavailable. For development:

```sh
git clone https://github.com/gmboquet/pysparkplug && cd pysparkplug
pip install -e ".[all]"
```

## Quickstart

A dataset where each record is a `(category, real, variable-length count sequence)` drawn from one of
two latent clusters. Define the generating model, sample a dataset, then recover the parameters with EM:

```python
import numpy as np
from pysp.stats import *
from pysp.inference import optimize

component = lambda mu, p: CompositeDistribution((
    CategoricalDistribution({'a': p, 'b': 1.0 - p}),
    GaussianDistribution(mu, 1.0),
    SequenceDistribution(PoissonDistribution(mu + 5.0),
                         len_dist=CategoricalDistribution({2: 0.5, 3: 0.5})),
))
truth = MixtureDistribution([component(0.0, 0.8), component(5.0, 0.2)], [0.6, 0.4])
data = truth.sampler(seed=1).sample(2000)
#   data[0] -> ('a', -0.59, [5, 7])      each record is a (label, real, count-sequence) tuple,
#   data[1] -> ('b',  4.54, [7, 7])      the sequence 2 or 3 long, all from one latent cluster

# The estimator mirrors the distribution's structure exactly.
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100, rng=np.random.RandomState(1))
print(model.w)   # ~ [0.6, 0.4] — the recovered mixing weights
```

The same model in the shorter [`pysp.ppl`](#probabilistic-programming-pyspppl) dialect is a few lines.

## Core concepts

Each family is five cooperating pieces:

| Piece             | Role                                                                       |
| ----------------- | ------------------------------------------------------------------------- |
| `...Distribution` | parameters + `log_density(x)` and vectorized `seq_log_density(encoded)`    |
| `...Sampler`      | draw observations — `dist.sampler(seed).sample(size)`                      |
| `...Estimator`    | declares the model to fit; closed-form M-step in `estimate()`             |
| `...Accumulator`  | sufficient statistics for the E-step, mergeable across data partitions     |
| `...DataEncoder`  | `seq_encode(data)` packs raw Python records into arrays for the fast path  |

`optimize(data, est)` (from `pysp.inference`) runs EM to convergence — vectorized NumPy/Numba locally,
or distributed by passing a `backend=` (see [Engines & orchestration](#engines--orchestration)).
Related entry points: `best_of` (random restarts), `StreamingEstimator` / `IncrementalEstimator`
(online EM), `fit_mle` / `fit_map` (autograd fitting with typed priors), and
`pysp.utils.automatic.get_estimator(data)` (infer an estimator from raw data).

**Library layout.** The families (objects) live in `pysp.stats`; the operations on them are grouped by
concern — `pysp.inference` (MLE / EM / MAP / conjugate / NUTS / VI / Fisher), `pysp.enumeration`
(rank / top-k / quantized unranking), and `pysp.ops` (quantize / condition / marginalize / project).
Drawing from a model is a method, not a concern: `pysp.stats.sample(model, n)` works for any samplable
object. `pysp.describe(x)` reports, in plain English, what any object supports.

## Distribution catalog

About 90 families live in `pysp.stats`, organized into subpackages (`leaf`, `multivariate`,
`combinator`, `sets`, `latent`, `graph`, `bayes`, `compute`) and re-exported at the top level. What
distinguishes pysparkplug is that the **combinators turn a heterogeneous record into one model** —
here is what a single datum looks like under each:

| Model | One observation |
| --- | --- |
| `GaussianDistribution` / `PoissonDistribution` | `-0.31` / `7` |
| `CategoricalDistribution` | `'b'` |
| `MultivariateGaussianDistribution` | `[1.2, -0.4, 0.8]` |
| `CompositeDistribution((Cat, Gaussian, Poisson))` | `('a', -0.31, 7)` |
| `RecordDistribution({...})` | `{'country': 'US', 'age': 41, 'spend': 12.5}` |
| `SequenceDistribution(Poisson)` | `[5, 4, 6]` (variable length) |
| `OptionalDistribution(Gaussian)` | `-0.31` or `None` (missing) |
| `BernoulliSetDistribution` | `{'sports', 'news'}` |
| `MixtureDistribution([...])` | a component's shape, with the cluster latent |
| `HiddenMarkovModelDistribution` | `[0.1, 5.0, 4.9, 0.2]` (states latent) |
| graph families | an adjacency matrix |

- **Scalar / univariate:** Gaussian, Student-t / Cauchy, Logistic, LogGaussian, Laplace, Uniform,
  Exponential, Gamma, Inverse Gamma, Inverse Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh,
  Pareto, Poisson, Bernoulli, Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet,
  categorical; multivariate / diagonal Gaussian, von Mises–Fisher, multivariate Student-t.
- **Combinators:** `CompositeDistribution` (tuples), `RecordDistribution` (named fields),
  `SequenceDistribution`, `OptionalDistribution` (missing data), `TransformDistribution`,
  `ConditionalDistribution`, `WeightedDistribution`.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains,
  hidden associations, IBP, Pitman-Yor processes, Bernoulli sets.
- **Permutations & graphs:** Mallows and Plackett-Luce rankings, matchings, spanning trees, random
  graphs (Erdős–Rényi, stochastic block, random dot-product), Spearman ranking.
- **Processes:** a general linear birth-death-sampling process (fossilized birth-death is the
  positive-`sampling_rate` case).
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet, SymmetricDirichlet)
  and variational Dirichlet-process / hierarchical-DP mixtures.

Estimators accept `pseudo_count` (regularization), `prior` (a conjugate prior; `None` is MLE), and
`keys` (tying statistics across model parts). HMM-family models take `use_numba=True` for parallel
Numba kernels (the first call pays a cached JIT cost).

**Naming.** One stem per family (`<Stem>Distribution` / `Estimator` / `Sampler` / `Accumulator`),
descriptive keyword arguments, and legacy spellings kept as aliases (prefer `weights` over `w`,
`covariance` over `covar`, `max_iter` over `max_its`); passing both raises `TypeError`.

## Probabilistic programming (`pysp.ppl`)

`pysp.ppl` is a concise dialect over the same distributions. **One rule:** any parameter slot is a
value, the token `free` (estimate it), or another distribution (a prior — random / hierarchical):

```python
from pysp.ppl import Normal, free

Normal(0.0, 1.0)             # fixed parameters
Normal(free, free)           # estimate the mean and standard deviation
Normal(Normal(0, 10), 1.0)   # a prior on the mean (hierarchical)
```

Build a model over your data, `.fit(...)`, and query it. A mixture over 1-D reals and an HMM over
real-valued sequences:

```python
from pysp.ppl import Normal, Mix, Markov, free

data = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1]            # reals from two clusters
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)
m.posterior(data)                                    # per-point responsibilities

sequences = [[0.1, 0.2, 5.1, 4.9], [4.8, 5.0], [0.0, 0.1, 0.2]]   # variable-length real sequences
Markov(Normal(free, free), states=2).fit(sequences)  # 2-state Gaussian HMM (k-means++ seeded)
```

A `free` coefficient times a `Field` makes the same surface a generalized linear model — `given=`
supplies the predictor columns:

```python
from pysp.ppl import Normal, Bernoulli, Field, free

#   y[i] ~ Normal(b0 + b1*x[i] + b2*z[i],  sd),  fit b0, b1, b2, sd
Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})

#   y[i] in {0, 1} ~ Bernoulli(sigmoid(b0 + b1*x[i]))   (logistic regression)
Bernoulli(free * Field("x") + free).fit(y, given={"x": x})
```

`how=` selects the inference route — `auto` (default) takes an exact path when one exists, otherwise
EM / gradient / sampling:

```python
Mix([Normal(free, free), Normal(free, free)]).fit(data, how="nuts")
Markov(Normal(free, free), states=2).fit(sequences, how="ensemble", chains=4, parallel=True)  # R̂ + ESS
# how = auto | conjugate | conjugate_mixture | em | map | vi | vmp | mcmc | hmc | nuts | ensemble
```

Constraints among named variables are plain comparisons (combine with `& | ~`); they shape both
inference and sampling:

```python
from pysp.ppl import Normal, Mix, constrain

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Mix([Normal(a, 1), Normal(b, 1)]).fit(data, constraints=a < b)   # ordered means break label-switching
constrain(2 * a - b >= 1).sample(100)                            # draw from the truncated joint
```

Model constructors: `Mix · Seq · Markov · LDA · MVN · DiagGaussian · LocalLevel · AR1 · Graph`;
`compare([model_a, model_b], data)` ranks fitted models best-first. For conjugate / exponential-family
/ mixture models `.fit(...)` returns the **exact** posterior in closed form; otherwise it falls back to
EM / gradient / sampling. The dialect is a thin layer — the `pysp.stats` classes underneath are
untouched.

## Frequentist & Bayesian

The prior is the only switch. No prior is maximum likelihood; a conjugate `prior=` makes the identical
machinery Bayesian:

```python
from pysp.inference.priors import NormalGammaPrior

GaussianEstimator()                          # MLE
GaussianEstimator(prior=NormalGammaPrior())  # closed-form conjugate posterior — same optimize() call
```

`optimize` / `fit` select the objective from the model (likelihood, MAP, or variational ELBO; override
with `objective=`). `BayesianStreamingEstimator` carries a posterior across batches, and
`pysp.stats.bayes` adds (hierarchical) Dirichlet-process mixtures. Gradient MAP with typed priors is
first-class:

```python
from pysp.engines import TorchEngine
from pysp.inference.gradient_fit import fit_map
from pysp.inference.priors import DirichletPrior, MixturePrior, NormalGammaPrior

enc = model.dist_to_encoder().seq_encode(data)
fitted, objective = fit_map(
    enc, model, engine=TorchEngine(device="cpu", dtype="float64"),
    priors=MixturePrior(
        components=[NormalGammaPrior(mu0=-2.0), NormalGammaPrior(mu0=2.0)],
        weights=DirichletPrior([2.0, 2.0]),
    ),
)
```

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
operations, device, and precision — so the same EM contract runs unchanged on NumPy, Numba, Torch (CPU
/ GPU / multi-device), or a symbolic backend.

```python
from pysp.engines import TorchEngine

optimize(data, est, engine=TorchEngine(device="cuda", dtype="float32"))   # GPU
optimize(data, est, engine=TorchEngine(mesh=mesh, shard="components"))    # multi-GPU (DTensor)
optimize(data, est, precision="auto")   # pick float32/float64 from the data; stats accumulate in float64
```

Scale out by changing the backend, not the model — local and distributed go through identical math:

```python
optimize(data, est)                                   # local NumPy / Numba
optimize(data, est, backend="mp", num_workers=8)      # multiprocessing
optimize(rdd,  est, backend="spark")                  # Spark
optimize(data, est, backend="dask", client=client)    # Dask
optimize(data, est, backend="mpi", comm=comm)         # MPI / torchrun
optimize(data, est, backend="ray")                    # Ray
optimize(data, est, backend="lightning")              # PyTorch Lightning
```

New frameworks register a factory (`register_encoded_data_backend`) rather than editing a dispatch.
The **planner** (`pysp.utils.parallel.planner`) turns a hardware budget into a memory-aware placement
(chunking, device assignment, Torch sharding) you compute once and reuse:

```python
from pysp.utils.parallel.planner import plan, Resources

placement = plan(data, model=model, estimator=est, resources=Resources.local(num_cpus=8))
optimize(data, est, placement=placement)
```

The `SymbolicEngine` runs a density through SymPy, so a model can emit its closed-form log-density as
LaTeX / SymPy / Sage:

```python
from pysp.engines import SYMBOLIC_ENGINE, to_latex

x = SYMBOLIC_ENGINE.symbol("x")
to_latex(GaussianDistribution(0.0, 1.0).backend_seq_log_density(x, SYMBOLIC_ENGINE))
# '- 0.5 x^{2} - 0.918938533204673'
```

## Enumeration & ranking

Discrete and structured models can **enumerate their support in descending-probability order** and
answer exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded.

```python
e = dist.enumerator()
e.top_k(5)            # the 5 most probable (value, log_prob), in order
e.top_p(0.95)         # smallest set covering 95% of the mass (the discrete nucleus)
e.rank(value)         # how many values are strictly more probable than `value`
e.seek(10_000)        # the ~10,000th most probable value, by structural count-DP
```

For decomposable families (`Composite` / `Record` / `Sequence` / `MarkovChain`), rank ↔ value is an
exact count dynamic program at any depth (`count_dp_rank`, `count_dp_seek`, `cumulative_probability`,
`count_dp_top_p`). For enormous or infinite supports, budget-bounded quantized indexes seek and unrank
over just the most-probable region (`count_budget_index`); with the `gmpy2` extra the underlying
big-integer count convolution uses GMP's FFT multiply. Non-decomposable families (mixtures, HMMs) have
provably hard exact marginal rank, so they return the Viterbi/tropical bound or a certified
Monte-Carlo estimate (`density_rank`, with a reported standard error) instead of a silent approximation.

**Continuous families** realize the same operations through the CDF and its inverse: every univariate
continuous leaf has an exact `cdf(x)` and `quantile(q)`; multivariate Gaussian and von Mises–Fisher
expose an exact probability-ordered cumulative plus `density_quantile(q)`. Any other samplable family
falls back to a Monte-Carlo estimate, so the four operations are reachable everywhere.

`pysp.enumeration` provides the shared machinery (bounded best-first union, quantization, count
convolution, k-best assignment / spanning-tree enumeration, HMM path enumeration).

## Beyond fitting

- **Inference & analysis** (`pysp.inference`): `mcmc` (Metropolis–Hastings / HMC / NUTS / VMP), `em`
  (hard, annealed, ECM, Monte-Carlo, variational, online, restart), `fisher` (Fisher-geometry views),
  and the `Posterior` algebra — `posterior(model, data, over="latent"|"params"|"predictive")` returns
  one uniform object you can `sample` / `mean` / `interval`. An engine-agnostic facade runs NUTS or
  ADVI on an arbitrary differentiable target (bring your own `value_and_grad`) with parallel chains
  (R̂ + pooled ESS).
- **Design & analysis of experiments** (`pysp.doe`): classical designs (`latin_hypercube`,
  `maximin_latin_hypercube`, `full_factorial`, `sobol_design`), GP Bayesian optimization
  (`minimize`, `expected_improvement`), and the analysis half — Sobol/Morris sensitivity, uncertainty
  propagation, and Kennedy-O'Hagan calibration.
- **Embeddings** (`pysp.utils.hvis`): model-based t-SNE / UMAP over per-record posteriors.
- **Supervised & non-iid models** (`pysp.models`): GP regression, neural regressors, random forests (a
  conditional `p(y | x)` leaf that fits the estimation framework), random graphs, grammars, and
  knowledge graphs.

## Companion packages

- [**pysparkplug-pde**](https://github.com/gmboquet/pysparkplug-pde) — PDE/ODE-constrained Bayesian
  inverse problems (diffusion, Navier–Stokes, wave/full-waveform inversion, FEM, level-set shape
  inference). It builds on pysp's field models and registers itself as a plugin: `import
  pysparkplug_pde` makes `PDE(operator).fit(snapshots)` and the forward solvers available, while pysp's
  own core stays free of PDE machinery.
- [**pysparkplug-notebooks**](https://github.com/gmboquet/pysparkplug-notebooks) — worked tutorials and
  master-level courses.

## Examples

Runnable scripts ship in [examples/](https://github.com/gmboquet/pysparkplug/tree/main/examples) —
`examples_pysp/` (core), `examples_bayes/`, `examples_spark/`, `examples_mp/`, `examples_mpi/`:

```sh
cd examples/examples_pysp
python mixture_example.py
python hidden_markov_example.py
```

Each script is self-contained: it samples from a known model, refits, and recovers it — no downloads.
The `gallery_*_example.py` scripts tour the families in bulk; the rest cover one model end to end.

**Running on Spark.** PySpark 4.x needs a JVM (Java 17/21), and workers must use the driver's Python:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PYSPARK_PYTHON=/path/to/venv/bin/python
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
python examples/examples_spark/mixture_example.py
```

RDD inputs are detected automatically, so a model fit locally and one fit on a cluster run identical math.

## Tests

```sh
python -m pytest                              # fast gate (parallel), ~25 s
python -m pytest -m "not optional and not benchmark"   # full suite incl. slow tests
```

The suite is `unittest.TestCase` collected under pytest with marker tiers (see
[`pysp/tests/README.md`](https://github.com/gmboquet/pysparkplug/blob/main/pysp/tests/README.md)).
`base_dist_test.py` exercises each family end to end: sampler repeatability, `str`/`eval` round-trips,
vectorized-vs-scalar density agreement, and EM convergence.

## License

MIT — see [LICENSE](https://github.com/gmboquet/pysparkplug/blob/main/LICENSE) and
[NOTICE](https://github.com/gmboquet/pysparkplug/blob/main/NOTICE). Originally developed at Lawrence
Livermore National Laboratory (LLNL-CODE-844837).

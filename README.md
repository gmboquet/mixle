<p align="left">
  <img src="https://raw.githubusercontent.com/gmboquet/pysparkplug/main/pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-2500%2B-brightgreen)

**Composable density estimation for heterogeneous records.** A single observation can be a tuple of a
category, a real, a count sequence, a vector, a set, or a tree — and pysparkplug fits a probabilistic
model to a dataset of them by expectation–maximization, locally on vectorized NumPy/Numba or
distributed across Spark, Dask, Ray, MPI, or Torch (GPU).

The unit of composition is the distribution: leaves (Gaussian, categorical, Poisson, …) combine into
tuples, tuples become mixture components, mixtures become HMM emissions, to any depth. A model and the
estimator that fits it have the same shape — so what you can express, you can fit.

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Core concepts](#core-concepts) ·
[Distribution catalog](#distribution-catalog) · [Probabilistic programming](#probabilistic-programming-pyspppl) ·
[Frequentist & Bayesian](#frequentist--bayesian) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Beyond fitting](#beyond-fitting) ·
[Examples](#examples) · [Tests](#tests) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12). On PyPI as `pysp-learn`; the import name is `pysp`.

```sh
pip install pysp-learn          # base (numpy, scipy, mpmath): every distribution + local EM
pip install "pysp-learn[all]"   # acceleration, scale-out, and connectors
```

The base install fits every distribution locally. Acceleration and scale-out are opt-in extras:

| Extra | Adds |
| --- | --- |
| `numba` | JIT-compiled hot paths (falls back to pure NumPy when absent) |
| `torch` | GPU / autograd engine |
| `spark` · `dask` · `mpi` | distributed estimation backends |
| `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `data` | data-source connectors |
| `gmpy2` | GMP-FFT big-integer multiply for count-DP ranking |
| `umap` | model-based UMAP embeddings |
| `sympy` · `sage` | symbolic / closed-form export |
| `grammar` | graph-grammar models (networkx) |

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

Each record here is a `(category, real, variable-length count sequence)`. Fit a two-component mixture
straight from a list of records:

```python
from pysp.stats import *
from pysp.inference import optimize

data = [
    ('a', -0.4, [5, 7]),       ('b', 4.9, [11, 9]),
    ('a',  0.2, [6, 5, 4]),    ('b', 5.3, [10, 12, 11]),
    ('a', -1.1, [4, 6]),       ('b', 4.5, [9, 10]),
    ('a',  0.7, [5, 5]),       ('b', 5.1, [12, 8]),
    ('a', -0.2, [7, 6, 5]),    ('b', 4.7, [9, 11]),
]

# The estimator mirrors the distribution's structure exactly.
est = MixtureEstimator([CompositeEstimator((
    CategoricalEstimator(),
    GaussianEstimator(),
    SequenceEstimator(PoissonEstimator(), len_estimator=CategoricalEstimator()),
))] * 2)

model = optimize(data, est, max_its=100)
model.sampler(seed=0).sample(3)   # draw new records from the fitted model
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
| `...DataEncoder`  | packs raw Python records into arrays for the fast path                     |

`optimize(data, est)` (in `pysp.inference`) runs EM to convergence — vectorized locally, or
distributed via `backend=`. Related entry points:

- `best_of` — multi-restart EM
- `StreamingEstimator` — online EM
- `fit_mle` / `fit_map` — autograd fitting with typed priors
- `pysp.utils.automatic.get_estimator(data)` — infer an estimator from raw data

Families live in `pysp.stats`; operations on them are grouped by concern:

- `pysp.inference` — fit: MLE / EM / MAP / conjugate / NUTS / VI / Fisher
- `pysp.enumeration` — rank / top-k / unranking
- `pysp.ops` — quantize / condition / marginalize / project
- `pysp.describe(x)` — report what any object supports

Drawing is a method, not a concern: `dist.sampler(seed).sample(n)`.

## Distribution catalog

About 90 families in `pysp.stats`. The distinguishing feature: the **combinators model a whole
heterogeneous record as one distribution**. One observation under each:

| Model | One observation |
| --- | --- |
| `GaussianDistribution` / `PoissonDistribution` / `CategoricalDistribution` | `-0.31` / `7` / `'b'` |
| `MultivariateGaussianDistribution` | `[1.2, -0.4, 0.8]` |
| `CompositeDistribution((Cat, Gaussian, Poisson))` | `('a', -0.31, 7)` |
| `RecordDistribution({...})` | `{'country': 'US', 'age': 41, 'spend': 12.5}` |
| `SequenceDistribution(Poisson)` | `[5, 4, 6]` (variable length) |
| `OptionalDistribution(Gaussian)` | `-0.31` or `None` |
| `MixtureDistribution([...])` / `HiddenMarkovModelDistribution` | a component's shape, with the cluster / state latent |

- **Univariate:** Gaussian, Student-t/Cauchy, Logistic, LogGaussian, Laplace, Uniform, Exponential,
  Gamma, Inverse Gamma/Gaussian, Half-Normal, Gumbel, Beta, Weibull, Rayleigh, Pareto, Poisson,
  Bernoulli, Geometric, Binomial, Negative Binomial, Log-Series, von Mises, Dirichlet, categorical;
  multivariate/diagonal Gaussian, von Mises–Fisher, multivariate Student-t.
- **Combinators:** Composite (tuples), Record (named fields), Sequence, Optional (missing data),
  Transform, Conditional, Weighted.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, probabilistic PCA, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains,
  hidden associations, IBP, Pitman-Yor processes, Bernoulli sets.
- **Permutations & graphs:** Mallows / Plackett-Luce, matchings, spanning trees, random graphs
  (Erdős–Rényi, stochastic block, random dot-product), Spearman ranking, and graph grammars over
  networks (vertex-replacement / NLC and hyperedge-replacement) — `log_density` is the marginal
  likelihood, computed by parsing the graph back to the start symbol.
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet, SymmetricDirichlet)
  and variational Dirichlet-process / hierarchical-DP mixtures.

Estimator knobs (every family): `pseudo_count` (regularization) · `prior=` (conjugate; `None` is MLE) ·
`keys` (tie statistics across parts). One stem per family
(`<Stem>Distribution` / `Estimator` / …); legacy spellings remain as aliases.

## Probabilistic programming (`pysp.ppl`)

A concise dialect over the same distributions. **One rule:** any parameter slot is a value, the token
`free` (estimate it), or another distribution (a prior).

```python
from pysp.ppl import Normal, Mix, Markov, Field, free

Normal(0.0, 1.0)              # fixed parameters
Normal(free, free)            # estimate the mean and standard deviation
Normal(Normal(0, 10), 1.0)    # a prior on the mean (hierarchical)

data = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1]                          # reals from two clusters
m = Mix([Normal(free, free), Normal(free, free)]).fit(data)
m.posterior(data)                                                 # per-point responsibilities

seqs = [[0.1, 5.1, 4.9], [4.8, 5.0], [0.0, 0.2]]                  # variable-length real sequences
Markov(Normal(free, free), states=2).fit(seqs)                    # 2-state Gaussian HMM

#   y[i] ~ Normal(b0 + b1*x[i] + b2*z[i], sd)   — a linear model
Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})

a, b = Normal(0, 10, name="a"), Normal(0, 10, name="b")
Mix([Normal(a, 1), Normal(b, 1)]).fit(data, constraints=a < b)    # ordered means break label-switching
```

- **`how=`** selects the route: `auto` takes an exact path when one exists, else
  `conjugate | em | map | vi | vmp | mcmc | hmc | nuts | ensemble`.
- **Constraints** among named variables are plain comparisons (combine with `& | ~`) and shape both
  inference and sampling.
- **Closed form:** for conjugate / exponential-family / mixture models, `.fit(...)` returns the exact
  posterior.
- **Constructors:** `Mix · Seq · Markov · LDA · MVN · DiagGaussian · LocalLevel · AR1 · Graph`;
  `compare([m1, m2], data)` ranks fitted models.

The dialect is thin — the `pysp.stats` classes underneath are untouched.

## Frequentist & Bayesian

The prior is the only switch — no prior is MLE; a conjugate `prior=` makes the same machinery Bayesian:

```python
from pysp.inference.priors import NormalGammaPrior

GaussianEstimator()                          # MLE
GaussianEstimator(prior=NormalGammaPrior())  # closed-form conjugate posterior — same optimize() call
```

- `optimize` / `fit` pick the objective from the model — likelihood, MAP, or variational ELBO.
- `BayesianStreamingEstimator` carries a posterior across batches; `pysp.stats.bayes` adds
  (hierarchical) Dirichlet-process mixtures.
- Gradient MAP with typed priors: `pysp.inference.gradient_fit.fit_map`
  (`NormalGammaPrior` / `DirichletPrior` / `MixturePrior`).
- **Honest densities:** `supports(x, ExactDensity)` / `describe(x)` flag when a model's `log_density`
  is a variational bound (e.g. LDA's per-document ELBO) rather than the exact `log p(x)`.

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
ops, device, and precision — so **scale-out is a backend argument, not a rewrite**:

```python
from pysp.engines import TorchEngine

optimize(data, est, engine=TorchEngine(device="cuda", dtype="float32"))   # GPU
optimize(data, est, precision="auto")                                     # stats still accumulate in float64
optimize(rdd,  est, backend="spark")                                      # also: mp · dask · mpi · ray · lightning
```

- The same EM contract runs unchanged on NumPy, Numba, Torch, or a symbolic backend.
- New frameworks register a factory (`register_encoded_data_backend`) — no dispatch to edit.
- The planner (`pysp.utils.parallel.planner`) turns a hardware budget into a memory-aware placement
  (chunking, device assignment, Torch sharding) you compute once and reuse.
- The `SymbolicEngine` runs a density through SymPy, so a model can emit its closed-form log-density
  as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded:

```python
e = dist.enumerator()
e.top_k(5)        # the 5 most probable (value, log_prob)
e.top_p(0.95)     # smallest set covering 95% of the mass (the nucleus)
e.rank(value)     # how many values are strictly more probable than `value`
e.seek(10_000)    # the ~10,000th most probable value, by structural count-DP
```

- **Decomposable families** (Composite / Record / Sequence / MarkovChain): rank ↔ value is an exact
  count-DP at any depth (`count_dp_rank`, `count_dp_seek`); budget-bounded quantized indexes
  (`count_budget_index`) seek the most-probable region of an infinite support (the `gmpy2` extra uses
  GMP's FFT multiply for the big-integer convolution).
- **Non-decomposable families** (mixtures, HMMs): exact marginal rank is provably hard, so they return
  the Viterbi bound or a certified Monte-Carlo estimate (`density_rank`, with a standard error) — never
  a silent approximation.
- **Continuous families** realize the same operations through `cdf(x)` / `quantile(q)`.

## Beyond fitting

- **Inference** (`pysp.inference`): `mcmc` (MH / HMC / NUTS / VMP), `em` (hard, annealed, ECM,
  Monte-Carlo, variational, online, restart), `fisher` (geometry views), and the `Posterior` algebra —
  `posterior(model, data, over="latent"|"params"|"predictive")` returns one object you `sample` /
  `mean` / `interval`. An engine-agnostic facade runs NUTS/ADVI on any differentiable target with
  parallel chains (R̂ + pooled ESS).
- **Design & analysis of experiments** (`pysp.doe`): space-filling designs, GP Bayesian optimization,
  and the analysis half — Sobol/Morris sensitivity, uncertainty propagation, Kennedy-O'Hagan calibration.
- **Embeddings** (`pysp.utils.hvis`): model-based t-SNE / UMAP over per-record posteriors.
- **Supervised & non-iid models** (`pysp.models`): GP regression, neural regressors, random forests
  (a conditional `p(y | x)` leaf), random graphs, grammars, knowledge graphs.
- **MLOps** (`pysp.inference`): reproducible model artifacts (`fit_with_provenance` → a `ModelHeader`
  with config, data hash, convergence, timing, resources, env), drift detection + `ModelMonitor`
  (retrain-and-swap), and a versioned `ModelRegistry` + `ModelService` (scoring + activity logging).
  A container / Kubernetes serving layer lives in the separate
  [pysparkplug-deploy](https://github.com/gmboquet/pysparkplug-deploy) package.

## Examples

Self-contained scripts in [examples/examples_pysp/](https://github.com/gmboquet/pysparkplug/tree/main/examples/examples_pysp)
— each samples from a known model, refits, and recovers it (no downloads):

```sh
cd examples/examples_pysp
python gallery_univariate_example.py    # tour the scalar families (also gallery_{multivariate,combinators,…})
python gallery_structured_example.py    # mixtures / HMMs / LDA / latent-variable models
python ppl_example.py                   # the equation-style pysp.ppl surface
python production_example.py            # provenance, registry, serving, drift, checkpoints
python scaling_example.py               # the same fit distributed by backend= (local / mp / mpi / spark)
```

**Distributed backends** (see `scaling_example.py`): `local` and `mp` run out of the box; `mpi` and Spark
need a launcher. Spark also needs a JVM (Java 17/21) with workers on the driver's Python:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PYSPARK_PYTHON=/path/to/venv/bin/python PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON
```

## Tests

```sh
python -m pytest                                       # fast gate (parallel), ~25 s
python -m pytest -m "not optional and not benchmark"   # full suite incl. slow tests
```

`base_dist_test.py` exercises each family end to end: sampler repeatability, `str`/`eval` round-trips,
vectorized-vs-scalar density agreement, EM convergence. See
[`pysp/tests/README.md`](https://github.com/gmboquet/pysparkplug/blob/main/pysp/tests/README.md).

## License

MIT — see [LICENSE](https://github.com/gmboquet/pysparkplug/blob/main/LICENSE). Originally developed at
Lawrence Livermore National Laboratory (LLNL-CODE-844837).

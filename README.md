<p align="left">
  <img src="pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-1300%2B-brightgreen)

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

`pysp.ppl` is a concise, optional dialect over the very same distributions. **One rule:** any
parameter slot can be a concrete value, the token `free` (estimate it), or another distribution
(make it random / hierarchical). Build a model, `.fit(data)`, then query it with `.sample`,
`.log_prob`, `.posterior`, and `.params`.

```python
from pysp.ppl import Normal, Mix, Markov, free

Normal(free, free).fit(data)                                   # estimate mean & sd

m = Mix([Normal(free, free), Normal(free, free)]).fit(data)    # 2-component Gaussian mixture
m.posterior(data)                                              # responsibilities

Markov(Normal(free, free), states=2).fit(sequences)           # 2-state Gaussian HMM (k-means++ seeded)
```

A `free` coefficient times a `Field` turns the same surface into generalized linear models:

```python
from pysp.ppl import Normal, Bernoulli, Poisson, Field, free

Normal(free * Field("x") + free * Field("z") + free, free).fit(y, given={"x": x, "z": z})  # linear
Bernoulli(free * Field("x") + free).fit(y, given={"x": x})                                 # logistic
Poisson(free * Field("x") + free).fit(y, given={"x": x})                                   # Poisson
```

`.fit(..., how=...)` picks the inference engine — `auto` (default), `em`, `map`, `conjugate`,
`hierarchical`, `vi`, `vmp`, `mcmc`, or `hmc`. Constructors cover the scalar families plus `Mix`,
`Seq`, `Markov`, `LDA`, `MVN`, `DiagGaussian`, `LocalLevel`, `AR1`, and `Graph` (a VMP factor graph
for conjugate-Gaussian DAGs); `compare(...)` ranks fitted models. The `pysp.stats` classes are
untouched — this is a thin surface over them.

## Core concepts

Each model family implements five cooperating pieces:

| Piece             | Role                                                                        |
| ----------------- | -------------------------------------------------------------------------- |
| `...Distribution` | Parameters + `log_density(x)` / vectorized `seq_log_density(enc)`          |
| `...Sampler`      | Draw samples (`dist.sampler(seed).sample(size)`)                           |
| `...Estimator`    | Specifies the model to fit; closed-form M-step via `estimate()`           |
| `...Accumulator`  | Collects sufficient statistics (E-step), mergeable across partitions       |
| `...DataEncoder`  | `seq_encode(data)` flattens raw Python data into NumPy for the fast path   |

Driver functions in `pysp.utils.estimation` tie these together — `optimize` runs EM to
convergence, and the same call scales out by swapping the data argument or the back-end:

```python
optimize(data, est, max_its=100)                         # local, vectorized NumPy/Numba
optimize(rdd,  est, backend='spark')                     # Spark RDDs
optimize(data, est, backend='dask', client=client)       # Dask workers
optimize(data, est, engine=TorchEngine(...))             # GPU / autograd
```

Also available: `best_of` (random restarts), `StreamingEstimator` / `IncrementalEstimator`
(online EM), `fit_mle` / `fit_map` (autograd fitting with typed priors), `RecordDistribution` /
`field(...)` (named dict/DataFrame observations), and `pysp.utils.automatic.get_estimator(data)`
(infer an estimator straight from raw data).

## Frequentist & Bayesian — one switch

The prior is the single switch. With no prior an estimator is plain maximum likelihood; attach a
conjugate `prior=` and the same machinery does the Bayesian thing:

- `estimate()` performs the closed-form **conjugate posterior** update and exposes
  `expected_log_density` (the variational E-step term);
- `optimize` / `fit` **auto-select the objective** from the model — maximum likelihood, MAP
  (penalized), or the variational ELBO — and you can force it with `objective='mle'|'map'|'vb'`;
- `fit(...)` is the posterior-returning counterpart of `optimize(...)`;
- `BayesianStreamingEstimator` carries posteriors across batches (posterior-carry / forgetting);
- `pysp.stats.dpm` / `pysp.stats.hdpm` provide (hierarchical) Dirichlet-process mixtures.

Conjugate priors (`NormalGammaDistribution`, `NormalWishartDistribution`, `DirichletDistribution`,
…) and gradient MAP fitting with typed priors are first-class:

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

## Distribution catalog

~75 composable families live in `pysp.stats`, grouped into subpackages (`leaf`, `multivariate`,
`combinator`, `sets`, `latent`, `graph`, `bayes`, `compute`) but all re-exported at the top level —
`from pysp.stats import GaussianDistribution` works regardless of where the file lives.

- **Scalar / basic:** Gaussian, Student-t / Cauchy, Logistic, LogGaussian, Laplace, Uniform,
  Exponential, Gamma, Beta, Weibull, Rayleigh, Pareto, Poisson, Bernoulli, Geometric, Binomial,
  Negative Binomial, von Mises–Fisher, multivariate / diagonal Gaussian, Dirichlet, categorical.
- **Combinators:** `CompositeDistribution` (tuples), `SequenceDistribution`, `OptionalDistribution`
  (missing data), `TransformDistribution`, `ConditionalDistribution`, `WeightedDistribution`.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains, hidden
  associations, IBP, random graphs (Erdős–Rényi, stochastic block), Spearman ranking, Bernoulli sets.
- **Bayesian:** conjugate priors (NormalGamma, NormalWishart, MvnGamma, Dirichlet, SymmetricDirichlet)
  and variational Dirichlet-process / hierarchical-DP mixtures.

Estimators accept `pseudo_count` (regularization), `prior` (a conjugate prior — `None` gives MLE),
and `keys` (tying statistics across model parts). HMM-family models take `use_numba=True` for
parallel Numba kernels (the first call pays a cached JIT cost).

### API naming

The API uses one stem per family (`<Stem>Distribution`, `<Stem>Estimator`, `<Stem>Sampler`,
`<Stem>Accumulator`, …) and descriptive argument names. Legacy and preferred spellings both work
(old names kept as aliases); new code should prefer `weights` over `w`, `prob_map` over `pmap`,
`prob_vec` over `p_vec`, `covariance` over `covar`, `num_values` over `num_vals`, and `max_iter`
over `max_its`. Passing both spellings raises `TypeError`.

## Beyond fitting

- **Engines** — distributions own the likelihood/statistic math; `pysp.engines`
  (`TorchEngine`, `NumbaKernelFactory`, `SymbolicEngine`) supplies array ops and device/precision
  policy. The same EM contract runs through any engine.
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

Every script is self-contained: it samples its own random data from a known model, then refits and
recovers it — no external corpora or downloads. The `gallery_*_example.py` scripts tour the
distribution families in bulk; the rest focus on individual models end to end.

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

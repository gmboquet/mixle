<p align="left">
  <img src="pysparkplug_logo.png" alt="pysparkplug" width="120"/>
</p>

# pysparkplug

Composable, distributed density estimation for messy, mixed-type records — tuples of
categories, counts, reals, sets, sequences, and trees. Specify a probabilistic model in a few
lines and fit it with EM, locally (vectorized NumPy + Numba) or at scale on Spark, Dask, or Torch.

Every distribution is **composable** and shares one five-part interface, so estimation, sampling,
and vectorized evaluation work uniformly at any depth of nesting: a Gaussian and a categorical
compose into a tuple model, tuple models become mixture components, mixtures become HMM emissions.

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

## Core concepts

Each model family implements five cooperating pieces:

| Piece             | Role                                                                          |
| ----------------- | ---------------------------------------------------------------------------- |
| `...Distribution` | Parameters + `log_density(x)` / vectorized `seq_log_density(enc)`            |
| `...Sampler`      | Draw samples (`dist.sampler(seed).sample(size)`)                             |
| `...Estimator`    | Specifies the model to fit; closed-form M-step via `estimate()`             |
| `...Accumulator`  | Collects sufficient statistics (E-step), mergeable across partitions         |
| `...DataEncoder`  | `seq_encode(data)` flattens raw Python data into NumPy for the fast path     |

Driver functions in `pysp.stats` / `pysp.utils.estimation` tie these together — `optimize` runs
EM to convergence, and the same call scales out by swapping the data argument or the back-end:

```python
optimize(data, est, max_its=100)                         # local, vectorized NumPy/Numba
optimize(rdd,  est, backend='spark')                     # Spark RDDs
optimize(data, est, backend='dask', client=client)       # Dask workers
optimize(data, est, engine=TorchEngine(...))             # GPU / autograd
```

Also available: `best_of` (random restarts), `StreamingEstimator` / `IncrementalEstimator`
(online EM), `fit_mle` / `fit_map` (autograd fitting with typed priors), `RecordDistribution` /
`field(...)` (named dict/DataFrame observations), and `pysp.utils.automatic.get_estimator(data)`
(infer an estimator from raw data).

## Compute engines

Distributions own the likelihood/statistic math; **compute engines** (`pysp.engines.TorchEngine`,
`NumbaKernelFactory`, `SymbolicEngine`) provide array ops and device/precision policy. The same EM
contract runs through any engine, and MAP fitting accepts typed priors:

```python
from pysp.engines import TorchEngine
from pysp.utils.estimation import fit_map
from pysp.utils.priors import DirichletPrior, MixturePrior, NormalGammaPrior

enc = model.dist_to_encoder().seq_encode(data)
fitted, objective = fit_map(enc, model, engine=TorchEngine(device="cpu", dtype="float64"),
                            priors=MixturePrior(
                                components=[NormalGammaPrior(mu0=-2.0), NormalGammaPrior(mu0=2.0)],
                                weights=DirichletPrior([2.0, 2.0])))
```

`pysp.utils.objectives` adds custom differentiable objectives, `pysp.utils.em` provides EM
variants (hard, annealed, ECM, Monte Carlo, variational, online, restart), `pysp.utils.mcmc`
adds Metropolis-Hastings / HMC samplers, and `pysp.models` holds non-iid models (GP regression,
neural regressors, random graphs, grammars, knowledge graphs).

## Distribution catalog

~60 composable families in `pysp.stats`:

- **Scalar/basic:** Gaussian, StudentT/Cauchy, Logistic, LogGaussian, Laplace, Uniform,
  Exponential, Gamma, Beta, Weibull, Rayleigh, Pareto, Poisson, Bernoulli, Geometric, Binomial,
  Negative Binomial, von Mises-Fisher, multivariate/diagonal Gaussian, Dirichlet, categorical.
- **Combinators:** `CompositeDistribution` (tuples), `SequenceDistribution`, `OptionalDistribution`
  (missing data), `TransformDistribution`, `ConditionalDistribution`, `WeightedDistribution`.
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA,
  PLSI, HMMs (standard, segmental, lookback, tree, quantized), PCFGs, Markov chains, hidden
  associations, random graphs (Erdős–Rényi, stochastic block), Spearman ranking, Bernoulli sets.
- **Bayesian (`pysp.bstats`):** conjugate/variational counterparts, posterior-carry streaming, and
  Dirichlet-process mixtures.

Estimators accept `pseudo_count` (regularization) and `keys` (tying statistics across model parts).
HMM-family models take `use_numba=True` for parallel Numba kernels (first call pays a cached JIT).

### API naming

The API is converging on one stem per family (`<Stem>Distribution`, `<Stem>Estimator`,
`<Stem>Sampler`, `<Stem>Accumulator`, …) and descriptive argument names. Legacy and preferred
spellings both work (old names kept as aliases); new code should prefer `weights` over `w`,
`prob_map` over `pmap`, `prob_vec` over `p_vec`, `covariance` over `covar`, `num_values` over
`num_vals`, and `max_iter` over `max_its`. Passing both spellings raises `TypeError`.

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

Most examples generate their own data; `set_example.py` and `wikipedia_*` need external corpora.

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
python -m pytest -m fast
python -m pytest -m "not optional and not benchmark"
```

Tests use `unittest.TestCase` internally with pytest markers/CI tiers (see
[`pysp/tests/README.md`](pysp/tests/README.md)). `base_dist_test.py` checks each distribution
end-to-end: sampler repeatability, `str`/`eval` round-trips, vectorized-vs-scalar densities, and
EM convergence.

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Originally developed at Lawrence Livermore
National Laboratory (LLNL-CODE-844837).

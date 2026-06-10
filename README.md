<p align="center">
  <img src="sparkplug_logo1.png" alt="pysparkplug" width="320"/>
</p>

# pysparkplug

**pysparkplug** is a Python package for distributed heterogeneous density estimation. With a few lines of code you can specify a complex probabilistic model over messy, variable-length, mixed-type records — tuples of categories, counts, reals, sets, sequences, trees — and fit it with EM, locally (vectorized NumPy + Numba) or at scale on Apache Spark.

The core idea: every distribution is **composable**. A Gaussian and a categorical compose into a tuple model; tuple models become mixture components; mixtures become HMM emissions; everything keeps the same five-part interface, so estimation, sampling, and vectorized evaluation work uniformly at any level of nesting.

## Installation

Python 3.10+ (developed against 3.12). NumPy 2.x, SciPy, Numba, and PySpark 4.x are supported.

```sh
git clone https://github.com/gmboquet/pysparkplug.git
cd pysparkplug
pip install -e .
```

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

| Piece | Role |
|---|---|
| `...Distribution` | Parameters + `log_density(x)` / vectorized `seq_log_density(enc)` |
| `...Sampler` | Draw samples (`dist.sampler(seed).sample(size)`) |
| `...Estimator` | Specifies the model to fit; builds accumulators; closed-form M-step via `estimate()` |
| `...Accumulator` | Collects sufficient statistics (E-step), mergeable across data partitions |
| `...DataEncoder` | `seq_encode(data)` flattens raw Python data into NumPy arrays for the vectorized path |

Driver functions in `pysp.stats` and `pysp.utils.estimation` tie these together:

- `seq_encode(data, model=...)` — encode once, reuse across EM iterations
- `seq_initialize` / `seq_estimate` / `seq_log_density_sum` — one vectorized EM step at a time
- `optimize(data, est, max_its, ...)` — EM to convergence
- `best_of(...)` — EM from multiple random restarts, keeping the best validation log-likelihood
- `pysp.utils.automatic.get_estimator(data)` — infer a reasonable estimator from raw data automatically

All of these accept either an in-memory list **or a Spark RDD** — the same code scales out by swapping the data argument.

## Distribution catalog

Around 60 composable families in `pysp.stats`, including:

- **Scalar/basic:** Gaussian, LogGaussian, Exponential, Gamma, Poisson, Geometric, Binomial, von Mises–Fisher, multivariate/diagonal Gaussian, Dirichlet, categorical & integer-categorical
- **Combinators:** `CompositeDistribution` (tuples), `SequenceDistribution` (variable-length i.i.d. with length model), `OptionalDistribution` (missing data), `IgnoredDistribution`, `ConditionalDistribution`, `WeightedDistribution`
- **Latent structure:** mixtures (plain, heterogeneous, hierarchical, joint, semi-supervised), LDA, PLSI, hidden Markov models (standard, lookback, tree-structured), Markov chains, hidden associations, Spearman ranking, Bernoulli set/edit-set models
- **Bayesian (`pysp.bstats`):** conjugate-prior/variational counterparts, Dirichlet-process mixtures (`bexamples/` shows DPM auto-modeling)

Estimators accept `pseudo_count` for regularization and `keys` for tying sufficient statistics across model parts. Several models (HMMs, tree HMM, PLSI) take `use_numba=True` to switch to parallel Numba kernels; the first call pays a JIT compile that is cached afterwards.

## Examples

Local examples live in [pysp/examples/](pysp/examples/) and run from that directory:

```sh
cd pysp/examples
python mixture_example.py
python hidden_markov_example.py
```

A few examples need datasets that are not shipped: `set_example.py` (NIPS submissions) and the two `wikipedia_*` examples (a Wikipedia corpus + stop-word list). Everything else generates its own data — e.g. `hmm_numba_example.py` fits an HMM to generated text-like sequences and compares Numba vs. pure-NumPy fitting. `lda_example.py` runs to a tight tolerance and takes a while.

## Running on Spark

Spark examples are in [pysp/examples_spark/](pysp/examples_spark/). PySpark 4.x needs a JVM (Java 17 or 21) and the workers must use the same Python as the driver:

```sh
export JAVA_HOME=$(/usr/libexec/java_home -v 17)   # or your JDK 17/21 path
export PYSPARK_PYTHON=/path/to/your/venv/bin/python
export PYSPARK_DRIVER_PYTHON=$PYSPARK_PYTHON

python pysp/examples_spark/mixture_example.py
```

The estimation helpers detect RDD inputs automatically: sampling (`pysp.stats.rdd_sampler`), initialization, encoding, and each EM step run per-partition, and sufficient statistics are merged on the driver — so a model fit locally and one fit on a cluster go through identical math.

## Tests

```sh
python -m unittest discover pysp/tests
```

`base_dist_test.py` checks each enabled distribution end-to-end: sampler repeatability, `str`/`eval` round-trips, vectorized-vs-scalar log densities, and that EM-to-convergence improves (in KL) with more data.

## License

pysparkplug is distributed under the terms of the MIT license. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Originally developed at Lawrence Livermore National Laboratory (LLNL-CODE-844837).

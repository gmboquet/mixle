<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gmboquet/mixle/main/assets/mixle_logo_dark.png"/>
    <img src="https://raw.githubusercontent.com/gmboquet/mixle/main/assets/mixle_logo.png" alt="mixle" width="480"/>
  </picture>
</p>

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-5000%2B-brightgreen)
[![docs](https://img.shields.io/badge/docs-gmboquet.github.io%2Fmixle-blue)](https://gmboquet.github.io/mixle/)

**mixle is a Python library for specifying, training, deploying, and maintaining models of heterogeneous
data.** Hand it raw data and it selects and fits a model; hand it a structure and it fits that. Underneath is
a full probabilistic-modeling stack — around 90 distributions, mixtures and hidden Markov models, automatic
model selection, Bayesian inference from EM to NUTS, design-of-experiments optimization, and calibrated,
monitored deployment — held together by one idea: a classical distribution, a neural network, and a
latent-variable model are the same kind of object, so they compose freely and a single `optimize(...)` call
fits the whole thing.

Fitting follows from the structure, not a flag — closed form where a part has one, gradient descent for a
neural leaf, EM for latent variables, all in one loop. The same model runs on the built-in engines (NumPy,
Numba, GPU) and distributes over Spark, Dask, Ray, or MPI by switching one argument. It models what
you actually have — numbers, text, categories, mixed and missing values, directional and angular data,
rankings, graphs — all the same way.

**Lab-grade AI, without the lab.** Three reasons people reach for it:

- **Less code.** No training loops, no batching or convergence boilerplate, no glue — point `optimize` at
  your data or your PyTorch module and it does the heavy lifting.
- **Lower cost.** Distill a slow, expensive model — a frontier LLM, an API, a rule — into a tiny local one
  that answers the easy cases itself and escalates only the hard ones.
- **Honest uncertainty.** It is calibrated to know when it is unsure and defer rather than guess — so you
  can gate on its confidence instead of trusting every answer.

**Docs:** [gmboquet.github.io/mixle](https://gmboquet.github.io/mixle/) · **Release notes:**
[CHANGELOG.md](CHANGELOG.md)

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Engines & orchestration](#engines--orchestration) ·
[Enumeration & ranking](#enumeration--ranking) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Package highlights](#package-highlights) · [Companion projects](#companion-projects) · [Examples](#examples) ·
[Tests](#tests) · [Maintainers & contributors](#maintainers--contributors) · [License](#license)

## Installation

Python 3.10+ (developed on 3.12), on PyPI as `mixle`. CI tests Linux x86_64; macOS (incl. Apple Silicon)
is the day-to-day dev platform and works in practice but isn't CI-gated; Windows is untested.

```sh
pip install mixle          # base (numpy, scipy, mpmath): every distribution, fit locally
pip install "mixle[all]"   # + numba, torch, the distributed backends, and core data connectors
```

Everything past the base is opt-in — install any subset, e.g. `pip install "mixle[torch,spark]"`:

- **Acceleration** — `numba` (JIT hot paths, falls back to NumPy), `torch` (GPU / autograd), `jax` (JAX engine + NUTS)
- **Scale-out** — `spark` · `dask` · `ray` · `lightning` · `mpi`
- **Data sources** — `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `arrays`
- **Other** — `gmpy2` (fast exact ranking) · `umap` (embeddings) · `sympy` · `sage` (symbolic export) · `grammar` (graph grammars)

`[all]` covers `numba`, `torch`, the scale-out backends, and `pandas`/`arrow`/`sql`; `jax`, `gmpy2`,
`sympy`/`sage`, and `mongo`/`hadoop`/`arrays` install separately.

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

**Hand it data, get a model back.** No estimator, no configuration: mixle infers the model and fits it.

```python
from mixle.inference import optimize

records = [                        # your rows: a number, a category, a flag — mixed, some missing
    (1.9, "paid", True), (0.4, "free", False), (2.1, "paid", True),
    (0.7, "free", False), (1.6, "paid", True), (0.3, "free", None),
]
model = optimize(records, out=None)   # mixle works out the model and fits it (out=None: quiet)

model.log_density(records[0])    # score an observation
model.sampler().sample(5)        # draw new ones
```

**Distill a slow, expensive model into a cheap one that knows when to defer.** Point `solve` at the
function doing the job today — an LLM, an API, a rule — and it trains a small local model that answers the
easy cases itself and escalates only the hard ones.

```python
from mixle.task import solve

# teacher = the function doing the job now; inputs = representative examples
assistant = solve(teacher, inputs)   # `teacher` labels once; a small model learns from it

assistant(x)            # answers locally when confident, calls `teacher` only when it is not
assistant.report()      # how often it matched the teacher, and how much it deferred
assistant.save("assistant/")
```

You pay the expensive model on only a fraction of requests. The same pattern distills classifiers,
extractors, and tool-callers, with conformal calibration, cascades, and a cost model underneath
(`mixle.task`).

**A PyTorch module fits in one line — the training code you did not write.** Any module exposing
`log_density(x)` fits with one call: no loop, no batching, no eval or convergence boilerplate.

```python
from mixle.inference import optimize

model = optimize(x, my_module)   # your nn.Module — trained
model.module                     # the raw module back, nothing trapped
```

Freeze submodules, swap the optimizer, or distribute the fit with `backend=`; parity with a hand-written
training loop is checked by a test, not claimed here.

**Compose to any depth; one call fits the whole thing.** You hand it estimators, not fitted
distributions — you don't know the parameters yet, and that's the point. Nest them and a single
`optimize` learns every level together: here, a hidden Markov model whose two states emit through
different learned models — a Gaussian mixture, and the neural density from above.

```python
from mixle.stats import GaussianEstimator, MixtureEstimator, HiddenMarkovEstimator
from mixle.models import GradEstimator

# sequences: a list of observation series; nothing below fixes a parameter
model = optimize(sequences, HiddenMarkovEstimator([
    MixtureEstimator([GaussianEstimator()] * 5),  # one state: a five-cluster mixture
    GradEstimator(my_module),                     # the other: a neural density
]))
```

One call, each part fit by the right M-step: Baum-Welch for the Markov dynamics, EM for the mixture
inside a state, gradient descent for the neural leaf. Every node is an estimator, so the tree nests as
deep as the model does — the call at the top never changes.

## Engines & orchestration

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
ops, device, and precision — so **scale-out is a backend argument, not a rewrite**:

```python
from mixle.engines import TorchEngine

optimize(..., engine=TorchEngine(device="cuda", dtype="float32"))  # GPU: one arg
optimize(..., precision="auto")   # mixed precision; stats accumulate in float64
optimize(..., backend="spark")    # distributed: mp · dask · mpi · ray · lightning
```

- The same fit runs unchanged on NumPy, Numba, Torch, or a symbolic backend.
- New frameworks register a factory (`register_encoded_data_backend`) — no dispatch to edit.
- The planner (`mixle.utils.parallel.planner`) turns a hardware budget into a memory-aware placement
  (chunking, device assignment, Torch sharding) you compute once and reuse.
- The `SymbolicEngine` runs a density through SymPy, so a model can emit its closed-form log-density
  as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded. This
works on a real neural LM and on a model you just fit.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mixle.enumeration import AutoregressiveEnumerable

name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).eval()
prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids

@torch.no_grad()
def next_logprobs(continuation):   # tokens so far -> [(token_id, log_prob), ...]
    ids = (torch.cat([prompt, torch.tensor([continuation])], 1)
           if continuation else prompt)
    return list(enumerate(torch.log_softmax(llm(ids).logits[0, -1], -1).tolist()))

# branch_cap tames the 49K-token vocab
continuations = AutoregressiveEnumerable(next_logprobs, max_len=3, branch_cap=8)
[tokenizer.decode(seq) for seq, _ in continuations.top_k(3)]
# -> [' located in the', ' the city of', ' the capital of']

# rank() inverts unrank(); cumulative_prob is exact, never approximated
continuations.rank(continuations.unrank(5)[0])   # -> rank=6, cumulative_prob=0.114
```

The same operations work on a model you just fit — here an HMM learns *when to stop* from an absorbing
terminal state, and its EOL-terminated support enumerates in descending probability:

```python
from mixle.inference import optimize
from mixle.stats import HiddenMarkovEstimator, CategoricalEstimator

# each sequence ends in an EOL token
sequences = [["team", "meet", "buy", "<EOL>"], ["now", "now", "<EOL>"], ...]
model = optimize(sequences,
    HiddenMarkovEstimator([CategoricalEstimator()] * 3, terminal_states={2}))

# most probable EOL-terminated sequences
model.enumerator().top_k(3)   # -> [(['buy', '<EOL>'], -2.09), ...]
```

- **Decomposable families** (Composite / Record / Sequence / MarkovChain): rank ↔ value is an exact
  count-DP at any depth; budget-bounded quantized indexes seek any rank of an infinite support directly,
  without materializing what comes before it.
- **Non-decomposable families** (mixtures, HMMs): exact marginal rank is provably hard, so they return
  the Viterbi bound or a certified Monte-Carlo estimate — never a silent approximation.
- **Continuous families** realize the same operations through `cdf(x)` / `quantile(q)`.

## Probabilistic programming (`mixle.ppl`)

A concise dialect over the same distributions. **One rule:** any parameter slot is a value, the token
`free` (estimate it), another distribution (a prior), or an expression over latents and data columns.

```python
from mixle.ppl import Normal, Mix, Markov, Field, free

data = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1]           # reals from two clusters
seqs = [[0.1, 5.1, 4.9], [4.8, 5.0], [0.0, 0.2]]   # variable-length sequences

Normal(free, free).fit(data)                             # estimate mean + standard deviation
Normal(Normal(0, 10), 1.0).fit(data)                      # a prior on the mean
Mix([Normal(free, free), Normal(free, free)]).fit(data)   # two-cluster mixture
Markov(Normal(free, free), states=2).fit(seqs)            # a 2-state Gaussian HMM
Normal(free * Field("x") + free * Field("z") + free, free).fit(
    ..., given={"x": ..., "z": ...})   # a regression
```

- **`how=`** picks the inference route from the model's structure (`conjugate | em | map | laplace | vi |
  mcmc | nuts | …`); `m.explain_fit()` reports the choice and why.
- **Hierarchies & GLMMs:** `.each(by=...)` and `Group(...)` are random effects; `potential(fn, *vars)` adds
  a custom log-factor; constraints (`a < b`) shape inference and sampling.
- **Neural densities:** `Flow`, `MDN`, `VAE` fit with `.fit()` and compose into mixtures like any distribution.
- **Diagnostics:** multi-chain fits fold R̂ / ESS into `m.result.summary()`; `waic` / `loo` / `compare` rank
  fitted models.

## Package highlights

- **Just pass data** — `optimize(data)` and `mixle.propose(data)` work out a model for you: they pick a
  family per field, notice when fields depend on each other, and (new in 0.7.0) fit a copula or vine when
  continuous columns are correlated — heavy joint tails included.
- **A torch module, fit for you** (`mixle.models`) — any module with a forward pass and an objective fits
  in one call (`optimize(x, module)`); freeze submodules, swap the optimizer, and get the raw module back —
  nothing is trapped. Parity with a hand-written loop is pinned by a test.
- **Distill big models into small ones, and route by cost** (`mixle.task`) — turn a slow, expensive teacher
  (an LLM, a rule, a human) into a tiny local model; a **conformal gate** decides when the local answer is
  safe, a **cascade / router** escalates only the hard cases to the teacher, and a **cost model** reports
  the dollars saved and the break-even volume so the trade-off is a number, not a guess. Soft-label
  distillation, density-gated and cost-aware routing thresholds, harvest-and-re-distill loops, and multi-tier
  routing with bandits / RL for the decision policy are all included.
- **Build by nesting** — mixtures, sequences, records, HMMs (segmental / lookback / tree / quantized), LDA,
  PCFGs, and more compose to any depth, with parameters tied across the structure by `keys=`; one call fits
  the whole tree.
- **A large catalog of building blocks** — around 90 families (continuous, discrete, directional,
  multivariate), copulas and vines for dependence, permutations and graphs (Mallows, matchings, spanning
  trees, grammars), and neural leaves: a Transformer LM, energy models, and constrained networks
  (physics-informed, monotonic, input-convex, conservation-preserving, permutation-invariant).
- **Scale with a flag** — the same fit runs local (NumPy / Numba / GPU) or distributed
  (Spark / Dask / Ray / MPI) via `backend=`; the Transformer LM trains sharded (FSDP2) under torchrun.
- **When you want the full picture** — exact enumeration and rank / quantile queries over discrete and
  structured models, MCMC and variational inference for posteriors (a `prior=` is the only switch from
  point estimate to posterior), design of experiments with Bayesian optimization, cross-modal
  representations, and reproducible artifacts with a serving gateway
  ([mixle-mlops](https://github.com/gmboquet/mixle-mlops)).

## Companion projects

The core library stands alone; three sibling projects build on it:

- **[mixle-notebooks](https://github.com/gmboquet/mixle-notebooks)** — runnable tutorials, data-science
  recipes, applied case studies, and architecture/scaling studies.
- **[mixle-mlops](https://github.com/gmboquet/mixle-mlops)** — an OpenAI-compatible gateway that serves
  fitted mixle models alongside open and hosted LLMs, with fine-tuning, registries, and monitoring.
- **[mixle-pde](https://github.com/gmboquet/mixle-pde)** — a differentiable PDE / physics stack
  (`Differential`, `make_ops`, `laplacian`, `NavierStokes2D`) for scientific inverse problems.

## Examples

Self-contained scripts in [examples/](https://github.com/gmboquet/mixle/tree/main/examples)
— each samples from a known model, refits, and recovers it (no downloads):

```sh
cd examples
python gallery_univariate_example.py   # scalar families (+ multivariate, …)
python gallery_structured_example.py   # mixtures / HMMs / LDA / latent models
python ppl_example.py                  # the equation-style mixle.ppl surface
python production_example.py           # provenance, registry, serving, drift
python scaling_example.py              # same fit by backend= (mp / mpi / spark)
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
[`mixle/tests/README.md`](https://github.com/gmboquet/mixle/blob/main/mixle/tests/README.md).

## Maintainers & contributors

Maintained by **Grant Boquet** ([@gmboquet](https://github.com/gmboquet) ·
grant.boquet@gmail.com).

Contributions, issues, and discussion are welcome — open a PR or an issue.

## License

MIT — see [LICENSE](https://github.com/gmboquet/mixle/blob/main/LICENSE).

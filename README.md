<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gmboquet/mixle/v0.8.0/assets/mixle_logo_dark.png"/>
    <img src="https://raw.githubusercontent.com/gmboquet/mixle/v0.8.0/assets/mixle_logo.png" alt="mixle" width="480"/>
  </picture>
</p>

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-15%2C000%2B-brightgreen)
[![docs](https://img.shields.io/badge/docs-gmboquet.github.io%2Fmixle-blue)](https://gmboquet.github.io/mixle/)

**mixle fits models of real, messy data — numbers, text, categories, sequences, missing values — with
one call.** Hand it raw data and it proposes and fits a model; hand it a structure and it fits that.

One idea holds the library together: a classical distribution, a neural network, and a latent-variable
model are **the same kind of object**. They nest inside each other freely, and a single `optimize(...)`
fits whatever you built — closed form where a part has one, gradient descent for a neural leaf, EM for
latent structure, all in one loop. Underneath sits a full probabilistic-modeling stack: ~90
distribution families, mixtures and hidden Markov models, automatic model selection, Bayesian inference
from conjugate updates to NUTS, design-of-experiments optimization, and calibrated, monitored
deployment.

**Three reasons people reach for it:**

- **Less code.** No training loops, no batching or convergence boilerplate, no glue — point `optimize`
  at your data or your PyTorch module and it does the fitting.
- **Lower cost.** Distill a slow, expensive teacher — a frontier LLM, an API, a rule — into a tiny
  local model that answers the easy cases itself and escalates only the hard ones.
- **Honest uncertainty.** Models can be calibrated to report when they are unsure and defer rather than
  guess, so an application can route low-confidence cases to a human or a stronger model.

mixle removes the boilerplate, not the modeling judgment: you still choose the data, the objective,
and the validation. Not every surface is equally settled, either — the
**[maturity guide](https://gmboquet.github.io/mixle/maturity.html)** separates the stable center
(distributions, estimators, `optimize`) from the provisional workflow layers, and headline claims trace
to it and to the [release-readiness](https://gmboquet.github.io/mixle/release-readiness.html) and
[validation](https://gmboquet.github.io/mixle/validation.html) evidence.

**Docs:** [gmboquet.github.io/mixle](https://gmboquet.github.io/mixle/) · **Release notes:**
[CHANGELOG.md](CHANGELOG.md)

## Contents

[Installation](#installation) · [Quickstart](#quickstart) · [Engines & scale](#engines--scale) ·
[Enumeration & ranking](#enumeration--ranking) · [Probabilistic programming](#probabilistic-programming-mixleppl) ·
[Beyond the basics](#beyond-the-basics) · [Related projects](#related-projects) · [Examples](#examples) ·
[Tests](#tests) · [Maintainers & contributors](#maintainers--contributors) · [License](#license)

## Installation

Python 3.11 or 3.12 (developed on 3.12), on PyPI as `mixle`. That is the whole supported range —
`requires-python` is `>=3.11,<3.13`, so 3.13 and later will not install. CI tests Linux x86_64 and
macOS arm64 (Apple Silicon) on every PR; Windows is untested.

```sh
pip install mixle          # base (numpy, scipy): ordinary distributions and local fitting
pip install "mixle[all]"   # + numba, torch, the distributed backends, and core data connectors
```

Everything past the base is opt-in — install any subset, e.g. `pip install "mixle[torch,spark]"`:

- **Acceleration** — `numba` (JIT hot paths, falls back to NumPy) · `torch` (GPU / autograd) · `jax` (JAX engine + NUTS)
- **Scale-out** — `spark` · `dask` · `ray` · `lightning` · `mpi`
- **Data sources** — `pandas` · `arrow` · `sql` · `mongo` · `hadoop` · `arrays`
- **Other** — `highprec` (mpmath fallback) · `gmpy2` (fast exact ranking) · `umap` · `sympy` / `sage` (symbolic export) · `grammar` (graph grammars)

`[all]` covers `numba`, `torch`, the scale-out backends, and `pandas`/`arrow`/`sql`; `jax`, `gmpy2`,
`sympy`/`sage`, and `mongo`/`hadoop`/`arrays` install separately. Every declared floor installs and
imports — CI pins each extra to its minimum and proves it.

Development: `git clone … && pip install -e ".[all]"`.

## Quickstart

**Hand it data, get a model back.** With no estimator argument, mixle infers a starting model and fits
it — a first pass to inspect and refine.

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
function doing the job today — an LLM, an API, a rule — and it trains a small local model that answers
the easy cases itself and escalates only the hard ones.

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
(`mixle.task`). mixle fits and routes these small models; it does not train frontier models.

**A PyTorch module fits in one line — the training code you did not write.** Any module exposing
`log_density(x)` fits with one call: no loop, no batching, no eval or convergence boilerplate.

```python
from mixle.inference import optimize

model = optimize(x, my_module)   # your nn.Module — trained
model.module                     # the raw module back, nothing trapped
```

Freeze submodules, swap the optimizer, or distribute the fit with `backend=`; parity with a
hand-written training loop is pinned by a test, not claimed here.

**Compose to any depth; one call fits the whole thing.** You hand it estimators, not fitted
distributions — you don't know the parameters yet, and that's the point. Nest them and a single
`optimize` learns every level together: here, a hidden Markov model whose two states emit through
different learned models — a Gaussian mixture, and the neural density from above.

```python
from mixle.inference import optimize
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

## Engines & scale

Distributions own the likelihood and sufficient-statistic math; **compute engines** supply the array
ops, device, and precision. Scale-out is usually a backend argument rather than a rewrite, within the
engines and backends mixle supports:

```python
from mixle.engines import TorchEngine

optimize(..., engine=TorchEngine(device="cuda", dtype="float32"))  # GPU: one arg
optimize(..., precision="auto")   # mixed precision; stats accumulate in float64
optimize(..., backend="spark")    # distributed: mp · dask · mpi · ray · lightning
```

> **`device=` changes the answer, not just the speed.** A fit on a GPU device does not reproduce the
> same fit on CPU: measured on Metal, per-observation log-density diverges ~2% relative after a
> single optimizer iteration, because the device backends use different float32 kernels. Pin the
> device alongside the seed when a result has to reproduce, and see
> [backend support](docs/backend-support.rst) for which backends are CI-gated — CUDA is
> hardware-gated: executed once for 0.8.0 on rented hardware
> ([receipt](release-checklists/0.8.0-cuda-receipt.json)), not gated in CI.

New frameworks register a factory (`register_encoded_data_backend`) — no dispatch to edit. The planner
(`mixle.utils.parallel.planner`) turns a hardware budget into a memory-aware placement you compute once
and reuse, and the `SymbolicEngine` runs a density through SymPy so a model can emit its closed-form
log-density as LaTeX / SymPy / Sage.

## Enumeration & ranking

Discrete and structured models **enumerate their support in descending-probability order** and answer
exact **rank / cumulative-probability** queries — even when the support is enormous or unbounded. It
works on a real neural LM:

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

The same operations work on a model you just fit — an HMM with an absorbing terminal state enumerates
its EOL-terminated support in descending probability, and `enumerator()` refuses with an
`EnumerationError` rather than enumerate a different distribution when the fit has not earned it.

- **Decomposable families** (Composite / Record / Sequence / MarkovChain): rank ↔ value is an exact
  count-DP at any depth; budget-bounded quantized indexes seek any rank of an infinite support directly.
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
- **Hierarchies & GLMMs:** `.each(by=...)` and `Group(...)` are random effects; `potential(fn, *vars)`
  adds a custom log-factor; constraints (`a < b`) shape inference and sampling.
- **Neural densities:** `Flow`, `MDN`, `VAE` fit with `.fit()` and compose into mixtures like any
  distribution.
- **Diagnostics:** multi-chain fits fold R̂ / ESS into `m.result.summary()`; `waic` / `loo` / `compare`
  rank fitted models.

## Beyond the basics

- **Automatic structure** — `mixle.propose(data)` picks a family per field, notices when fields depend
  on each other, and fits a copula or vine when continuous columns are correlated — heavy joint tails
  included.
- **A deep catalog** — continuous, discrete, directional, and multivariate families; copulas and vines;
  permutations and graphs (Mallows, matchings, spanning trees, grammars); structured latent models
  (segmental / lookback / tree / quantized HMMs, LDA, PCFGs); and neural leaves — a Transformer LM,
  energy models, and constrained networks (physics-informed, monotonic, input-convex,
  conservation-preserving, permutation-invariant). Parameters tie across any structure with `keys=`.
- **Posteriors when you want them** — a `prior=` is the only switch from point estimate to posterior;
  MCMC and variational routes share the same model objects.
- **Design of experiments** — Bayesian optimization with failure-aware ledgers, constraint geometry,
  and information-gain acquisition (`mixle.doe`).
- **Serious scale-out** — sufficient-statistic fits over Spark / Dask / Ray / MPI; packed Transformer
  training with an explicit PyTorch DeviceMesh backend (DDP, HSDP/FSDP2, MLP TP, CUDA CP) and a
  Megatron Bridge adapter for full TP / PP / CP / EP / ETP. Unsupported combinations fail before
  fitting.
- **Production surfaces** — reproducible artifacts, registries, drift checks, and a serving gateway
  ([mixle-mlops](https://github.com/gmboquet/mixle-mlops)).

## Related projects

Mixle Core 0.8.0 is a standalone release. The independently versioned projects below are related
development efforts, not members of the 0.8.0 artifact set, and this release makes no co-installation
or compatibility claim for them:

- **[mixle-notebooks](https://github.com/gmboquet/mixle-notebooks)** — runnable tutorials, data-science
  recipes, and applied case studies, including every demonstration on real public datasets (the core
  repository itself carries no dataset downloads).
- **[mixle-mlops](https://github.com/gmboquet/mixle-mlops)** — an OpenAI-compatible gateway that serves
  fitted mixle models alongside open and hosted LLMs, with fine-tuning, registries, and monitoring.
- **[mixle-pde](https://github.com/gmboquet/mixle-pde)** — a differentiable PDE / physics stack for
  scientific inverse problems.
- **[mixle-discrete](https://github.com/gmboquet/mixle-discrete)** — integer/binary least squares,
  finite-field arithmetic, and lattice cryptography.
- **[mixle-agent](https://github.com/gmboquet/mixle-agent)** — an open agent system (CLI, web GUI,
  desktop) with skills, MCP tool interop, and pluggable model providers.
- **[mixle-demos](https://github.com/gmboquet/mixle-demos)** — standalone end-to-end demonstration
  harnesses exercising the ecosystem against synthetic truth.

## Examples

Every example in this repository runs on synthetic data it generates itself — none downloads a
dataset. The five below are additionally dependency-free beyond the base install, in the
[version-bound examples directory](https://github.com/gmboquet/mixle/tree/v0.8.0/examples):

```sh
cd examples
python gallery_univariate_example.py   # scalar families (+ multivariate, …)
python gallery_structured_example.py   # mixtures / HMMs / LDA / latent models
python ppl_example.py                  # the equation-style mixle.ppl surface
python production_example.py           # provenance, registry, serving, drift
python scaling_example.py              # same fit by backend= (mp / mpi / spark)
```

A few examples load pinned open model weights (CLIP, a small LM) on first run; each pin, its cache
behavior, and its release-evidence status are recorded in the
[example execution manifest](https://gmboquet.github.io/mixle/0.8.0/example-execution-manifest.html).
Real-dataset walkthroughs live in
[mixle-notebooks](https://github.com/gmboquet/mixle-notebooks).

## Tests

15,000+ tests, organized into purpose/time-budget tiers (see the
[test tiers guide](https://gmboquet.github.io/mixle/test-tiers.html)):

```sh
python -m pytest path/to/focused_test.py    # the smallest relevant test while developing
python -m pytest                            # the fast local gate (~3 min on a laptop)
python -m pytest -m full -m ""              # everything non-optional (~15 min at -n auto)
```

Hosted CI runs the same tiers sharded across runners — core, full (four shards plus a combined
coverage floor), optional extras, and scheduled numerical/hardware lanes.
`base_dist_test.py` alone exercises 40 of its 41 base-distribution families end to end: sampler
repeatability, `str`/`eval` round-trips, vectorized-vs-scalar density agreement, EM convergence.
See [`mixle/tests/README.md`](https://github.com/gmboquet/mixle/blob/v0.8.0/mixle/tests/README.md).

## Maintainers & contributors

Maintained by **Grant Boquet** ([@gmboquet](https://github.com/gmboquet) ·
grant.boquet@gmail.com).

Contributions, issues, and discussion are welcome — open a PR or an issue.

## License

MIT — see [LICENSE](https://github.com/gmboquet/mixle/blob/v0.8.0/LICENSE).
